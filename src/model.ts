import { logger, LoggerInterface, LoggerOptions } from '@voqse/logger';
import { Candle } from '@debut/types';
import { file } from '@debut/plugin-utils';
import { DistributionSegment, getDistribution, getPredictPrices, getQuoteRatioData, RatioCandle } from './utils';
import { CortexForecast } from './index';
import * as tf from '@tensorflow/tfjs-node';
import '@tensorflow/tfjs-backend-cpu';
import path from 'path';

let log: LoggerInterface;

export interface Layer {
    type: 'dense' | 'gru';
    units: number;
}

export interface ModelOptions extends LoggerOptions {
    saveDir: string;
    loadDir: string;

    /**
     * Logging directory for tensorboard.
     */
    logDir?: string;
    segmentsCount?: number;
    /**
     * Size of window in candles for prediction.
     *
     * Defaults to 20.
     */
    inputSize?: number;
    /**
     * Number of candles at one time step.
     *
     * Defaults to 1.
     */
    // features?: number;
    /**
     * Number of epochs with no improvement after which training will be stopped.
     *
     * Defaults to 100.
     */
    earlyStop?: number;
    /**
     * Array of positive integers, defines neurons count in each hidden layer.
     *
     * Defaults to [32, 16, 8].
     */
    rnnLayers?: number[];
    nnLayers?: number[];
    layers?: Layer[];
    /**
     * Array of positive integers, defines layers to add to existing model in next training.
     *
     * Defaults to undefined.
     */
    additionalLayers?: number[];
    /**
     * Weather or not original layers will be frozen while new training. Works
     * only with pretrained model and extra hiddenLayers provided.
     *
     * Defaults to true.
     */
    freezeLayers?: boolean;
    /**
     * Number of output candles.
     *
     * Defaults to 3.
     */
    outputSize?: number;
    batchSize?: number;
    epochs?: number;
    dropoutRate?: number;
}

export class Model {
    private opts: ModelOptions;
    private model: tf.Sequential | tf.LayersModel;
    private pretrainedModel: tf.Sequential;
    private datasets: RatioCandle[][] = [];
    private trainingSet: { input: number[][]; output: number[] }[] = [];
    private distribution: DistributionSegment[][] = [];
    private prevCandle: Candle[] = [];
    private input: number[][] = [];
    private callback = [];

    constructor(opts: ModelOptions) {
        const defaultOpts: Partial<ModelOptions> = {
            segmentsCount: 11,
            inputSize: 20,
            // features: 1,
            earlyStop: 10,
            rnnLayers: [32, 16, 8],
            freezeLayers: true,
            outputSize: 3,
        };

        this.opts = { ...defaultOpts, ...opts };
        log = logger('cortex/model', this.opts);

        if (this.opts.logDir) {
            log.info(
                `Use the command below to bring up tensorboard server:`,
                `\n  tensorboard --logdir ${this.opts.logDir}`,
            );

            this.callback.push(
                tf.node.tensorBoard(this.opts.logDir, {
                    updateFreq: 'epoch',
                }),
            );
        }

        if (this.opts.earlyStop) {
            this.callback.push(
                tf.callbacks.earlyStopping({
                    monitor: 'acc',
                    patience: this.opts.earlyStop,
                }),
            );
        }
    }

    private createModel(opts: Partial<ModelOptions>): typeof this.model {
        const { inputSize, outputSize, rnnLayers, nnLayers, additionalLayers, dropoutRate } = opts;
        const model = tf.sequential();

        if (this.pretrainedModel) {
            const layersCount = this.pretrainedModel.layers.length;

            if (!additionalLayers?.length) {
                return this.pretrainedModel;
            }
            log.info('Additional layers provided. Inserting...');

            const outputLayer = this.pretrainedModel.getLayer(undefined, layersCount - 2);
            const shavedModel = tf.model({ inputs: this.pretrainedModel.inputs, outputs: outputLayer.output });

            this.pretrainedModel.layers.forEach((layer) => {
                layer.trainable = !this.opts.freezeLayers;
            });

            model.add(shavedModel);
            additionalLayers?.forEach((units, index) => {
                model.add(
                    tf.layers.dense({
                        units,
                        activation: 'relu',
                        name: `hidden-${Math.round(Math.random() * 100)
                            .toString()
                            .padStart(3, '0')}-${index + layersCount - 2}`,
                    }),
                );
            });
        } else {
            const [inputUnits = inputSize, ...hiddenRnnLayers] = rnnLayers;
            const randomSeed = () =>
                Math.round(Math.random() * 100)
                    .toString()
                    .padStart(3, '0');

            model.add(
                tf.layers.gru({
                    inputShape: [inputSize, this.datasets.length],
                    units: inputUnits,
                    name: 'input',
                    returnSequences: !!hiddenRnnLayers.length,
                }),
            );

            hiddenRnnLayers?.forEach((units, index) => {
                model.add(
                    tf.layers.gru({
                        units,
                        // activation: 'relu',
                        name: `rnn-hidden-${randomSeed()}-${index}`,
                        returnSequences: index < hiddenRnnLayers.length - 1,
                    }),
                );
            });

            nnLayers?.forEach((units, index) => {
                model.add(
                    tf.layers.dense({
                        units,
                        activation: 'relu',
                        name: `nn-hidden-${randomSeed()}-${index}`,
                    }),
                );
            });
        }

        if (dropoutRate) {
            model.add(tf.layers.dropout({ rate: dropoutRate, name: 'dropout' }));
        }
        model.add(tf.layers.dense({ units: outputSize, name: 'output' }));
        return model;
    }

    /**
     * Accumulate data for future training.
     *
     * Right way is to provide all the features in one time step.
     * Desired input shape [inputSize, candles.length] ie [192, 2]
     *
     * @param candles
     */
    addTrainingData(candles: Candle[]): void {
        candles.forEach((candle, index) => {
            const ratioCandle = this.prevCandle[index] && getQuoteRatioData(candle, this.prevCandle[index]);

            if (ratioCandle) {
                if (!this.datasets[index]) this.datasets[index] = [];

                this.datasets[index].push(ratioCandle);
            }

            this.prevCandle[index] = candle;
        });
    }

    serveTrainingData(): void {
        log.info('Preparing training data...');

        const { inputSize, outputSize, segmentsCount } = this.opts;

        this.datasets.forEach((dataset, featureIndex) => {
            this.distribution[featureIndex] = getDistribution(dataset, segmentsCount);

            dataset.forEach((ratioCandle, timestepIndex) => {
                const groupId = this.distribution[featureIndex].findIndex(
                    (group) => ratioCandle.ratio >= group.ratioFrom && ratioCandle.ratio < group.ratioTo,
                );
                const normalisedGroupId = this.normalize(groupId);

                if (!this.input[timestepIndex]) this.input[timestepIndex] = [];
                this.input[timestepIndex].push(normalisedGroupId);
            });
        });

        for (let start = 0; start < this.input.length - inputSize - outputSize; start++) {
            const end = start + inputSize;

            const input = this.input.slice(start, end);
            const output = this.input.slice(end, end + outputSize).map((val) => val[0]); // Take only first feature for output value

            this.trainingSet.push({ input, output });

            // console.log('Input\n', input, 'Output\n', output);
        }
    }

    private convertToTensor(data: typeof this.trainingSet) {
        return tf.tidy(() => {
            // tf.util.shuffle(data);

            const inputData = data.map((d) => d.input);
            const outputData = data.map((d) => d.output);

            const inputs = tf.tensor3d(inputData);
            const outputs = tf.tensor2d(outputData);

            return { inputs, outputs };
        });
    }

    private getInput(candles: Candle[]): typeof this.input {
        const input = [...this.input];
        const timestep = [];

        candles.forEach((candle, featureIndex) => {
            const ratioCandle =
                this.prevCandle[featureIndex] && getQuoteRatioData(candle, this.prevCandle[featureIndex]);

            if (ratioCandle) {
                let groupId = this.distribution[featureIndex].findIndex(
                    (group) => ratioCandle.ratio >= group.ratioFrom && ratioCandle.ratio < group.ratioTo,
                );
                if (groupId === -1) {
                    groupId =
                        ratioCandle.ratio < this.distribution[featureIndex][0].ratioFrom
                            ? 0
                            : this.distribution[featureIndex].length - 1;
                }
                const normalisedGroupId = this.normalize(groupId);

                timestep.push(normalisedGroupId);
            }
        });

        if (timestep.length) {
            input.push(timestep);

            if (input.length > this.opts.inputSize) {
                input.shift();
            }
        }

        return input;
    }

    private getOutput(input: typeof this.input, candles: Candle[]): CortexForecast[] | undefined {
        if (input.length === this.opts.inputSize) {
            const forecast = tf.tidy(() => {
                const inputTensor = tf.tensor2d(input).expandDims(0);
                // console.log(input.length);
                const prediction = this.pretrainedModel.predict(inputTensor) as tf.Tensor;
                return Array.from(prediction.dataSync());
            });

            const output: CortexForecast[] = [];

            for (let i = 0; i < forecast.length; i++) {
                const cast = forecast[i];
                const denormalized = this.denormalize(cast);
                const group = this.distribution[0][denormalized];

                if (!group) return;

                output.push(getPredictPrices(candles[0].c, group.ratioFrom, group.ratioTo));
            }

            return output;
        }
    }

    /**
     * Get forecast at the moment
     */
    momentValue(candles: Candle[]) {
        const input = this.getInput(candles);

        return this.getOutput(input, candles);
    }

    /**
     * Run forecast
     */
    nextValue(candles: Candle[]) {
        this.input = this.getInput(candles);
        candles.forEach((candle, index) => {
            this.prevCandle[index] = candle;
        });

        return this.getOutput(this.input, candles);
    }

    async saveModel() {
        const { saveDir } = this.opts;
        const groupsFile = 'groups.json';

        file.ensureFile(path.resolve(saveDir, groupsFile));
        file.saveFile(path.resolve(saveDir, groupsFile), this.distribution);

        await this.model.save(`file://${path.resolve(saveDir)}`);
    }

    async loadModel() {
        const { loadDir } = this.opts;
        const groupsFile = 'groups.json';
        const groupsData = file.readFile(path.resolve(loadDir, groupsFile));

        if (!groupsData) {
            throw `Unknown data in ${groupsFile}, or file does not exists, please run training before use`;
            // process.exit(0);
        }

        this.distribution = JSON.parse(groupsData);
        this.pretrainedModel = <tf.Sequential>await tf.loadLayersModel(`file://${path.resolve(loadDir)}/model.json`);
    }

    async training() {
        log.debug('Training set length:', this.trainingSet.length);
        try {
            log.info('Looking for existing model...');
            await this.loadModel();
            log.debug('Existing model loaded');
        } catch (e) {
            log.info('No model found. Creating new one...');
        }

        this.model = this.createModel(this.opts);
        this.model.summary();
        this.model.compile({
            optimizer: tf.train.adam(),
            // TODO: Custom loss function
            loss: tf.losses.absoluteDifference,
            metrics: ['accuracy'],
        });

        log.info('Starting training...');
        const { batchSize, epochs } = this.opts;
        const { inputs, outputs } = this.convertToTensor(this.trainingSet);

        const { history } = await this.model.fit(inputs, outputs, {
            batchSize,
            epochs,
            // shuffle: true,
            callbacks: this.callback,
        });
        log.info('Training finished with accuracy:', history.acc.pop());
    }

    // private vectorize(index: number): number[] {
    //     const { segmentsCount } = this.opts;
    //     const vector = new Array(segmentsCount).fill(0);
    //
    //     vector[index] = 1;
    //     return vector;
    // }

    private normalize(groupId: number) {
        return groupId / this.opts.segmentsCount;
    }

    private denormalize(value: number) {
        return Math.min(Math.round(value * this.opts.segmentsCount), this.opts.segmentsCount - 1);
    }
}
