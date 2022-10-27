import { logger, LoggerInterface, LoggerOptions } from '@voqse/logger';
import { Candle } from '@debut/types';
import { file } from '@debut/plugin-utils';
import { DistributionSegment, getDistribution, getPredictPrices, getQuoteRatioData, RatioCandle } from './utils';
import { Forecast } from './index';
import tfn, { Sequential, Tensor, GRULayerArgs } from '@tensorflow/tfjs-node';
import path from 'path';

/**
 * TODO: 1. Refactor dataset into readable array of objects
 *       2. Make batch standardisation function
 *       3. Pass whole candle data as features
 */

let tf: typeof tfn;
let log: LoggerInterface;

type NormCandle = Partial<Candle>;
type Range = {
    min: number;
    max: number;
};
type CandleRange = {
    p: Range;
    v: Range;
};
type NormBundle = { candles: NormCandle[]; range: CandleRange };

export interface Layer extends Partial<GRULayerArgs> {
    type: 'dense' | 'gru';
    units: number;
}

export type Dataset = { input: number[][]; output: number[] };

export interface ModelOptions extends LoggerOptions {
    gpu?: boolean;
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
     * Number of output candles.
     *
     * Defaults to 3.
     */
    outputSize?: number;
    /**
     * Size of validation set.
     *
     * Defaults to 10.
     */
    validationSize?: number;
    /**
     * Number of epochs with no improvement after which training will be stopped.
     *
     * Defaults to 100.
     */
    earlyStop?: number;
    /**
     * Array of positive integers, defines neurons count in each hidden layer.
     */
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
    batchSize?: number;
    epochs?: number;
    dropoutRate?: number;
}

export class Model {
    private readonly opts: ModelOptions;
    private model: Sequential;
    private pretrainedModel: Sequential;
    private candlesData: Candle[][] = [];
    private trainingSet: Dataset[] = [];
    private validationSet: Dataset[] = [];
    private distribution: DistributionSegment[][] = [];
    private prevCandle: Candle[] = [];
    private input: number[][] = [];
    private callback = [];
    private lastRange = [];

    constructor(opts: ModelOptions) {
        const defaultOpts: Partial<ModelOptions> = {
            segmentsCount: 11,
            inputSize: 20,
            earlyStop: 10,
            freezeLayers: true,
            outputSize: 3,
        };

        this.opts = { ...defaultOpts, ...opts };
        log = logger('cortex/model', this.opts);

        if (this.opts.gpu) {
            log.info('Using GPU for training');
            tf = require('@tensorflow/tfjs-node-gpu');
        } else {
            log.info('Using CPU for training');
            tf = require('@tensorflow/tfjs-node');
        }
    }

    private createModel(opts: Partial<ModelOptions>): typeof this.model {
        const { inputSize, outputSize, layers, additionalLayers, dropoutRate } = opts;
        const model = tf.sequential();

        if (this.pretrainedModel) {
            const layersCount = this.pretrainedModel.layers.length;

            if (!additionalLayers?.length) {
                return this.pretrainedModel;
            }
            log.info('Additional layers provided. Inserting...');

            const outputLayer = this.pretrainedModel.getLayer(undefined, layersCount - 2);
            const shavedModel = tf.model({
                inputs: this.pretrainedModel.inputs,
                outputs: outputLayer.output,
            }) as Sequential;

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
            const randomSeed = () =>
                Math.round(Math.random() * 100)
                    .toString()
                    .padStart(3, '0');

            const [inputLayer, ...hiddenLayers] = layers;
            const { type, ...args } = inputLayer;

            model.add(
                tf.layers[type]({
                    inputShape: [inputSize, this.candlesData.length * 4],
                    name: `input`,
                    ...args,
                }),
            );

            hiddenLayers.forEach((layer, index) => {
                const { type, ...args } = layer;

                model.add(
                    tf.layers[type]({
                        name: `hidden-${randomSeed()}-${index}`,
                        ...args,
                    }),
                );
            });
        }

        if (dropoutRate) {
            model.add(tf.layers.dropout({ name: 'dropout', rate: dropoutRate }));
        }

        model.add(tf.layers.dense({ name: 'output', units: outputSize }));

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
            if (!this.candlesData[index]) this.candlesData[index] = [];
            this.candlesData[index].push(candle);
        });
    }

    serveTrainingData(): void {
        log.info('Preparing training data...');
        const { inputSize, outputSize, validationSize } = this.opts;

        this.fillTrainingSet(inputSize, outputSize);

        if (!validationSize) return;
        this.fillValidationSet(validationSize);
    }

    private fillTrainingSet(inputSize: number, outputSize: number): void {
        const [mainCandleSet, ...corrCandleSets] = this.candlesData;

        for (let start = 0; start < mainCandleSet.length - inputSize - outputSize; start++) {
            const end = start + inputSize;
            const flat = ({ h, l, c, v }: NormCandle) => [h, l, c, v];

            const { candles: normInputCandles, range } = this.scaleCandleSet(mainCandleSet.slice(start, end));
            const normCorrCandles = corrCandleSets.map((candleSet) => {
                const bundle = candleSet.slice(start, end);
                return this.scaleCandleSet(bundle).candles;
            });

            // const input = [flat(normInputCandles), ...flat(normCorrCandles)];
            const input = normInputCandles.map((inputCandle, index) => {
                return [...flat(inputCandle), ...flat(normCorrCandles[0][index])];
            });
            const output = mainCandleSet
                .slice(end, end + outputSize)
                .map((candle) => this.scaleCandle(candle, range).c);

            this.trainingSet.push({ input, output });

            // log.debug(`Input (${input.length}, ${input[0].length}):\n`, input, `\nOutput (${output.length})\n`, output);
        }
    }

    private fillValidationSet(validationSize: number): void {
        for (let i = 0; i < validationSize; i++) {
            const index = Math.floor(Math.random() * this.trainingSet.length);
            const [item] = this.trainingSet.splice(index, 1);

            this.validationSet.push(item);
        }
    }

    private scaleCandle({ o, h, l, c, v }: Candle, { p, v: vl }: CandleRange) {
        // Clamp function for prediction normalization
        // needed due output can go outside of window range
        const clamp = (num) => Math.min(Math.max(num, 0), 1);
        const scale = (num, min, max) => clamp((num - min) / (max - min));

        return {
            o: scale(o, p.min, p.max),
            h: scale(h, p.min, p.max),
            l: scale(l, p.min, p.max),
            c: scale(c, p.min, p.max),
            v: scale(v, vl.min, vl.max),
        };
    }

    private scaleCandleSet(candles: Candle[]): NormBundle {
        const hs = candles.map(({ h }) => h);
        const ls = candles.map(({ l }) => l);
        const vs = candles.map(({ v }) => v);

        const range = {
            p: {
                min: Math.min(...ls),
                max: Math.max(...hs),
            },
            v: {
                min: Math.min(...vs),
                max: Math.max(...vs),
            },
        };

        return {
            candles: candles.map((candle) => this.scaleCandle(candle, range)),
            range,
        };
    }

    private convertToTensor(data: Dataset[]) {
        if (!data || !data.length) return;

        return tf.tidy(() => {
            // tf.util.shuffle(data);

            const inputData = data.map((d) => d.input);
            const outputData = data.map((d) => d.output);

            const inputTensor = tf.tensor3d(inputData);
            const outputTensor = tf.tensor2d(outputData);

            return [inputTensor, outputTensor];
        });
    }

    private getInput(candles: Candle[]): typeof this.candlesData {
        const input = [...this.candlesData];
        // const timestep = [];

        candles.forEach((candle, index) => {
            if (!input[index]) input[index] = [];
            input[index].push(candle);

            if (input[index].length > this.opts.inputSize) {
                input[index].shift();
            }
        });

        // if (timestep.length) {
        //     input.push(timestep);
        //
        //     if (input.length > this.opts.inputSize) {
        //         input.shift();
        //     }
        // }

        return input;
    }

    private getForecast(input: typeof this.candlesData, candles: Candle[]): Forecast[] | undefined {
        if (input[0].length === this.opts.inputSize) {
            // TODO: prepareInput affects inner value. Do not use for momentValue
            const normalized = [];

            this.candlesData.forEach((candleSet, index) => {
                const { candles, range } = this.scaleCandleSet(candleSet);
                this.lastRange[index] = range;

                candles.forEach(({ h, l, c, v }, candleIndex) => {
                    if (!normalized[candleIndex]) normalized[candleIndex] = [];
                    normalized[candleIndex].push(h, l, c, v);
                });
            });

            // console.log(normalized);

            return tf.tidy(() => {
                const inputTensor = tf.tensor3d([normalized], [1, this.opts.inputSize, this.candlesData.length * 4]);
                const forecast = this.pretrainedModel.predict(inputTensor) as Tensor;
                const normForecast = Array.from(forecast.dataSync());

                const { min, max } = this.lastRange[0].p;
                return normForecast.map((item) => item * (max - min) + min);
            });
        }
    }

    /**
     * Get forecast at the moment
     */
    momentValue(candles: Candle[]) {
        const input = this.getInput(candles);

        return this.getForecast(input, candles);
    }

    /**
     * Run forecast
     */
    nextValue(candles: Candle[]) {
        this.candlesData = this.getInput(candles);
        // console.log('Data', this.candlesData[0].length);

        return this.getForecast(this.candlesData, candles);
    }

    async saveModel() {
        const { saveDir } = this.opts;
        // const groupsFile = 'groups.json';
        //
        // file.ensureFile(path.resolve(saveDir, groupsFile));
        // file.saveFile(path.resolve(saveDir, groupsFile), this.distribution);

        await this.model.save(`file://${path.resolve(saveDir)}`);
    }

    async loadModel() {
        const { loadDir } = this.opts;
        // const groupsFile = 'groups.json';
        // const groupsData = file.readFile(path.resolve(loadDir, groupsFile));
        //
        // if (!groupsData) {
        //     throw `Unknown data in ${groupsFile}, or file does not exists, please run training before use`;
        //     // process.exit(0);
        // }
        //
        // this.distribution = JSON.parse(groupsData);
        log.info('Looking for existing model...');

        try {
            this.pretrainedModel = <Sequential>await tf.loadLayersModel(`file://${path.resolve(loadDir)}/model.json`);
            log.debug('Existing model loaded');
        } catch (e) {
            log.info('No model found. Creating new one...');
        }
    }

    async training() {
        log.debug('Training set length:', this.trainingSet.length);
        await this.loadModel();

        this.model = this.createModel(this.opts);
        this.model.summary();

        // Save model every epoch. Upd: this will not work
        // just as example.
        // this.callback.push(
        //     new tf.CustomCallback({
        //         onEpochEnd: this.saveModel,
        //     }),
        // );

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
                    monitor: this.opts.validationSize ? 'val_acc' : 'acc',
                    patience: this.opts.earlyStop,
                }),
            );
        }

        this.model.compile({
            optimizer: tf.train.adam(),
            loss: tf.losses.absoluteDifference,
            metrics: ['accuracy'],
        });

        log.info('Starting training...');
        const { batchSize, epochs } = this.opts;
        const [inputs, outputs] = this.convertToTensor(this.trainingSet);
        const validationData = this.convertToTensor(this.validationSet);

        const { history } = await this.model.fit(inputs, outputs, {
            batchSize,
            epochs,
            // shuffle: true,
            callbacks: this.callback,
            // @ts-ignore
            validationData,
        });
        log.info('Training finished with accuracy:', history.acc.pop());
    }

    // private normalize(groupId: number) {
    //     return groupId / this.opts.segmentsCount;
    // }
    //
    // private denormalize(value: number) {
    //     return Math.min(Math.round(value * this.opts.segmentsCount), this.opts.segmentsCount - 1);
    // }
}
