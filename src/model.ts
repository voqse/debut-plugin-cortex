import { logger, LoggerInterface, LoggerOptions } from '@voqse/logger';
import { Candle } from '@debut/types';
import { file } from '@debut/plugin-utils';
import { DistributionSegment, getDistribution, getPredictPrices, getQuoteRatioData, RatioCandle } from './utils';
import { CortexForecast } from './index';
import * as tf from '@tensorflow/tfjs-node';
import '@tensorflow/tfjs-backend-cpu';
import path from 'path';

let log: LoggerInterface;

export interface ModelOptions extends LoggerOptions {
    saveDir: string;
    loadDir: string;
    segmentsCount?: number;
    inputSize?: number;
    hiddenLayers?: number[];
    outputSize?: number;
    batchSize?: number;
    epochs?: number;
}

export class Model {
    private opts: ModelOptions;
    private model: tf.Sequential;
    private dataset: RatioCandle[][] = [];
    private trainingSet: { input: number[]; output: number[] }[] = [];
    private distribution: DistributionSegment[][] = [];
    private prevCandle: Candle[] = [];
    private input: number[][] = [];

    constructor(opts: ModelOptions) {
        const defaultOpts: Partial<ModelOptions> = {
            segmentsCount: 11,
            inputSize: 20,
            hiddenLayers: [32, 16, 8],
            outputSize: 3,
        };

        this.opts = { ...defaultOpts, ...opts };
        log = logger('cortex/model', this.opts);
    }

    private createModel(opts: Partial<ModelOptions>): typeof this.model {
        const { inputSize, outputSize, hiddenLayers } = opts;
        const [inputUnits = inputSize, ...hiddenUnits] = hiddenLayers;
        const model = tf.sequential();

        // Add an input layer
        model.add(tf.layers.dense({ inputShape: [inputSize], units: inputUnits }));
        // Add hidden layers
        hiddenUnits?.forEach((units) => {
            model.add(tf.layers.dense({ units, activation: 'relu' }));
        });
        // Add an output layer
        model.add(tf.layers.dense({ units: outputSize }));

        return model;
    }

    addTrainingData(...candles: Candle[]): void {
        candles.forEach((candle, index) => {
            const ratioCandle = this.prevCandle[index] && getQuoteRatioData(candle, this.prevCandle[index]);

            if (ratioCandle) {
                if (!this.dataset[index]) this.dataset[index] = [];
                this.dataset[index].push(ratioCandle);
            }

            this.prevCandle[index] = candle;
        });
    }

    serveTrainingData(): void {
        const { inputSize, outputSize } = this.opts;
        const candleInputSize = inputSize / this.dataset.length;

        this.dataset.forEach((dataset, index) => {
            this.distribution[index] = getDistribution(dataset, this.opts.segmentsCount);

            dataset.forEach((ratioCandle) => {
                const groupId = this.distribution[index].findIndex(
                    (group) => ratioCandle.ratio >= group.ratioFrom && ratioCandle.ratio < group.ratioTo,
                );
                const normalisedGroupId = this.normalize(groupId);

                if (!this.input[index]) this.input[index] = [];
                this.input[index].push(normalisedGroupId);
            });
        });

        for (
            let windowStart = 0, windowEnd = candleInputSize;
            windowEnd < this.input[0].length - outputSize;
            windowEnd = ++windowStart + candleInputSize
        ) {
            const output = [...this.input[0]].slice(windowEnd, windowEnd + outputSize);
            const input = this.input.map((input) => Array.from(input).slice(windowStart, windowEnd));

            this.trainingSet.push({ input: input.flat(), output });

            log.verbose(
                'Input:',
                `\n${input.map((row) => row.join(' ')).join('\n')}`,
                `(${input.flat().length})`,
                '\nOutput:',
                output.join(' '),
                `(${output.length})`,
            );
        }
    }

    private convertToTensor(data: typeof this.trainingSet) {
        return tf.tidy(() => {
            tf.util.shuffle(data);

            const inputData = data.map((d) => d.input);
            const outputData = data.map((d) => d.output);

            const inputs = tf.tensor2d(inputData);
            const outputs = tf.tensor2d(outputData);

            return { inputs, outputs };
        });
    }

    private getInput(...candles: Candle[]): typeof this.input {
        const input = [...this.input];
        const singleInputSize = this.opts.inputSize / candles.length;

        candles.forEach((candle, index) => {
            const ratioCandle = this.prevCandle[index] && getQuoteRatioData(candle, this.prevCandle[index]);

            if (ratioCandle) {
                let groupId = this.distribution[index].findIndex(
                    (group) => ratioCandle.ratio >= group.ratioFrom && ratioCandle.ratio < group.ratioTo,
                );
                if (groupId === -1) {
                    groupId =
                        ratioCandle.ratio < this.distribution[index][0].ratioFrom
                            ? 0
                            : this.distribution[index].length - 1;
                }
                const normalisedGroupId = this.normalize(groupId);
                if (!input[index]) input[index] = [];
                input[index].push(normalisedGroupId);

                if (input[index].length > singleInputSize) {
                    input[index].shift();
                }
            }
        });

        return input;
    }

    private getOutput(input: typeof this.input, ...candles: Candle[]): CortexForecast[] | undefined {
        const flattenedInput = input.flat();

        if (flattenedInput.length === this.opts.inputSize) {
            const forecast = tf.tidy(() => {
                const input = tf.tensor2d(flattenedInput, [1, flattenedInput.length]);
                const prediction = this.model.predict(input) as tf.Tensor;
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

            log.verbose(
                'Input:',
                `\n${input.map((row) => row.join(' ')).join('\n')}`,
                `(${input.flat().length})`,
                '\nOutput:',
                output.join(' '),
                `(${output.length})`,
            );

            return output;
        }
    }

    /**
     * Get forecast at the moment
     */
    momentValue(...candles: Candle[]) {
        const input = this.getInput(...candles);

        return this.getOutput(input, ...candles);
    }

    /**
     * Run forecast
     */
    nextValue(...candles: Candle[]) {
        this.input = this.getInput(...candles);
        candles.forEach((candle, index) => {
            this.prevCandle[index] = candle;
        });

        return this.getOutput(this.input, ...candles);
    }

    async saveModel() {
        // TODO: 1. Make mode consistent
        //       2. Handle errors
        const { saveDir } = this.opts;

        file.ensureFile(path.resolve(saveDir, 'groups.json'));
        file.saveFile(path.resolve(saveDir, 'groups.json'), this.distribution);

        await this.model.save(`file://${path.resolve(saveDir)}`);
    }

    async loadModel() {
        const { loadDir } = this.opts;
        const groupsData = file.readFile(path.resolve(loadDir, 'groups.json'));

        if (!groupsData) {
            log.error('Unknown data in groups.json, or file does not exists, please run training before use');
            process.exit(0);
        }

        this.distribution = JSON.parse(groupsData);
        this.model = <tf.Sequential>await tf.loadLayersModel(`file://${path.resolve(loadDir)}/model.json`);
    }

    async training() {
        try {
            log.debug('Looking for existing model...');
            await this.loadModel();
            log.debug('Existing model loaded');
        } catch (e) {
            log.debug('No model found. Creating new one...');
        }

        if (!this.model) {
            this.model = this.createModel(this.opts);
        }

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
            shuffle: true,
            callbacks: tf.callbacks.earlyStopping({
                monitor: 'acc',
            }),
        });
        log.info('Training finished with accuracy:', history.acc.pop());
    }

    private normalize(groupId: number) {
        return groupId / this.opts.segmentsCount;
    }

    private denormalize(value: number) {
        return Math.min(Math.round(value * this.opts.segmentsCount), this.opts.segmentsCount - 1);
    }
}
