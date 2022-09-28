import { logger, LoggerInterface, LoggerLevel, LoggerOptions } from '@voqse/logger';
import { Candle } from '@debut/types';
import { file } from '@debut/plugin-utils';
import { DistributionSegment, getDistribution, getPredictPrices, getQuoteRatioData, RatioCandle } from './utils';
import { CortexForecast } from './index';
import * as tf from '@tensorflow/tfjs-node';
import '@tensorflow/tfjs-backend-cpu';
import path from 'path';

let log: LoggerInterface;

export interface NeuronsOptions extends LoggerOptions {
    hiddenLayers: number[];
    segmentsCount: number;
    inputSize: number;
    outputSize?: number;
    batchSize?: number;
    epochs?: number;
    savePath?: string;
}

export class Neurons {
    private model: tf.Sequential;
    private dataset: RatioCandle[][] = [];
    private trainingSet: { input: number[]; output: number[] }[] = [];
    private distribution: DistributionSegment[][] = [];
    private prevCandle: Candle[] = [];
    private input: number[][] = [];
    private layersPath: string;
    private gaussPath: string;

    constructor(private opts: NeuronsOptions) {
        log = logger('cortex/neurons', opts);

        this.model = this.createModel(opts);
        this.gaussPath = path.resolve(opts.savePath, 'groups.json');
        this.layersPath = path.resolve(opts.savePath);
    }

    createModel(options: NeuronsOptions): typeof this.model {
        const { inputSize = 60, outputSize = 3, hiddenLayers = [inputSize] } = options;
        const inputUnits = hiddenLayers.shift();
        const model = tf.sequential();

        // Add a single input layer
        model.add(tf.layers.dense({ inputShape: [inputSize], units: inputUnits }));
        // Add hidden layers
        hiddenLayers.forEach((units) => {
            model.add(tf.layers.dense({ units, activation: 'relu' }));
        });
        // Add an output layer
        model.add(tf.layers.dense({ units: outputSize }));

        model.compile({
            optimizer: tf.train.adam(),
            loss: tf.losses.meanSquaredError,
            metrics: ['accuracy'],
        });

        return model;
    }

    /**
     * Add training set with ratios and forecast ratio as output
     */
    addTrainingData(...candles: Candle[]) {
        candles.forEach((candle, index) => {
            const ratioCandle = this.prevCandle[index] && getQuoteRatioData(candle, this.prevCandle[index]);

            if (ratioCandle) {
                if (!this.dataset[index]) this.dataset[index] = [];
                this.dataset[index].push(ratioCandle);
            }

            this.prevCandle[index] = candle;
        });
    }

    serveTrainingData() {
        log.debug('Candles count:', this.dataset.length);

        const { inputSize, outputSize = 3, logLevel } = this.opts;
        const realInputSize = inputSize / this.dataset.length;

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
            let windowStart = 0, windowEnd = realInputSize;
            windowEnd < this.input[0].length - outputSize;
            windowEnd = ++windowStart + realInputSize
        ) {
            const output = [...this.input[0]].slice(windowEnd, windowEnd + outputSize);
            const input = this.input.map((input) => Array.from(input).slice(windowStart, windowEnd));

            this.trainingSet.push({ input: input.flat(), output });

            if (logLevel === LoggerLevel.debug) {
                const inputRows = input.map((row) => row.join(' '));
                log.debug(
                    'Input:',
                    `\n${inputRows.join('\n')}`,
                    `(${input.flat().length})`,
                    '\nOutput:',
                    output.join(' '),
                    `(${output.length})`,
                );
            }
        }
    }

    convertToTensor(data) {
        // Wrapping these calculations in a tidy will dispose any
        // intermediate tensors.

        return tf.tidy(() => {
            // Step 1. Shuffle the data
            tf.util.shuffle(data);

            // Step 2. Convert data to Tensor
            const inputs = data.map((d) => d.input);
            const outputs = data.map((d) => d.output);

            const inputTensor = tf.tensor2d(inputs);
            const outputTensor = tf.tensor2d(outputs);

            return {
                inputs: inputTensor,
                outputs: outputTensor,
            };
        });
    }

    private getInput(...candles: Candle[]): typeof this.input {
        log.debug('Candles count:', candles.length);

        const input = [...this.input];
        const { inputSize } = this.opts;
        const singleInputSize = inputSize / candles.length;

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
        // console.log('Total input size:', flattenedInput.length);

        if (flattenedInput.length === this.opts.inputSize) {
            const forecast = tf.tidy(() => {
                const input = tf.tensor2d(flattenedInput, [1, flattenedInput.length]);
                const prediction = this.model.predict(input) as tf.Tensor;
                return Array.from(prediction.dataSync());
            });
            // console.log(forecast);
            const output: CortexForecast[] = [];

            for (let i = 0; i < forecast.length; i++) {
                const cast = forecast[i];
                const denormalized = this.denormalize(cast);
                const group = this.distribution[0][denormalized];

                if (group) output.push(getPredictPrices(candles[0].c, group.ratioFrom, group.ratioTo));
            }

            if (this.opts.logLevel === LoggerLevel.debug) {
                const inputRows = input.map((row) => row.join(' '));
                log.debug(
                    'Input:',
                    `\n${inputRows.join('\n')}`,
                    `(${input.flat().length})`,
                    '\nOutput:',
                    forecast.join(' '),
                    `(${forecast.length})`,
                );
            }
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

    async save() {
        file.ensureFile(this.gaussPath);
        // file.ensureFile(this.layersPath);
        file.saveFile(this.gaussPath, this.distribution);

        await this.model.save(`file://${this.layersPath}`);
    }

    async load() {
        const groupsData = file.readFile(this.gaussPath);

        if (!groupsData) {
            throw 'Unknown data in gaussian-groups.json, or file does not exists, please run training before use';
        }
        this.distribution = JSON.parse(groupsData);
        this.model = <tf.Sequential>await tf.loadLayersModel(`file://${this.layersPath}/model.json`);
    }

    async training() {
        log.info('Starting training...');
        // console.log(this.trainingSet);

        const { batchSize, epochs } = this.opts;
        const { inputs, outputs } = this.convertToTensor(this.trainingSet);

        await this.model.fit(inputs, outputs, {
            batchSize,
            epochs,
            shuffle: true,
        });
        log.debug('Training finished');
    }

    private normalize(groupId: number) {
        // return groupId;
        return groupId / this.opts.segmentsCount;
    }

    private denormalize(value: number) {
        // return Math.round(value);
        return Math.min(Math.round(value * this.opts.segmentsCount), this.opts.segmentsCount - 1);
    }
}
