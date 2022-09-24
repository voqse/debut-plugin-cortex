import * as tf from '@tensorflow/tfjs-node';
import '@tensorflow/tfjs-backend-cpu';
import { file, math } from '@debut/plugin-utils';
import { Candle } from '@debut/types';
import path from 'path';
import {
    DistributionSegment,
    getDistribution,
    getPredictPrices,
    getQuoteRatioData,
    printStatus,
    RatioCandle,
    timeToNow,
} from './utils';
import { NeuroVision, NeuroVisionPluginOptions } from './index';
import { logger, LoggerInterface, LoggerLevel } from '@voqse/logger';

let log: LoggerInterface;

interface Params extends NeuroVisionPluginOptions {
    workingDir: string;
}

export class Network {
    private model: tf.Sequential;
    private dataset: RatioCandle[][] = [];
    private trainingSet: { input: number[]; output: number[] }[] = [];
    private distribution: DistributionSegment[][] = [];
    private prevCandle: Candle[] = [];
    private input: number[][] = [];
    private layersPath: string;
    private gaussPath: string;

    constructor(private params: Params) {
        log = logger('neurons', params);

        this.model = this.createModel(params);

        this.gaussPath = path.resolve(params.workingDir, 'gaussian-groups.json');
        this.layersPath = path.resolve(params.workingDir);
    }

    createModel(options: Params): typeof this.model {
        const { inputSize = 60, outputSize = 3, hiddenLayers = [32, 16] } = options;
        const model = tf.sequential();
        // Add a single input layer
        model.add(tf.layers.dense({ inputShape: [inputSize], units: inputSize, useBias: true }));
        // Add hidden layers
        hiddenLayers.forEach((layer) => {
            model.add(tf.layers.dense({ units: layer, activation: 'relu' }));
        });
        // Add an output layer
        model.add(tf.layers.dense({ units: outputSize, useBias: true }));

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
        console.log('Candles count:', this.dataset.length);

        const { inputSize, outputSize = 3 } = this.params;
        const realInputSize = inputSize / this.dataset.length;

        this.dataset.forEach((dataset, index) => {
            this.distribution[index] = getDistribution(dataset, this.params.segmentsCount);

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
        console.log('Candles count:', candles.length);

        const input = [...this.input];
        const { inputSize } = this.params;
        const singleInputSize = inputSize / candles.length;

        candles.forEach((candle, index) => {
            const ratioCandle = this.prevCandle[index] && getQuoteRatioData(candle, this.prevCandle[index]);
            this.prevCandle[index] = candle;

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

    private getOutput(input: typeof this.input, ...candles: Candle[]): NeuroVision[] | undefined {
        const flattenedInput = input.flat();
        console.log('Total input size:', flattenedInput.length);

        if (flattenedInput.length === this.params.inputSize) {
            const forecast = tf.tidy(() => {
                const input = tf.tensor2d(flattenedInput, [1, flattenedInput.length]);
                const prediction = this.model.predict(input) as tf.Tensor;
                return Array.from(prediction.dataSync());
            });
            console.log(forecast);
            const output: NeuroVision[] = [];

            for (let i = 0; i < forecast.length; i++) {
                const cast = forecast[i];
                const denormalized = this.denormalize(cast);
                const group = this.distribution[0][denormalized];

                if (group) output.push(getPredictPrices(candles[0].c, group.ratioFrom, group.ratioTo));
            }

            if (this.params.logLevel === LoggerLevel.debug) {
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

        const { batchSize, epochs } = this.params;
        const { inputs, outputs } = this.convertToTensor(this.trainingSet);

        await this.model.fit(inputs, outputs, {
            batchSize,
            epochs,
            shuffle: true,
            // callbacks: {
            //     onYield(...args) {
            //         console.log(args);
            //     },
            // },
        });
        log.debug('Training finished');
    }

    private normalize(groupId: number) {
        // return groupId;
        return groupId / this.params.segmentsCount;
    }

    private denormalize(value: number) {
        // return Math.round(value);
        return Math.min(Math.round(value * this.params.segmentsCount), this.params.segmentsCount - 1);
    }
}
