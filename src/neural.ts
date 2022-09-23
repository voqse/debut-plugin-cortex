// @ts-ignore
import { CrossValidate, INeuralNetworkOptions, NeuralNetwork } from 'brain.js';
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
    private network: NeuralNetwork<number[], number[]> = null!;
    private crossValidate: CrossValidate<() => typeof this.network> = null!;
    private dataset: RatioCandle[][] = [];
    private trainingSet: { input: number[]; output: number[] }[] = [];
    private distribution: DistributionSegment[][] = [];
    private prevCandle: Candle[] = [];
    private input: number[][] = [];
    private layersPath: string;
    private gaussPath: string;

    constructor(private params: Params) {
        log = logger('neurons', params);

        const nnOpts: INeuralNetworkOptions = {
            hiddenLayers: params.hiddenLayers || [32, 16], // array of ints for the sizes of the hidden layers in the network
            activation: 'sigmoid', // supported activation types: ['sigmoid', 'relu', 'leaky-relu', 'tanh'],
            leakyReluAlpha: 0.01,
        };
        if (this.params.crossValidate) {
            this.crossValidate = new CrossValidate(() => new NeuralNetwork(nnOpts));
        } else {
            this.network = new NeuralNetwork(nnOpts);
        }

        this.gaussPath = path.resolve(params.workingDir, 'gaussian-groups.json');
        this.layersPath = path.resolve(params.workingDir, 'nn-layers.json');
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
        this.dataset.forEach((dataset, index) => {
            this.distribution[index] = getDistribution(dataset, this.params.segmentsCount, this.params.precision);

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
            let windowStart = 0, windowEnd = this.params.inputSize;
            windowEnd < this.input[0].length - this.params.prediction;
            windowEnd = ++windowStart + this.params.inputSize
        ) {
            const output = [...this.input[0]].slice(windowEnd, windowEnd + this.params.prediction);
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

    private getInput(...candles: Candle[]): typeof this.input {
        const input = [...this.input];

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

                if (input[index].length > this.params.inputSize) {
                    input[index].shift();
                }
            }
        });

        return input;
    }

    private getOutput(input: typeof this.input, ...candles: Candle[]): NeuroVision[] | undefined {
        const length = candles.length;
        const flattenedInput = input.flat();

        if (flattenedInput.length === this.params.inputSize * length) {
            const forecast = this.network.run(flattenedInput);
            const output: NeuroVision[] = [];

            for (let i = 0; i < forecast.length; i++) {
                const cast = forecast[i];
                const denormalized = this.denormalize(cast);
                const group = this.distribution[0][denormalized];

                output.push(getPredictPrices(candles[0].c, group.ratioFrom, group.ratioTo));
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
        candles.forEach((candle, index) => {
            this.prevCandle[index] = candle;
        });
        return this.getOutput(this.input, ...candles);
    }

    save() {
        const source = this.crossValidate || this.network;
        file.ensureFile(this.gaussPath);
        file.ensureFile(this.layersPath);
        file.saveFile(this.gaussPath, this.distribution);
        file.saveFile(this.layersPath, source.toJSON());
    }

    load() {
        const groupsData = file.readFile(this.gaussPath);
        const nnLayersData = file.readFile(this.layersPath);

        if (!groupsData) {
            throw 'Unknown data in gaussian-groups.json, or file does not exists, please run training before use';
        }

        if (!nnLayersData) {
            throw 'Unknown data in nn-layers.json, or file does not exists, please run training before use';
        }

        const nnLayers = JSON.parse(nnLayersData);

        this.distribution = JSON.parse(groupsData);

        if (this.params.crossValidate) {
            this.network = this.crossValidate.fromJSON(nnLayers);
        } else {
            this.network.fromJSON(nnLayers);
        }
    }

    training() {
        log.info('Starting training...');
        const source = this.crossValidate || this.network;
        const statuses = [];
        const startTime = new Date().getTime();
        let prevTime = new Date().getTime();

        const logStatus = ({ iterations, error }) => {
            const now = new Date().getTime();
            const speed = math.toFixed((10 * 1000) / (now - prevTime), 6);

            if (statuses.length === 5) {
                statuses.shift();
            }

            statuses.push({ totalTime: timeToNow(startTime), time: timeToNow(prevTime), iterations, error, speed });
            printStatus(statuses);
            prevTime = now;
        };

        source.train(this.trainingSet, {
            // Defaults values --> expected validation
            iterations: 40000, // the maximum times to iterate the training data --> number greater than 0
            errorThresh: 0.001, // the acceptable error percentage from training data --> number between 0 and 1
            log: logStatus, // true to use console.log, when a function is supplied it is used --> Either true or a function
            logPeriod: 10, // iterations between logging out --> number greater than 0
            learningRate: 0.3, // scales with delta to effect training rate --> number between 0 and 1
            momentum: 0.1, // scales with next layer's change value --> number between 0 and 1
            timeout: 3600000 * 6,
        });

        if (!this.network) {
            this.network = this.crossValidate.toNeuralNetwork();
        }
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
