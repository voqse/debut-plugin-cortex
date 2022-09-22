// @ts-ignore
import { CrossValidate, INeuralNetworkOptions, NeuralNetwork } from 'brain.js';
import { file, math } from '@debut/plugin-utils';
import { Candle } from '@debut/types';
import { logger, LoggerInterface } from '@voqse/logger';
import path from 'path';
import { getDistribution, getQuoteRatioData, RatioCandle, DistributionSegment, timeToNow, printStatus } from '../utils';
import { NeuroVision, NeuroVisionPluginOptions } from '../index';

let log: LoggerInterface;

export interface NetworkOptions extends NeuroVisionPluginOptions {
    neuronsDir: string;
}

export class Network {
    private network: NeuralNetwork<number[], number[]>;
    private crossValidate: CrossValidate<() => typeof this.network> = null!;
    private xDataset: RatioCandle[] = [];
    private yDataset: RatioCandle[] = [];
    private trainingSet: Array<{ input: number[]; output: number[] }> = [];
    private xDistribution: DistributionSegment[] = [];
    private yDistribution: DistributionSegment[] = [];
    private prevXCandle: Candle | null = null;
    private prevYCandle: Candle | null = null;
    private input: number[] = [];
    private layersPath: string;
    private gaussPathX: string;
    private gaussPathY: string;

    constructor(private opts: NetworkOptions) {
        log = logger('neurons', opts);

        const nnOpts: INeuralNetworkOptions = {
            // inputSize: opts.windowSize * 2,
            // outputSize: 1,
            hiddenLayers: opts.hiddenLayers || [32, 16], // array of ints for the sizes of the hidden layers in the network
            activation: 'sigmoid', // supported activation types: ['sigmoid', 'relu', 'leaky-relu', 'tanh'],
            leakyReluAlpha: 0.01,
        };
        if (this.opts.crossValidate) {
            this.crossValidate = new CrossValidate(() => new NeuralNetwork(nnOpts));
            log.debug('Cross Validation created');
        } else {
            this.network = new NeuralNetwork(nnOpts);
            log.debug('Normal network created');
        }

        this.gaussPathX = path.resolve(opts.neuronsDir, 'neuro-3/gaussian-groups-x.json');
        this.gaussPathY = path.resolve(opts.neuronsDir, 'neuro-3/gaussian-groups-y.json');
        this.layersPath = path.resolve(opts.neuronsDir, 'neuro-3/nn-layers.json');
    }

    /**
     * Add training set with ratios and forecast ratio as output
     */
    addTrainingData = (xCandle: Candle, yCandle: Candle) => {
        const ratioXCandle = this.prevXCandle && getQuoteRatioData(xCandle, this.prevXCandle);
        const ratioYCandle = this.prevYCandle && getQuoteRatioData(yCandle, this.prevYCandle);

        if (ratioXCandle && ratioYCandle) {
            this.xDataset.push(ratioXCandle);
            this.yDataset.push(ratioYCandle);
        }

        this.prevXCandle = xCandle;
        this.prevYCandle = yCandle;
    };

    serveTrainingData = () => {
        this.xDistribution = getDistribution(this.xDataset, this.opts.segmentsCount, this.opts.precision);
        this.yDistribution = getDistribution(this.yDataset, this.opts.segmentsCount, this.opts.precision);
        if (this.opts.debug) {
            log.debug(this.xDistribution, this.yDistribution);
        }

        for (let i = 0; i < this.xDataset.length; i++) {
            const ratioXCandle = this.xDataset[i];
            const ratioYCandle = this.yDataset[i];
            const groupXId = this.normalize(
                this.xDistribution.findIndex(
                    (group) => ratioXCandle.ratio >= group.ratioFrom && ratioXCandle.ratio < group.ratioTo,
                ),
            );
            const groupYId = this.normalize(
                this.yDistribution.findIndex(
                    (group) => ratioYCandle.ratio >= group.ratioFrom && ratioYCandle.ratio < group.ratioTo,
                ),
            );

            this.input.push(groupXId, groupYId);

            if (this.input.length === this.opts.inputSize * 2) {
                const forecastingRatio = this.xDataset[i + 1]?.ratio;

                if (!forecastingRatio) {
                    break;
                }

                const outputGroupId = this.xDistribution.findIndex(
                    (group) => forecastingRatio >= group.ratioFrom && forecastingRatio < group.ratioTo,
                );
                const normalizedOutput = this.normalize(outputGroupId);

                this.trainingSet.push({ input: [...this.input], output: [normalizedOutput] });
                this.input.splice(0, 2);
                // console.log('Input: ', [...this.input], ' Output: ', [normalizedOutput]);
            }

            // this.input.push(this.normalize(groupId));
        }
        log.debug('trainingSet length:', this.trainingSet.length);
    };

    /**
     * Run forecast
     */
    activate(xCandle: Candle, yCandle: Candle): number {
        const ratioXCandle = this.prevXCandle && getQuoteRatioData(xCandle, this.prevXCandle);
        const ratioYCandle = this.prevYCandle && getQuoteRatioData(yCandle, this.prevYCandle);

        this.prevXCandle = xCandle;
        this.prevYCandle = yCandle;

        if (ratioXCandle && ratioYCandle) {
            let idxX = this.xDistribution.findIndex(
                (group) => ratioXCandle.ratio >= group.ratioFrom && ratioXCandle.ratio < group.ratioTo,
            );
            let idxY = this.yDistribution.findIndex(
                (group) => ratioYCandle.ratio >= group.ratioFrom && ratioYCandle.ratio < group.ratioTo,
            );

            if (idxX === -1) {
                idxX = ratioXCandle.ratio < this.xDistribution[0].ratioFrom ? 0 : this.xDistribution.length - 1;
            }

            if (idxY === -1) {
                idxY = ratioYCandle.ratio < this.yDistribution[0].ratioFrom ? 0 : this.yDistribution.length - 1;
            }

            const groupXId = this.normalize(idxX);
            const groupYId = this.normalize(idxY);

            this.input.push(groupXId, groupYId);

            if (this.input.length === this.opts.inputSize * 2) {
                const forecast = this.network.run(this.input);

                this.input.splice(0, 2);

                const denormalized = this.denormalize(forecast[0]);
                const group = this.xDistribution[denormalized];

                if (!group) {
                    log.debug(denormalized);
                }

                return denormalized;
            }
        }
    }

    save() {
        log.debug('Saving neurons schema...');
        const source = this.crossValidate || this.network;
        file.ensureFile(this.gaussPathX);
        file.ensureFile(this.gaussPathY);
        file.ensureFile(this.layersPath);
        file.saveFile(this.gaussPathX, this.xDistribution);
        file.saveFile(this.gaussPathY, this.yDistribution);
        file.saveFile(this.layersPath, source.toJSON());
        log.debug('Neurons schema saved');
    }

    restore() {
        log.info('Loading neurons schema...');
        const groupsDataX = file.readFile(this.gaussPathX);
        const groupsDataY = file.readFile(this.gaussPathY);
        const nnLayersData = file.readFile(this.layersPath);

        if (!groupsDataX || !groupsDataY) {
            log.error('Neurons schema load fail');
            throw 'Unknown data in gaussian-groups.json, or file does not exists, please run training before use';
        }

        if (!nnLayersData) {
            log.error('Neurons schema load fail');
            throw 'Unknown data in nn-layers.json, or file does not exists, please run training before use';
        }

        const nnLayers = JSON.parse(nnLayersData);

        this.xDistribution = JSON.parse(groupsDataX);
        this.yDistribution = JSON.parse(groupsDataY);

        if (this.opts.crossValidate) {
            this.network = this.crossValidate.fromJSON(nnLayers);
        } else {
            this.network.fromJSON(nnLayers);
        }
        log.debug('Neurons schema loaded');
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
            errorThresh: 0.01, // the acceptable error percentage from training data --> number between 0 and 1
            log: logStatus, // true to use console.log, when a function is supplied it is used --> Either true or a function
            logPeriod: 10, // iterations between logging out --> number greater than 0
            learningRate: 0.3, // scales with delta to effect training rate --> number between 0 and 1
            momentum: 0.2, // scales with next layer's change value --> number between 0 and 1
            timeout: 3600000 * 6,
        });

        if (!this.network) {
            this.network = this.crossValidate.toNeuralNetwork();
        }
        log.debug('Training finished');
    }

    private normalize(groupId: number) {
        // return groupId;
        return math.toFixed(groupId / (this.opts.segmentsCount - 1), this.opts.precision);
    }

    private denormalize(value: number) {
        // return Math.round(value);
        return Math.min(Math.floor(value * this.opts.segmentsCount), this.opts.segmentsCount - 1);
    }
}
