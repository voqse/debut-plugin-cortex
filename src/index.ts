import { Candle, PluginInterface } from '@debut/types';
import { cli } from '@debut/plugin-utils';
import { Network } from './neural';
import { logger, LoggerOptions } from '@voqse/logger';

export enum NeuroVision {
    'HIGH_UPTREND',
    'LOW_UPTREND',
    'NEUTRAL',
    'LOW_DOWNTREND',
    'HIGH_DOWNTREND',
}

export interface NeuroVisionPluginArgs {
    neuroTrain: boolean;
}

export interface NeuroVisionPluginOptions extends LoggerOptions {
    windowSize: number; // 25;
    segmentsCount: number; // 6
    precision: number; // 3
    hiddenLayers?: number[];
    debug?: boolean;
    crossValidate?: boolean;
    LSTM?: boolean;
}

interface NeuroVisionMethodsInterface {
    nextValue(xCandle: Candle, yCandle: Candle): NeuroVision | undefined;
    addTrainValue(xCandle: Candle, yCandle: Candle): void;
    isTraining(): boolean;
}

interface NeuroVisionPluginInterface extends PluginInterface {
    name: 'neuroVision';
    api: NeuroVisionMethodsInterface;
}

export interface NeuroVisionPluginAPI {
    neuroVision: NeuroVisionMethodsInterface;
}

export function neuroVisionPlugin(opts: NeuroVisionPluginOptions): NeuroVisionPluginInterface {
    const log = logger('neuro', opts);
    const neuroTrain = 'neuroTrain' in cli.getArgs<NeuroVisionPluginArgs>();
    let neuralNetwork: Network;

    return {
        name: 'neuroVision',
        api: {
            nextValue: (xCandle, yCandle) => neuralNetwork.activate(xCandle, yCandle),
            addTrainValue: (xCandle, yCandle) => neuralNetwork.addTrainingData(xCandle, yCandle),
            isTraining: () => neuroTrain,
        },

        async onInit() {
            log.info('Plugin initializing...');
            const botData = await cli.getBotData(this.debut.getName())!;
            const workingDir = `${botData?.src}/neuro-vision/${this.debut.opts.ticker}/`;

            neuralNetwork = new Network({ ...opts, workingDir });

            if (!neuroTrain) {
                neuralNetwork.restore();
            }
        },

        // async onStart() {
        //     if (!neuroTrain) {
        //         neuralNetwork.restore();
        //     }
        // },

        async onDispose() {
            log.info('Shutting down plugin...');
            if (neuroTrain) {
                neuralNetwork.serveTrainingData();
                neuralNetwork.training();
                neuralNetwork.save();
            }
        },
    };
}
