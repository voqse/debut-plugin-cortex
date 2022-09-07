import { Candle, PluginInterface } from '@debut/types';
import { logger, LoggerOptions } from '@voqse/logger';
import { cli } from '@debut/plugin-utils';
import { Network } from './neural';

export const pluginName = 'neurons';

export enum NeuronsType {
    'HIGH_UPTREND',
    'LOW_UPTREND',
    'NEUTRAL',
    'LOW_DOWNTREND',
    'HIGH_DOWNTREND',
}

export interface NeuronsPluginArgs {
    neuroTrain: boolean;
}

export interface NeuronsPluginOptions extends LoggerOptions {
    windowSize: number; // 25;
    segmentsCount: number; // 6
    precision: number; // 3
    hiddenLayers?: number[];
    debug?: boolean;
    crossValidate?: boolean;
    LSTM?: boolean;
}

interface NeuronsMethodsInterface {
    nextValue(xCandle: Candle, yCandle: Candle): NeuronsType | undefined;
    addTrainValue(xCandle: Candle, yCandle: Candle): void;
    isTraining(): boolean;
}

interface NeuronsPluginInterface extends PluginInterface {
    name: string;
    api: NeuronsMethodsInterface;
}

export interface NeuronsPluginAPI {
    [pluginName]: NeuronsMethodsInterface;
}

export function neuronsPlugin(opts: NeuronsPluginOptions): NeuronsPluginInterface {
    const log = logger(pluginName, opts);
    const neuroTrain = 'neuroTrain' in cli.getArgs<NeuronsPluginArgs>();
    let neuralNetwork: Network;

    return {
        name: pluginName,
        api: {
            nextValue: (xCandle, yCandle) => neuralNetwork.activate(xCandle, yCandle),
            addTrainValue: (xCandle, yCandle) => neuralNetwork.addTrainingData(xCandle, yCandle),
            isTraining: () => neuroTrain,
        },

        async onInit() {
            log.info('Initializing plugin...');
            const botData = await cli.getBotData(this.debut.getName())!;
            const workingDir = `${botData?.src}/${pluginName}/${this.debut.opts.ticker}/`;

            log.debug('Creating neural network...');
            neuralNetwork = new Network({ ...opts, workingDir });

            if (!neuroTrain) {
                neuralNetwork.restore();
            }
        },

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
