import { Candle, PluginInterface } from '@debut/types';
import { cli } from '@debut/plugin-utils';
import { Network } from './neural';
import { logger, LoggerOptions } from '@voqse/logger';

export interface NeuroVision {
    low: number;
    high: number;
    avg: number;
}

export interface NeuroVisionPluginArgs {
    neuroTrain: boolean;
}

export interface NeuroVisionPluginOptions extends LoggerOptions {
    inputSize: number; // 25;
    outputSize?: number;
    hiddenLayers: number[];
    segmentsCount: number; // 6
    name?: string;
    batchSize?: number;
    epochs?: number;
}

interface Methods {
    momentValue(...candles: Candle[]): NeuroVision[] | undefined;
    nextValue(...candles: Candle[]): NeuroVision[] | undefined;
    addTrainValue(...candles: Candle[]): void;
    isTraining(): boolean;
}

interface NeuroVisionPluginInterface extends PluginInterface {
    name: 'neuroVision';
    api: Methods;
}

export interface NeuroVisionPluginAPI {
    neuroVision: Methods;
}

export function neuroVisionPlugin(params: NeuroVisionPluginOptions): NeuroVisionPluginInterface {
    const log = logger('neuroVision', params);
    const neuroTrain = 'neuroTrain' in cli.getArgs<NeuroVisionPluginArgs>();
    let neural: Network;

    return {
        name: 'neuroVision',
        api: {
            momentValue: (...candles: Candle[]) => neural.momentValue(...candles),
            nextValue: (...candles: Candle[]) => neural.nextValue(...candles),
            addTrainValue(...candles: Candle[]) {
                neural.addTrainingData(...candles);
            },
            isTraining() {
                return neuroTrain;
            },
        },

        async onInit() {
            log.info('Initializing plugin...');

            const botData = await cli.getBotData(this.debut.getName())!;
            const workingDir = `${botData?.src}/neuro-vision/${this.debut.opts.ticker}/${params.name || 'default'}`;

            log.debug('Creating neural network...');
            neural = new Network({ ...params, workingDir });

            if (!neuroTrain) {
                await neural.load();
            }
        },

        async onDispose() {
            log.info('Shutting down plugin...');

            if (neuroTrain) {
                neural.serveTrainingData();
                await neural.training();
                await neural.save();
            }
        },
    };
}
