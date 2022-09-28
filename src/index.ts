import { Candle, PluginInterface } from '@debut/types';
import { cli } from '@debut/plugin-utils';
import { Cortex } from './cortex';
import { logger, LoggerOptions } from '@voqse/logger';

export interface CortexForecast {
    low: number;
    high: number;
    avg: number;
}

export interface CortexPluginArgs {
    neuroTrain: boolean;
}

export interface CortexPluginOptions extends LoggerOptions {
    inputSize: number; // 25;
    outputSize?: number;
    hiddenLayers: number[];
    segmentsCount: number; // 6
    name?: string;
    batchSize?: number;
    epochs?: number;
}

interface CortexPluginMethods {
    momentValue(...candles: Candle[]): CortexForecast[] | undefined;
    nextValue(...candles: Candle[]): CortexForecast[] | undefined;
    addTrainValue(...candles: Candle[]): void;
    isTraining(): boolean;
}

interface CortexPluginInterface extends PluginInterface {
    name: 'cortex';
    api: CortexPluginMethods;
}

export interface CortexPluginAPI {
    cortex: CortexPluginMethods;
}

export function cortexPlugin(params: CortexPluginOptions): CortexPluginInterface {
    const log = logger('neuroVision', params);
    const isTraining = 'neuroTrain' in cli.getArgs<CortexPluginArgs>();
    let neural: Cortex;

    return {
        name: 'cortex',
        api: {
            momentValue: (...candles) => neural.momentValue(...candles),
            nextValue: (...candles) => neural.nextValue(...candles),
            addTrainValue: (...candles) => neural.addTrainingData(...candles),
            isTraining: () => isTraining,
        },

        async onInit() {
            log.info('Initializing plugin...');
            const botData = await cli.getBotData(this.debut.getName())!;
            const workingDir = `${botData?.src}/neuro-vision/${this.debut.opts.ticker}/${params.name || 'default'}`;

            log.debug('Creating neural network...');
            neural = new Cortex({ ...params, workingDir });

            if (!isTraining) {
                await neural.load();
            }
        },

        async onDispose() {
            log.info('Shutting down plugin...');

            if (isTraining) {
                neural.serveTrainingData();
                await neural.training();
                await neural.save();
            }
        },
    };
}
