import { logger, LoggerOptions } from '@voqse/logger';
import { Candle, PluginInterface } from '@debut/types';
import { cli } from '@debut/plugin-utils';
import { NeuronsOptions, Neurons } from './neurons';

export interface CortexForecast {
    low: number;
    high: number;
    avg: number;
}

export interface CortexPluginArgs {
    neuroTrain: boolean;
}

export interface CortexPluginOptions extends LoggerOptions, NeuronsOptions {
    name?: string;
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

export function cortexPlugin(opts: CortexPluginOptions): CortexPluginInterface {
    const log = logger('cortex', opts);

    const isTraining = 'neuroTrain' in cli.getArgs<CortexPluginArgs>();
    let neurons: Neurons;

    return {
        name: 'cortex',
        api: {
            momentValue: (...candles) => neurons.momentValue(...candles),
            nextValue: (...candles) => neurons.nextValue(...candles),
            addTrainValue: (...candles) => neurons.addTrainingData(...candles),
            isTraining: () => isTraining,
        },

        async onInit() {
            log.info('Initializing plugin...');
            const botData = await cli.getBotData(this.debut.getName())!;
            const workingDir = `${botData?.src}/cortex/${this.debut.opts.ticker}/${opts.name || 'default'}`;

            log.debug('Creating neural network...');
            neurons = new Neurons({ ...opts, workingDir });

            if (!isTraining) {
                await neurons.load();
            }
        },

        async onDispose() {
            log.info('Shutting down plugin...');

            if (isTraining) {
                neurons.serveTrainingData();
                await neurons.training();
                await neurons.save();
            }
        },
    };
}
