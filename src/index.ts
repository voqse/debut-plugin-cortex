import { logger, LoggerOptions } from '@voqse/logger';
import { Candle, PluginInterface } from '@debut/types';
import { cli } from '@debut/plugin-utils';
import path from 'path';
import { Model, ModelOptions } from './model';

export interface CortexForecast {
    low: number;
    high: number;
    avg: number;
}

export interface CortexPluginArgs {
    neuroTrain: boolean;
}

export interface CortexPluginOptions extends LoggerOptions, Partial<ModelOptions> {
    /**
     * Whether or not generate logs for tensorboard
     */
    tensorboard?: boolean;
}

interface CortexPluginMethods {
    momentValue(...candles: Candle[]): CortexForecast[] | undefined;
    nextValue(...candles: Candle[]): CortexForecast[] | undefined;
    addTrainValue(...candles: Candle[]): void;
    isTraining(): boolean;
}

interface CortexPluginInterface extends PluginInterface {
    name: string;
    api: CortexPluginMethods;
}

export interface CortexPluginAPI {
    cortex: CortexPluginMethods;
}

export function cortexPlugin(opts: CortexPluginOptions): CortexPluginInterface {
    const log = logger('cortex', opts);
    const isTraining = 'neuroTrain' in cli.getArgs<CortexPluginArgs>();
    let model: Model;

    async function initModel() {
        if (model) return;

        const { src } = (await cli.getBotData(this.debut.getName()))!;
        const { ticker } = this.debut.opts;

        let { loadDir = 'default', saveDir = loadDir, tensorboard } = opts;

        saveDir = path.join(src, 'cortex', ticker, saveDir);
        loadDir = path.join(src, 'cortex', ticker, loadDir);
        const logDir = tensorboard && path.join(saveDir, 'logs');

        log.debug('Creating neural network...');
        model = new Model({ ...opts, saveDir, loadDir, logDir });

        if (!isTraining) {
            log.info('Loading neural network...');
            await model.loadModel();
        }
    }

    return {
        name: 'cortex',
        api: {
            momentValue: (...candles) => model.momentValue(candles),
            nextValue: (...candles) => model.nextValue(candles),
            addTrainValue: (...candles) => model.addTrainingData(candles),
            isTraining: () => isTraining,
        },

        async onLearn() {
            await initModel.apply(this);
        },

        async onStart() {
            await initModel.apply(this);
        },

        async onDispose() {
            if (isTraining) {
                model.serveTrainingData();
                await model.training();
                await model.saveModel();
            }
        },
    };
}
