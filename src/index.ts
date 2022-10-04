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
    saveDir?: string;
    loadDir?: string;
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

    return {
        name: 'cortex',
        api: {
            momentValue: (...candles) => model.momentValue(...candles),
            nextValue: (...candles) => model.nextValue(...candles),
            addTrainValue: (...candles) => model.addTrainingData(...candles),
            isTraining: () => isTraining,
        },

        async onInit() {
            log.info('Initializing plugin...');
            const botData = (await cli.getBotData(this.debut.getName()))!;
            const { saveDir = 'default', loadDir = saveDir } = opts;
            const savePath = path.join(botData?.src, 'cortex', this.debut.opts.ticker, saveDir);
            const loadPath = path.join(botData?.src, 'cortex', this.debut.opts.ticker, loadDir);

            // log.debug('Creating neural network...');
            model = new Model({ ...opts, saveDir: savePath, loadDir: loadPath });

            if (!isTraining) {
                await model.loadModel();
            }
        },

        async onDispose() {
            if (isTraining) {
                model.serveTrainingData();
                await model.training();
                await model.saveModel();
            }
            log.info('Shutting down plugin...');
        },
    };
}
