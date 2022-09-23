import { Candle } from '@debut/types';
import { math } from '@debut/plugin-utils';
import { NeuroVision } from './index';
import * as readline from 'readline';

/**
 * Special candle format with ratio instead value
 */
export interface RatioCandle {
    time: number;
    volume: number;
    ratio: number;
}

/**
 * Ratio to count same ratio's distribution
 */
export interface DistributionSegment {
    ratioFrom: number;
    ratioTo: number;
    count: number;
}

/**
 * Ratio to count same ratio's distribution
 */
interface DistributionData {
    count: number;
    ratio: number;
}

/**
 * Replace close prices to ratio (percent increment) prices with prev price and current
 */
export function getQuoteRatioData(current: Candle, prev: Candle): RatioCandle {
    const prevValue = (prev.o + prev.h + prev.l + prev.c) / 4;
    const currentValue = (current.o + current.h + current.l + current.c) / 4;

    return {
        time: current.time,
        ratio: currentValue / prevValue,
        volume: current.v,
    };
}

/**
 * Create gausiian distribution of percent increment ratio's
 * Split distribution data to same equal (in counts) segments in rages of ratios
 */
export function getDistribution(ratioCandles: RatioCandle[], segmentsCount = 6, precision = 4) {
    const map: Map<number, number> = new Map();

    for (let i = 0; i < ratioCandles.length; i++) {
        const candle = ratioCandles[i];
        const key = math.toFixed(candle.ratio, precision);
        const counter = map.get(key) || 0;

        map.set(key, counter + 1);
    }

    const sortedRatioKeys = Array.from(map.keys()).sort();
    const gaussianDistr: DistributionData[] = [];

    sortedRatioKeys.forEach((key) => {
        gaussianDistr.push({
            count: map.get(key) || 0,
            ratio: key,
        });
    });

    const segments: DistributionSegment[] = [];
    const segmentSize = Math.ceil(ratioCandles.length / segmentsCount);
    let localCountSum = 0;
    let ratioFrom = gaussianDistr[0].ratio;

    gaussianDistr.forEach((item, idx) => {
        const isLast = idx === gaussianDistr.length - 1;
        const isFilled = segments.length === segmentsCount - 1;
        const nextSum = localCountSum + item.count;

        if ((nextSum > segmentSize && !isFilled) || isLast) {
            segments.push({
                ratioFrom,
                ratioTo: item.ratio,
                count: localCountSum,
            });
            localCountSum = item.count;
            ratioFrom = item.ratio;
        } else {
            localCountSum = nextSum;
        }
    });

    return segments;
}

export function getPredictPrices(price: number, ratioFrom: number, ratioTo: number): NeuroVision {
    const low = price * ratioFrom;
    const high = price * ratioTo;
    const avg = (low + high) / 2;

    return { low, high, avg };
}

export function timeToNow(time: number): string {
    const duration = new Date().getTime() - time;
    const result = [];

    const toNow = {
        d: Math.floor(duration / (1000 * 60 * 60 * 24)),
        h: Math.floor((duration % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60)),
        m: Math.floor((duration % (1000 * 60 * 60)) / (1000 * 60)),
        s: Math.floor((duration % (1000 * 60)) / 1000),
    };

    for (const [key, value] of Object.entries(toNow)) {
        if (!value) continue;
        result.push(`${value}${key}`);
    }

    return result.join(' ');
}

export function printStatus(statuses: { totalTime; time; error; iterations; speed }[]) {
    const last = [...statuses].pop();
    const error = math.toFixed(last.error * 100);
    const accuracy = math.toFixed(100 - error);

    const statusRows = [];
    const rows = [];

    for (let i = 0; i < 5; i++) {
        const status = statuses[i];

        statusRows.push([
            '|',
            `${status?.iterations || ''}`.padStart(10),
            '|',
            `${status?.error || ''}`.padStart(29),
            '|',
            `${status?.time || ''}`.padStart(9),
            '|',
            `${status?.speed || ''}`.padStart(19),
            '|',
        ]);
    }

    rows.push(
        ['+------------+-------------------------------+-----------+---------------------+'],
        ['| Iterations |                Training Error | Exec.Time | Speed, iterations/s |'],
        ['+------------+-------------------------------+-----------+---------------------+'],
        ...statusRows,
        ['+------------+-------------------------------+-----------+---------------------+'],
        [` Total time: ${last.totalTime}`.padEnd(25), `Accuracy: ${accuracy}%  Error: ${error}% `.padStart(54)],
    );

    if (last.iterations > 10) {
        readline.cursorTo(process.stdout, 0, process.stdout.rows - rows.length - 2);
        readline.clearScreenDown(process.stdout);
    }

    for (const row of rows) {
        process.stdout.write(row.join(' '));
        process.stdout.write('\n');
    }
    process.stdout.write('\n');
}
