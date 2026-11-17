import { describe, expect, it } from 'vitest';
import {
    opacityAtFrame, removeKeyframeAt, sanitizeKeyframes, upsertKeyframe,
} from './keyframes';

describe('opacity keyframes', () => {
    it('sanitizes, dedupes and sorts', () => {
        const points = sanitizeKeyframes([
            { frame: 10, opacity: 2 },
            { frame: 0, opacity: -1 },
            { frame: 10, opacity: 0.5 },
        ]);
        expect(points).toHaveLength(2);
        expect(points[0]).toMatchObject({ frame: 0, opacity: 0 });
        expect(points[1]).toMatchObject({ frame: 10, opacity: 0.5 });
    });

    it('upserts and removes by frame', () => {
        const points = upsertKeyframe([{ frame: 0, opacity: 1 }], { frame: 12, opacity: 0.25 });
        expect(points).toHaveLength(2);
        const remaining = removeKeyframeAt(points, 0);
        expect(remaining).toHaveLength(1);
        expect(remaining[0]).toMatchObject({ frame: 12, opacity: 0.25 });
    });

    it('interpolates and holds at the ends', () => {
        const points = [{ frame: 0, opacity: 0 }, { frame: 10, opacity: 1 }];
        expect(opacityAtFrame(points, -5)).toBe(0);
        expect(opacityAtFrame(points, 0)).toBe(0);
        expect(opacityAtFrame(points, 5)).toBeCloseTo(0.5);
        expect(opacityAtFrame(points, 10)).toBe(1);
        expect(opacityAtFrame(points, 99)).toBe(1);
        expect(opacityAtFrame([], 4)).toBe(1);
    });
});
