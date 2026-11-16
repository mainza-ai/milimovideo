import { describe, expect, it } from 'vitest';
import { bladeSourceFrame, rollCut, shotDurationFrames, slipShot, trimShot } from './ops';

const shot = { numFrames: 100, trimIn: 10, trimOut: 20 };

describe('timeline ops', () => {
    it('trims in and out without crossing the source bounds', () => {
        expect(trimShot(shot, 'in', 15)).toEqual({ trimIn: 25, trimOut: 20 });
        expect(trimShot(shot, 'in', -50)).toEqual({ trimIn: 0, trimOut: 20 });
        expect(trimShot(shot, 'out', 10)).toEqual({ trimIn: 10, trimOut: 10 });
        expect(trimShot(shot, 'out', -200)).toEqual({ trimIn: 10, trimOut: 89 });
        expect(trimShot(shot, 'in', 999)).toEqual({ trimIn: 79, trimOut: 20 });
    });

    it('slips without changing duration', () => {
        const result = slipShot(shot, 5);
        expect(result).toEqual({ trimIn: 15, trimOut: 15 });
        expect(shotDurationFrames({ ...shot, ...result! })).toBe(shotDurationFrames(shot));
        expect(slipShot(shot, -20)).toBeNull();
        expect(slipShot(shot, 0)).toBeNull();
    });

    it('rolls a cut within both clips', () => {
        const left = { numFrames: 100, trimIn: 0, trimOut: 20 };
        const right = { numFrames: 100, trimIn: 10, trimOut: 0 };
        expect(rollCut(left, right, 10)).toEqual({
            left: { trimIn: 0, trimOut: 10 },
            right: { trimIn: 20, trimOut: 0 },
        });
        expect(rollCut(left, right, -20)).toEqual({
            left: { trimIn: 0, trimOut: 30 },
            right: { trimIn: 0, trimOut: 0 },
        });
        expect(rollCut(left, right, 0)).toBeNull();
    });

    it('blades only inside the trimmed clip', () => {
        expect(bladeSourceFrame(shot, 1, 25)).toBe(25);
        expect(bladeSourceFrame(shot, 0, 25)).toBeNull();
        expect(bladeSourceFrame(shot, 3, 25)).toBeNull();
    });
});
