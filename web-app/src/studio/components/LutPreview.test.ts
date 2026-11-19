import { describe, expect, it } from 'vitest';
import { parseCubeLut } from './LutPreview';

describe('parseCubeLut', () => {
    it('parses a 2×2×2 cube into an RGB8 texture in red-fastest order', () => {
        const rows = ['LUT_3D_SIZE 2'];
        for (let blue = 0; blue < 2; blue += 1) {
            for (let green = 0; green < 2; green += 1) {
                for (let red = 0; red < 2; red += 1) {
                    rows.push(`${red} ${green} ${blue}`);
                }
            }
        }
        const lut = parseCubeLut(rows.join('\n'));
        expect(lut.size).toBe(2);
        expect(lut.data).toHaveLength(24);
        expect(Array.from(lut.data.slice(0, 3))).toEqual([0, 0, 0]);
        expect(Array.from(lut.data.slice(3, 6))).toEqual([255, 0, 0]);
        expect(Array.from(lut.data.slice(21, 24))).toEqual([255, 255, 255]);
    });

    it('rejects a cube without size or data', () => {
        expect(() => parseCubeLut('LUT_3D_SIZE 2\n0 0 0')).toThrow(/LUT_3D_SIZE/);
    });

    it('parses input domains and ranges', () => {
        const rows = ['LUT_3D_SIZE 2', 'DOMAIN_MIN 0 0.1 0', 'DOMAIN_MAX 1 0.9 1'];
        for (let index = 0; index < 8; index += 1) rows.push('0 0 0');
        expect(parseCubeLut(rows.join('\n')).inputMin).toEqual([0, 0.1, 0]);

        const ranged = ['LUT_3D_SIZE 2', 'LUT_3D_INPUT_RANGE 0 2'];
        for (let index = 0; index < 8; index += 1) ranged.push('0 0 0');
        const parsed = parseCubeLut(ranged.join('\n'));
        expect(parsed.inputMin).toEqual([0, 0, 0]);
        expect(parsed.inputMax).toEqual([2, 2, 2]);
    });

    it('rejects 1D LUTs explicitly', () => {
        expect(() => parseCubeLut('LUT_1D_SIZE 16\n0 0 0')).toThrow(/1D/);
    });
});
