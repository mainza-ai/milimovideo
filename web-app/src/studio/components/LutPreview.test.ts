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
});
