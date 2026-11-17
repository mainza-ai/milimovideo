import { describe, expect, it } from 'vitest';
import { parseCaptions } from './parse';

describe('caption import', () => {
    it('parses SRT blocks into frames', () => {
        const cues = parseCaptions(`1
00:00:01,000 --> 00:00:02,500
First line

2
00:00:03,000 --> 00:00:04,000
Second <i>line</i>
`, 25);
        expect(cues).toHaveLength(2);
        expect(cues[0]).toMatchObject({ startFrame: 25, endFrame: 63, text: 'First line' });
        expect(cues[1]).toMatchObject({ startFrame: 75, endFrame: 100, text: 'Second line' });
    });

    it('parses WebVTT and skips headers', () => {
        const cues = parseCaptions(`WEBVTT

00:00:00.500 --> 00:00:01.500
Hello there
`, 25);
        expect(cues).toHaveLength(1);
        expect(cues[0]).toMatchObject({ startFrame: 13, endFrame: 38, text: 'Hello there' });
    });

    it('returns an empty list for invalid input', () => {
        expect(parseCaptions('not captions', 25)).toEqual([]);
        expect(parseCaptions('', 25)).toEqual([]);
    });
});
