import { describe, expect, it } from 'vitest';
import { buildCandidates } from './candidates';

function cue(start: number, end: number, text: string) {
    return { start, end, text };
}

describe('short candidates', () => {
    const cues = [
        cue(0, 5, 'Welcome to the studio.'),
        cue(5, 10, 'Today we show the secret of lighting.'),
        cue(10, 15, 'First, measure the key light.'),
        cue(15, 20, 'Then add a rim light.'),
        cue(30, 36, 'This is the best mistake to avoid.'),
        cue(36, 42, 'You will save three hours.'),
    ];

    it('builds windows inside the duration bounds', () => {
        const candidates = buildCandidates(cues, 15, 45);
        expect(candidates.length).toBeGreaterThan(0);
        for (const candidate of candidates) {
            expect(candidate.duration).toBeGreaterThanOrEqual(15);
            expect(candidate.duration).toBeLessThanOrEqual(45);
        }
    });

    it('scores hook and number dense windows higher', () => {
        const candidates = buildCandidates(cues, 15, 45);
        expect(candidates[0].score).toBeGreaterThan(0);
        expect(candidates.some((candidate) => /three|secret|best/i.test(candidate.text))).toBe(true);
    });

    it('respects a long silence as a boundary', () => {
        const candidates = buildCandidates(cues, 15, 45);
        const crossSilence = candidates.find(
            (candidate) => candidate.start === 0 && candidate.end > 20,
        );
        expect(crossSilence).toBeUndefined();
    });
});
