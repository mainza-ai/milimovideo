import { describe, expect, it } from 'vitest';
import { cueAtFrame, removeCue, sanitizeCues, upsertCue } from './cues';

describe('caption cues', () => {
    it('sorts, clamps and drops blank cues', () => {
        const cues = sanitizeCues([
            { id: 'b', startFrame: 50, endFrame: 40, text: '  later  ' },
            { id: 'a', startFrame: -5, endFrame: 10, text: 'first' },
            { id: 'blank', startFrame: 0, endFrame: 5, text: '   ' },
        ]);
        expect(cues.map((cue) => cue.id)).toEqual(['a', 'b']);
        expect(cues[0]).toMatchObject({ startFrame: 0, text: 'first' });
        expect(cues[1]).toMatchObject({ startFrame: 50, endFrame: 51 });
    });

    it('upserts by id without duplicating', () => {
        const base = [{ id: 'a', startFrame: 0, endFrame: 25, text: 'one' }];
        const updated = upsertCue(base, { id: 'a', startFrame: 0, endFrame: 25, text: 'two' });
        expect(updated).toHaveLength(1);
        expect(updated[0].text).toBe('two');
        const added = upsertCue(updated, { id: 'b', startFrame: 25, endFrame: 50, text: 'three' });
        expect(added.map((cue) => cue.id)).toEqual(['a', 'b']);
    });

    it('removes and finds cues by frame', () => {
        const cues = [
            { id: 'a', startFrame: 0, endFrame: 25, text: 'one' },
            { id: 'b', startFrame: 25, endFrame: 50, text: 'two' },
        ];
        expect(cueAtFrame(cues, 24)?.id).toBe('a');
        expect(cueAtFrame(cues, 25)?.id).toBe('b');
        expect(cueAtFrame(cues, 51)).toBeUndefined();
        expect(removeCue(cues, 'a').map((cue) => cue.id)).toEqual(['b']);
    });
});
