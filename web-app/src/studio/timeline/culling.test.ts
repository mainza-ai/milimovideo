import { describe, expect, it } from 'vitest';
import { cullTimelineClips } from './culling';
import type { TimelineClip } from '../../utils/timelineUtils';

function clip(index: number): TimelineClip {
    return {
        id: `clip-${index}`,
        start: index * 2,
        duration: 2,
        end: index * 2 + 2,
        track: 0,
        shot: { id: `clip-${index}` },
    };
}

describe('timeline culling', () => {
    it('mounts only the visible slice of a long cut', () => {
        const clips = Array.from({ length: 200 }, (_, index) => clip(index));
        const visible = cullTimelineClips(clips, 400, 1200, 48);
        expect(visible.length).toBeGreaterThan(0);
        expect(visible.length).toBeLessThan(40);
    });

    it('keeps overscan clips at the edges', () => {
        const clips = Array.from({ length: 10 }, (_, index) => clip(index));
        const visible = cullTimelineClips(clips, 0, 200, 48, 200);
        expect(visible[0].id).toBe('clip-0');
    });
});
