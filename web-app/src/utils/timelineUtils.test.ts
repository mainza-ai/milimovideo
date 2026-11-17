import { describe, expect, it } from 'vitest';
import { computeTimelineLayout, computeTotalDuration } from './timelineUtils';

describe('computeTimelineLayout', () => {
    it('keeps V1 magnetic from array order and ignores startFrame', () => {
        const project = {
            fps: 25,
            shots: [
                { id: 'a', trackIndex: 0, numFrames: 50, startFrame: 999, trimIn: 0, trimOut: 0 },
                { id: 'b', trackIndex: 0, numFrames: 25, startFrame: 0, trimIn: 0, trimOut: 0 },
            ],
        };
        const clips = computeTimelineLayout(project);
        expect(clips.map(c => [c.id, c.start, c.duration])).toEqual([
            ['a', 0, 2],
            ['b', 2, 1],
        ]);
        expect(computeTotalDuration(clips)).toBe(3);
    });

    it('honors trimIn/trimOut and free-track startFrame', () => {
        const project = {
            fps: 25,
            shots: [
                { id: 'v1', trackIndex: 0, numFrames: 100, trimIn: 10, trimOut: 10 },
                { id: 'v2', trackIndex: 1, numFrames: 25, startFrame: 75, trimIn: 0, trimOut: 0 },
            ],
        };
        const clips = computeTimelineLayout(project);
        const v1 = clips.find(c => c.id === 'v1')!;
        const v2 = clips.find(c => c.id === 'v2')!;
        expect(v1.duration).toBe(80 / 25);
        expect(v2.start).toBe(3);
    });

    it('never produces a zero-length clip', () => {
        const clips = computeTimelineLayout({
            fps: 25,
            shots: [{ id: 'x', trackIndex: 0, numFrames: 10, trimIn: 5, trimOut: 5 }],
        });
        expect(clips[0].duration).toBe(1 / 25);
    });
});
