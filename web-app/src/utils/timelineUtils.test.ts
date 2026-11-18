import { describe, expect, it } from 'vitest';
import { computeTimelineLayout, timelineThumbnail } from './timelineUtils';

function project(shots: Array<Record<string, unknown>>) {
    return { fps: 25, shots };
}

function shot(id: string, overrides: Record<string, unknown> = {}) {
    return {
        id,
        trackIndex: 0,
        numFrames: 25,
        trimIn: 0,
        trimOut: 0,
        speed: 1,
        startFrame: 0,
        ...overrides,
    };
}

describe('computeTimelineLayout transitions', () => {
    it('overlaps a V1 clip by its transition and keeps a magnetic tail', () => {
        const layout = computeTimelineLayout(project([
            shot('a'),
            shot('b', { transitionIn: 5 }),
            shot('c'),
        ]));

        expect(layout[0]).toMatchObject({ start: 0, end: 1 });
        expect(layout[1].start).toBeCloseTo(0.8, 5);
        expect(layout[1].end).toBeCloseTo(1.8, 5);
        expect(layout[2].start).toBeCloseTo(1.8, 5);
    });

    it('clamps an oversized transition to half of both clips', () => {
        const layout = computeTimelineLayout(project([
            shot('a'),
            shot('b', { transitionIn: 100 }),
        ]));
        expect(layout[1].start).toBeCloseTo(0.5, 5);
    });

    it('does not overlap the first clip', () => {
        const layout = computeTimelineLayout(project([shot('a', { transitionIn: 10 })]));
        expect(layout[0].start).toBe(0);
        expect(layout[0].end).toBe(1);
    });
});

describe('timelineThumbnail', () => {
    it('prefers the poster and falls back to still media', () => {
        expect(timelineThumbnail({ thumbnailUrl: '/thumb.jpg', videoUrl: '/clip.mp4' })).toBe('/thumb.jpg');
        expect(timelineThumbnail({ videoUrl: '/still.png' })).toBe('/still.png');
        expect(timelineThumbnail({ videoUrl: '/clip.mp4' })).toBeNull();
    });
});
