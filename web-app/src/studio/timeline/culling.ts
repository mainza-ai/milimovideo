import type { TimelineClip } from '../../utils/timelineUtils';

/** Clips that intersect the horizontal timeline viewport plus overscan. */
export function cullTimelineClips(
    clips: TimelineClip[],
    scrollLeft: number,
    viewportWidth: number,
    pixelsPerSecond: number,
    overscanPx = 400,
): TimelineClip[] {
    const zoom = Math.max(1, pixelsPerSecond);
    const left = scrollLeft - overscanPx;
    const right = scrollLeft + Math.max(1, viewportWidth) + overscanPx;
    return clips.filter((clip) => (
        clip.end * zoom >= left && clip.start * zoom <= right
    ));
}
