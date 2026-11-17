import type { OpacityKeyframe } from '../../stores/types';

export function sanitizeKeyframes(keyframes: OpacityKeyframe[]): OpacityKeyframe[] {
    const points = new Map<number, number>();
    for (const keyframe of keyframes || []) {
        const frame = Math.max(0, Math.round(keyframe.frame || 0));
        const opacity = Math.max(0, Math.min(1, Number(keyframe.opacity ?? 1)));
        points.set(frame, opacity);
    }
    return [...points.entries()]
        .sort(([a], [b]) => a - b)
        .map(([frame, opacity]) => ({ frame, opacity }));
}

export function upsertKeyframe(
    keyframes: OpacityKeyframe[],
    keyframe: OpacityKeyframe,
): OpacityKeyframe[] {
    return sanitizeKeyframes([...keyframes, keyframe]);
}

export function removeKeyframeAt(
    keyframes: OpacityKeyframe[],
    frame: number,
): OpacityKeyframe[] {
    return sanitizeKeyframes(keyframes.filter((item) => item.frame !== frame));
}

export function opacityAtFrame(keyframes: OpacityKeyframe[], frame: number): number {
    const points = sanitizeKeyframes(keyframes);
    if (points.length === 0) return 1;
    if (frame <= points[0].frame) return points[0].opacity;
    const last = points[points.length - 1];
    if (frame >= last.frame) return last.opacity;

    for (let index = 0; index < points.length - 1; index += 1) {
        const left = points[index];
        const right = points[index + 1];
        if (frame >= left.frame && frame <= right.frame) {
            const span = Math.max(1, right.frame - left.frame);
            const ratio = (frame - left.frame) / span;
            return left.opacity + (right.opacity - left.opacity) * ratio;
        }
    }
    return last.opacity;
}
