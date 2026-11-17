import type { OpacityKeyframe } from '../../stores/types';

export function sanitizeKeyframes(keyframes: OpacityKeyframe[]): OpacityKeyframe[] {
    const points = new Map<number, OpacityKeyframe>();
    for (const keyframe of keyframes || []) {
        const frame = Math.max(0, Math.round(keyframe.frame || 0));
        points.set(frame, {
            frame,
            opacity: Math.max(0, Math.min(1, Number(keyframe.opacity ?? 1))),
            scale: Math.max(0.1, Math.min(4, Number(keyframe.scale ?? 1))),
            x: Math.max(-1, Math.min(1, Number(keyframe.x ?? 0))),
            y: Math.max(-1, Math.min(1, Number(keyframe.y ?? 0))),
            volume: Math.max(0, Math.min(2, Number(keyframe.volume ?? 1))),
        });
    }
    return [...points.values()].sort((a, b) => a.frame - b.frame);
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
