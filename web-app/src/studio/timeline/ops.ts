import type { Shot } from '../../stores/types';

/**
 * Frame-accurate timeline operations. These are pure: they return the new shot
 * values and never touch the server. V1 is magnetic, so changing a trim ripples
 * downstream clips by construction of computeTimelineLayout().
 */

export interface TrimResult {
    trimIn: number;
    trimOut: number;
}

const MIN_VISIBLE_FRAMES = 1;

function clamp(value: number, min: number, max: number): number {
    return Math.max(min, Math.min(max, value));
}

export function shotDurationFrames(shot: Pick<Shot, 'numFrames' | 'trimIn' | 'trimOut'>): number {
    return Math.max(MIN_VISIBLE_FRAMES, (shot.numFrames || 0) - (shot.trimIn || 0) - (shot.trimOut || 0));
}

export function trimShot(
    shot: Pick<Shot, 'numFrames' | 'trimIn' | 'trimOut'>,
    edge: 'in' | 'out',
    deltaFrames: number,
): TrimResult | null {
    const numFrames = shot.numFrames || 0;
    const trimIn = shot.trimIn || 0;
    const trimOut = shot.trimOut || 0;
    if (numFrames <= MIN_VISIBLE_FRAMES) return null;

    if (edge === 'in') {
        const maxIn = Math.max(0, numFrames - trimOut - MIN_VISIBLE_FRAMES);
        const next = clamp(trimIn + Math.round(deltaFrames), 0, maxIn);
        if (next === trimIn) return null;
        return { trimIn: next, trimOut };
    }

    const maxOut = Math.max(0, numFrames - trimIn - MIN_VISIBLE_FRAMES);
    const next = clamp(trimOut - Math.round(deltaFrames), 0, maxOut);
    if (next === trimOut) return null;
    return { trimIn, trimOut: next };
}

/**
 * Slip keeps the clip's position and duration: the source window moves inside
 * the take. trimIn and trimOut shift in opposite directions by the same delta.
 */
export function slipShot(
    shot: Pick<Shot, 'numFrames' | 'trimIn' | 'trimOut'>,
    deltaFrames: number,
): TrimResult | null {
    const trimIn = shot.trimIn || 0;
    const trimOut = shot.trimOut || 0;
    const numFrames = shot.numFrames || 0;
    const delta = Math.round(deltaFrames);
    if (delta === 0) return null;

    const nextIn = trimIn + delta;
    const nextOut = trimOut - delta;
    if (nextIn < 0 || nextOut < 0) return null;
    if (numFrames - nextIn - nextOut < MIN_VISIBLE_FRAMES) return null;
    return { trimIn: nextIn, trimOut: nextOut };
}

/**
 * Roll moves one cut between two adjacent clips: the left clip gives/takes
 * frames at its out point while the right clip does the same at its in point.
 */
export function rollCut(
    left: Pick<Shot, 'numFrames' | 'trimIn' | 'trimOut'>,
    right: Pick<Shot, 'numFrames' | 'trimIn' | 'trimOut'>,
    deltaFrames: number,
): { left: TrimResult; right: TrimResult } | null {
    const delta = Math.round(deltaFrames);
    if (delta === 0) return null;

    const leftTrimIn = left.trimIn || 0;
    const leftTrimOut = left.trimOut || 0;
    const rightTrimIn = right.trimIn || 0;
    const leftMaxOut = Math.max(0, (left.numFrames || 0) - leftTrimIn - MIN_VISIBLE_FRAMES);
    const rightMaxIn = Math.max(0, (right.numFrames || 0) - (right.trimOut || 0) - MIN_VISIBLE_FRAMES);

    // Positive delta moves the cut later: left grows (trimOut falls) and the
    // right clip starts later (trimIn rises), keeping the pair adjacent.
    const lower = Math.max(-rightTrimIn, leftTrimOut - leftMaxOut);
    const upper = Math.min(leftTrimOut, rightMaxIn - rightTrimIn);
    const applied = clamp(delta, lower, upper);
    if (applied === 0) return null;

    return {
        left: { trimIn: leftTrimIn, trimOut: leftTrimOut - applied },
        right: { trimIn: rightTrimIn + applied, trimOut: right.trimOut || 0 },
    };
}

export function bladeSourceFrame(
    shot: Pick<Shot, 'numFrames' | 'trimIn' | 'trimOut'>,
    playheadInClipSeconds: number,
    fps: number,
): number | null {
    const duration = shotDurationFrames(shot);
    const frame = Math.round(playheadInClipSeconds * Math.max(1, fps));
    if (frame <= 0 || frame >= duration) return null;
    return frame;
}
