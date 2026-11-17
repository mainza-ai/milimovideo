import type { CaptionCue } from '../../stores/types';

export function sanitizeCues(cues: CaptionCue[]): CaptionCue[] {
    return [...cues]
        .map((cue) => ({
            ...cue,
            startFrame: Math.max(0, Math.round(cue.startFrame || 0)),
            endFrame: Math.max(
                Math.max(0, Math.round(cue.startFrame || 0)) + 1,
                Math.round(cue.endFrame || 0),
            ),
            text: (cue.text || '').replace(/\s+/g, ' ').trim(),
        }))
        .filter((cue) => cue.text.length > 0)
        .sort((a, b) => a.startFrame - b.startFrame);
}

export function upsertCue(cues: CaptionCue[], cue: CaptionCue): CaptionCue[] {
    const exists = cues.some((item) => item.id === cue.id);
    return sanitizeCues(
        exists ? cues.map((item) => (item.id === cue.id ? cue : item)) : [...cues, cue],
    );
}

export function removeCue(cues: CaptionCue[], cueId: string): CaptionCue[] {
    return cues.filter((cue) => cue.id !== cueId);
}

export function cueAtFrame(cues: CaptionCue[], frame: number): CaptionCue | undefined {
    return cues.find((cue) => frame >= cue.startFrame && frame < cue.endFrame);
}
