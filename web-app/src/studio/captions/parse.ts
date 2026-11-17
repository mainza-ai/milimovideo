import type { CaptionCue } from '../../stores/types';
import { sanitizeCues } from './cues';

const TIME_RE = /(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})/;

function toSeconds(stamp: string): number | null {
    const match = stamp.match(TIME_RE);
    if (!match) return null;
    const [, hours, minutes, seconds, fraction] = match;
    return Number(hours) * 3600 + Number(minutes) * 60 + Number(seconds)
        + Number(fraction.padEnd(3, '0')) / 1000;
}

let sequence = 0;

/**
 * Parse SRT or WebVTT into caption cues at project fps.
 * Cues are sanitized and sorted, reusing the manual-cue path.
 */
export function parseCaptions(text: string, fps: number): CaptionCue[] {
    const normalized = text.replace(/\r\n/g, '\n').replace(/^\uFEFF/, '');
    const blocks = normalized.split(/\n{2,}/);
    const cues: CaptionCue[] = [];

    for (const block of blocks) {
        const lines = block.split('\n').map((line) => line.trimEnd()).filter(Boolean);
        if (lines.length === 0) continue;
        if (/^WEBVTT/i.test(lines[0])) continue;

        const timeIndex = lines.findIndex((line) => line.includes('-->'));
        if (timeIndex === -1) continue;
        const [startRaw, endRaw] = lines[timeIndex].split('-->');
        const start = toSeconds(startRaw || '');
        const end = toSeconds(endRaw || '');
        if (start == null || end == null) continue;
        const body = lines.slice(timeIndex + 1).join(' ').replace(/<[^>]+>/g, '').trim();
        if (!body) continue;

        sequence += 1;
        cues.push({
            id: `import_${Date.now()}_${sequence}`,
            startFrame: Math.max(0, Math.round(start * fps)),
            endFrame: Math.max(1, Math.round(end * fps)),
            text: body,
        });
    }

    return sanitizeCues(cues);
}
