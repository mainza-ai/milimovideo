export interface ShortCue {
    start: number;
    end: number;
    text: string;
}

export interface ShortCandidate {
    id: string;
    start: number;
    end: number;
    duration: number;
    score: number;
    text: string;
}

const HOOK_WORDS = new Set([
    'how', 'why', 'what', 'secret', 'never', 'always', 'best', 'worst', 'mistake',
    'truth', 'because', 'first', 'finally', 'stop', 'start', 'money', 'love',
]);

/**
 * Build short candidates from transcript cues: sliding windows of 15–45s with
 * a local heuristics score (density, hook words, numbers). No model calls.
 */
export function buildCandidates(
    cues: ShortCue[],
    minSeconds = 15,
    maxSeconds = 45,
): ShortCandidate[] {
    const ordered = [...cues]
        .filter((cue) => cue.text.trim())
        .map((cue) => ({
            ...cue,
            start: Math.max(0, cue.start),
            end: Math.max(cue.start + 0.2, cue.end),
        }))
        .sort((a, b) => a.start - b.start);

    const candidates: ShortCandidate[] = [];
    for (let startIndex = 0; startIndex < ordered.length; startIndex += 1) {
        const start = ordered[startIndex].start;
        let end = start;
        let lastEnd = start;
        const words: string[] = [];

        for (let index = startIndex; index < ordered.length; index += 1) {
            const cue = ordered[index];
            if (cue.start - lastEnd > 2.5 && index > startIndex) break;
            if (cue.end - start > maxSeconds) break;
            end = cue.end;
            lastEnd = cue.end;
            words.push(...cue.text.toLowerCase().split(/[^a-z0-9']+/).filter(Boolean));
        }

        const duration = end - start;
        if (duration < minSeconds) continue;

        const hooks = words.filter((word) => HOOK_WORDS.has(word)).length;
        const numbers = words.filter((word) => /^\d+$/.test(word)).length;
        const density = words.length / Math.max(1, duration);
        const score = Math.round((density * 20 + hooks * 8 + numbers * 3) * 10) / 10;
        const text = ordered
            .filter((cue) => cue.start >= start && cue.end <= end)
            .map((cue) => cue.text)
            .join(' ');

        candidates.push({
            id: `cand_${Math.round(start * 1000)}_${Math.round(end * 1000)}`,
            start,
            end,
            duration,
            score,
            text,
        });
    }

    const seen = new Set<string>();
    return candidates
        .sort((a, b) => b.score - a.score)
        .filter((candidate) => {
            if (seen.has(candidate.id)) return false;
            seen.add(candidate.id);
            return true;
        })
        .slice(0, 12);
}
