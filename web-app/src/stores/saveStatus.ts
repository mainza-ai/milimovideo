import type { Project, Shot } from './types';

export type SaveStatus = 'idle' | 'dirty' | 'saving' | 'saved' | 'error' | 'conflict';

/**
 * One vocabulary for every save-state surface. `idle` means the document has
 * never been written (for example the built-in placeholder), not "unsaved
 * changes"; mutations move it to `dirty`.
 */
export function saveStatusLabel(status: SaveStatus): string {
    switch (status) {
        case 'saved': return 'Saved';
        case 'saving': return 'Saving…';
        case 'dirty': return 'Unsaved';
        case 'conflict': return 'Save conflict';
        case 'error': return 'Save failed';
        default: return 'Not saved';
    }
}

export function saveStatusTone(status: SaveStatus): 'good' | 'warn' | 'bad' | undefined {
    switch (status) {
        case 'saved': return 'good';
        case 'saving':
        case 'dirty':
        case 'idle': return 'warn';
        case 'conflict':
        case 'error': return 'bad';
        default: return undefined;
    }
}

/** Runtime job fields do not dirty the document. */
export const RUNTIME_SHOT_KEYS = new Set<keyof Shot>([
    'isGenerating',
    'progress',
    'status',
    'statusMessage',
    'stage',
    'etaSeconds',
    'currentPrompt',
    'enhancedPromptResult',
    'lastJobId',
    'selectedTakeId',
    'favoriteTakeId',
    // Server/job-owned media pointers. Generation completion must not mark the
    // project dirty; the backend already wrote the shot row.
    'videoUrl',
    'thumbnailUrl',
]);

export function isDocumentChange(updates: Partial<Shot>): boolean {
    return Object.keys(updates).some((key) => !RUNTIME_SHOT_KEYS.has(key as keyof Shot));
}

type ShotStatusLike = Pick<Shot, 'status' | 'videoUrl' | 'thumbnailUrl'>;

/** A single truth for the card badge and the take count. */
export function effectiveShotStatus(shot: ShotStatusLike): string {
    const status = shot.status || 'pending';
    if (shotHasTake(shot)) {
        return status === 'failed' ? 'failed' : 'completed';
    }
    if (status === 'completed') return 'missing';
    return status;
}

export function shotStatusKey(shot: ShotStatusLike): string {
    return effectiveShotStatus(shot);
}

export function shotStatusLabel(status?: string | ShotStatusLike): string {
    const value = typeof status === 'object' && status !== null
        ? effectiveShotStatus(status)
        : (status || 'pending');
    switch (value) {
        case 'generating': return 'Running';
        case 'completed': return 'Done';
        case 'missing': return 'Missing media';
        case 'failed': return 'Failed';
        case 'pending':
        case undefined:
        case '': return 'Pending';
        default: return value.charAt(0).toUpperCase() + value.slice(1);
    }
}

/**
 * Content signature used for Saved/Unsaved. Runtime job fields, media pointers,
 * and the revision counter are excluded: they are server-owned, and generation
 * completion must not mark the document dirty.
 */
export function documentSignature(project: Project): string {
    const { revision: _revision, shots = [], ...rest } = project;
    const stableShots = shots.map((shot) => {
        const copy: Record<string, unknown> = {};
        for (const [key, value] of Object.entries(shot)) {
            if (RUNTIME_SHOT_KEYS.has(key as keyof Shot)) continue;
            copy[key] = value;
        }
        return copy;
    });
    const text = JSON.stringify({ ...rest, shots: stableShots });
    // Store a compact two-lane FNV-1a hash: the baseline must not double the
    // localStorage footprint of a 500-shot project.
    let first = 0x811c9dc5;
    let second = 0x9e3779b9;
    for (let index = 0; index < text.length; index += 1) {
        const code = text.charCodeAt(index);
        first = Math.imul(first ^ code, 0x01000193);
        second = Math.imul(second ^ code, 0x85ebca6b);
    }
    return `${text.length}:${(first >>> 0).toString(36)}:${(second >>> 0).toString(36)}`;
}

export function shotHasTake(shot: Pick<Shot, 'videoUrl' | 'thumbnailUrl'> & { status?: string }): boolean {
    return !!shot.videoUrl || !!shot.thumbnailUrl;
}
