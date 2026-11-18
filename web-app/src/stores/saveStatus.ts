import type { Shot } from './types';

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
]);

export function isDocumentChange(updates: Partial<Shot>): boolean {
    return Object.keys(updates).some((key) => !RUNTIME_SHOT_KEYS.has(key as keyof Shot));
}

export function shotStatusLabel(status?: string): string {
    switch (status) {
        case 'generating': return 'Running';
        case 'completed': return 'Done';
        case 'failed': return 'Failed';
        case 'pending':
        case undefined:
        case '': return 'Pending';
        default: return status.charAt(0).toUpperCase() + status.slice(1);
    }
}

export function shotHasTake(shot: Pick<Shot, 'status' | 'videoUrl' | 'thumbnailUrl'>): boolean {
    return shot.status === 'completed' || !!shot.videoUrl || !!shot.thumbnailUrl;
}
