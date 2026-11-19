import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { temporal } from 'zundo';
import { useShallow } from 'zustand/react/shallow';
import type { TimelineState } from '../../stores/types';
import { documentSignature } from '../../stores/saveStatus';

import { createProjectSlice } from '../../stores/slices/projectSlice';
import { createShotSlice } from '../../stores/slices/shotSlice';
import { createPlaybackSlice } from '../../stores/slices/playbackSlice';
import { createUISlice } from '../../stores/slices/uiSlice';
import { createTrackSlice } from '../../stores/slices/trackSlice';
import { createElementSlice } from '../../stores/slices/elementSlice';
import { createServerSlice } from '../../stores/slices/serverSlice';
import { createModelSlice } from '../../stores/slices/modelSlice';

/**
 * Studio-owned document store.
 *
 * The legacy editor keeps its own store at `/`; the studio no longer shares its
 * document/save/undo state. Both talk to the same backend contract. The slice
 * implementations are reused so the migration is behavior-preserving while the
 * source of truth is now separated per route.
 */
export const useStudioDocumentStore = create<TimelineState>()(
    temporal(
        persist(
            (...args) => ({
                ...createProjectSlice(...args),
                ...createShotSlice(...args),
                ...createPlaybackSlice(...args),
                ...createUISlice(...args),
                ...createTrackSlice(...args),
                ...createElementSlice(...args),
                ...createServerSlice(...args),
                ...createModelSlice(...args),
            }),
            {
                name: 'milimo-studio-document',
                partialize: (state: TimelineState) => ({
                    project: state.project,
                    savedBaseline: state.savedBaseline,
                    saveStatus: state.saveStatus === 'saved'
                        || state.saveStatus === 'dirty'
                        || state.saveStatus === 'idle'
                        || state.saveStatus === 'conflict'
                        ? state.saveStatus
                        : 'dirty',
                }),
                merge: (persistedState: unknown, currentState: TimelineState) => {
                    const persisted = (persistedState as Partial<TimelineState>) || {};
                    const persistedProject = persisted.project;
                    const project = persistedProject
                        ? {
                            ...persistedProject,
                            shots: (persistedProject.shots || []).map((shot) => ({
                                ...shot,
                                isGenerating: false,
                                status: shot.status === 'generating' ? 'pending' : shot.status,
                                progress: undefined,
                                statusMessage: undefined,
                                currentPrompt: undefined,
                                etaSeconds: undefined,
                            })),
                        }
                        : currentState.project;
                    return {
                        ...currentState,
                        ...persisted,
                        project,
                        toasts: [] as TimelineState['toasts'],
                        isPlaying: false as boolean,
                        isEditing: false as boolean,
                        transientDuration: null as TimelineState['transientDuration'],
                        generatingElementIds: {} as TimelineState['generatingElementIds'],
                    };
                },
            },
        ),
        {
            limit: 30,
            partialize: (state: TimelineState) => ({ project: state.project }),
            equality: (past, current) => {
                const strip = (state: Pick<TimelineState, 'project'>) => JSON.stringify({
                    ...state.project,
                    shots: (state.project.shots || []).map(({
                        isGenerating, progress, statusMessage, currentPrompt,
                        etaSeconds, enhancedPromptResult, lastJobId, ...shot
                    }) => shot),
                });
                return strip(past) === strip(current);
            },
        },
    ),
);

// Derive Saved/Unsaved from the last successful save, so undo, redo, board
// applies, and zundo restores are all truthful. A conflict stays sticky until
// the user reloads or explicitly overwrites.
useStudioDocumentStore.subscribe((state, previous) => {
    if (state.project === previous.project) return;
    if (state.saveStatus === 'conflict' || state.saveStatus === 'saving') return;
    const next = state.savedBaseline
        ? (documentSignature(state.project) === state.savedBaseline ? 'saved' : 'dirty')
        : 'dirty';
    if (state.saveStatus !== next) {
        useStudioDocumentStore.setState({ saveStatus: next });
    }
});

export function useStudioDocument() {
    return useStudioDocumentStore(useShallow((state) => ({
        project: state.project,
        selectedShotId: state.selectedShotId,
        updateShot: state.updateShot,
        patchShot: state.patchShot,
        addShot: state.addShot,
        deleteShot: state.deleteShot,
        splitShot: state.splitShot,
        reorderShots: state.reorderShots,
        setProject: state.setProject,
        saveProject: state.saveProject,
        saveStatus: state.saveStatus,
        saveError: state.saveError,
    })));
}

export const studioDocument = {
    get project() {
        return useStudioDocumentStore.getState().project;
    },
    updateShot: (...args: Parameters<ReturnType<typeof useStudioDocumentStore.getState>['updateShot']>) =>
        useStudioDocumentStore.getState().updateShot(...args),
    saveProject: () => useStudioDocumentStore.getState().saveProject(),
};

export type * from '../../stores/types';
