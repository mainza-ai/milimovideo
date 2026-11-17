import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { temporal } from 'zundo';
import type { TimelineState } from './types';

// Slices
import { createProjectSlice } from './slices/projectSlice';
import { createShotSlice } from './slices/shotSlice';
import { createPlaybackSlice } from './slices/playbackSlice';
import { createUISlice } from './slices/uiSlice';
import { createTrackSlice } from './slices/trackSlice';
import { createElementSlice } from './slices/elementSlice';
import { createServerSlice } from './slices/serverSlice';
import { createModelSlice } from './slices/modelSlice';

// Export Types for consumers
export * from './types';
const LAST_PROJECT_KEY = 'milimo_last_project_id';

export const getLastProjectId = (): string | null => {
    try {
        return localStorage.getItem(LAST_PROJECT_KEY);
    } catch (e) {
        console.warn('Failed to get last project ID:', e);
        return null;
    }
};

export const useTimelineStore = create<TimelineState>()(
    temporal(
        persist(
            (...a) => ({
                ...createProjectSlice(...a),
                ...createShotSlice(...a),
                ...createPlaybackSlice(...a),
                ...createUISlice(...a),
                ...createTrackSlice(...a),
                ...createElementSlice(...a),
                ...createServerSlice(...a),
                ...createModelSlice(...a),
            }),
            {
                name: 'milimo-timeline-storage',
                partialize: (state: TimelineState) => ({ project: state.project }),
                merge: (persistedState: any, currentState: TimelineState) => {
                    const persisted = (persistedState as Partial<TimelineState>) || {};
                    const persistedProject = persisted.project;
                    const project = persistedProject ? {
                        ...persistedProject,
                        shots: (persistedProject.shots || []).map(shot => ({
                            ...shot,
                            // Job/progress state is server-owned. Persisting it
                            // across a reload is how spinners survive dead jobs.
                            isGenerating: false,
                            status: shot.status === 'generating' ? 'pending' : shot.status,
                            progress: undefined,
                            statusMessage: undefined,
                            currentPrompt: undefined,
                            etaSeconds: undefined,
                        })),
                    } : currentState.project;
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
            }
        ),
        {
            limit: 20,
            partialize: (state: TimelineState) => ({ project: state.project }),
            // Server-driven job progress must not consume undo steps.
            equality: (past, current) => {
                const strip = (state: any) => JSON.stringify({
                    ...state.project,
                    shots: (state.project.shots || []).map(({
                        isGenerating, progress, statusMessage, currentPrompt,
                        etaSeconds, enhancedPromptResult, lastJobId, ...shot
                    }: any) => shot),
                });
                return strip(past) === strip(current);
            },
        }
    )
);
