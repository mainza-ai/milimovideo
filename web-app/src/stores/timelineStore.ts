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
            }),
            {
                name: 'milimo-timeline-storage',
                partialize: (state: TimelineState) => ({ project: state.project }),
                merge: (persistedState: any, currentState: TimelineState) => ({
                    ...currentState,
                    ...(persistedState as Partial<TimelineState>),
                    // Reset transient UI states on load
                    toasts: [] as TimelineState['toasts'],
                    isPlaying: false as boolean,
                    isEditing: false as boolean,
                    transientDuration: null as TimelineState['transientDuration'],
                    generatingElementIds: {} as TimelineState['generatingElementIds'],
                }),
            }
        ),
        {
            limit: 20,
            partialize: (state: TimelineState) => ({ project: state.project })
        }
    )
);
