import type { StateCreator } from 'zustand';
import type { TimelineState, TrackSlice } from '../types';

export const createTrackSlice: StateCreator<TimelineState, [], [], TrackSlice> = (set) => ({
    trackStates: {
        0: { muted: false, locked: false, hidden: false },
        1: { muted: false, locked: false, hidden: false },
        2: { muted: false, locked: false, hidden: false },
    },

    toggleTrackMute: (idx) => set(state => ({
        trackStates: {
            ...state.trackStates,
            [idx]: { ...state.trackStates[idx], muted: !state.trackStates[idx]?.muted }
        }
    })),
    toggleTrackLock: (idx) => set(state => ({
        trackStates: {
            ...state.trackStates,
            [idx]: { ...state.trackStates[idx], locked: !state.trackStates[idx]?.locked }
        }
    })),
    toggleTrackHidden: (idx) => set(state => ({
        trackStates: {
            ...state.trackStates,
            [idx]: { ...state.trackStates[idx], hidden: !state.trackStates[idx]?.hidden }
        }
    })),
});
