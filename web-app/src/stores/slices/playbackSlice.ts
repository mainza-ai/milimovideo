import type { StateCreator } from 'zustand';
import type { TimelineState, PlaybackSlice } from '../types';

export const createPlaybackSlice: StateCreator<TimelineState, [], [], PlaybackSlice> = (set) => ({
    currentTime: 0,
    isPlaying: false as boolean,
    setCurrentTime: (t) => set({ currentTime: t }),
    setIsPlaying: (p) => set({ isPlaying: p }),
});
