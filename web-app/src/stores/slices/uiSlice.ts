import type { StateCreator } from 'zustand';
import type { TimelineState, UISlice } from '../types';
import { v4 as uuidv4 } from 'uuid';

export const createUISlice: StateCreator<TimelineState, [], [], UISlice> = (set, get) => ({
    // Notifications
    toasts: [] as TimelineState['toasts'],
    addToast: (message, type = 'info') => {
        const id = uuidv4();
        set(state => ({
            toasts: [...state.toasts, { id, message, type }]
        }));
        // Auto-dismiss after 3 seconds
        setTimeout(() => {
            set(state => ({
                toasts: state.toasts.filter(t => t.id !== id)
            }));
        }, 3000);
    },
    removeToast: (id) => set(state => ({ toasts: state.toasts.filter(t => t.id !== id) })),

    // UI State
    transientDuration: null as number | null,
    setTransientDuration: (d) => set({ transientDuration: d }),

    // View Mode
    viewMode: 'timeline',
    setViewMode: (mode) => set({ viewMode: mode }),

    // Selection
    selectedShotId: 'shot-init', // Default, overwritten by load
    selectShot: (id) => set({ selectedShotId: id }),

    // In-Painting UI
    isEditing: false as boolean,
    setEditing: (e) => set({ isEditing: e, isPlaying: !e ? get().isPlaying : false }),
});
