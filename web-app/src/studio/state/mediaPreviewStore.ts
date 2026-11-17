import { create } from 'zustand';

export interface MediaPreviewItem {
    url: string;
    title?: string;
    subtitle?: string;
    kind?: 'image' | 'video' | 'audio' | 'text';
}

interface MediaPreviewState {
    item: MediaPreviewItem | null;
    open: (item: MediaPreviewItem) => void;
    close: () => void;
}

/** Global media viewer shared by Library, Story, Generate, Edit, Review. */
export const useMediaPreview = create<MediaPreviewState>((set) => ({
    item: null,
    open: (item) => set({ item }),
    close: () => set({ item: null }),
}));
