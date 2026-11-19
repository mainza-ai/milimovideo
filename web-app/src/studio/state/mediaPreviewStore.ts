import { create } from 'zustand';

export interface MediaPreviewItem {
    url: string;
    /** 720p review proxy; the full-resolution URL stays available for delivery. */
    proxyUrl?: string;
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
