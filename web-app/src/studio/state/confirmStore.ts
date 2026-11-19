import { create } from 'zustand';

export interface ConfirmOptions {
    title: string;
    body?: string;
    confirmLabel?: string;
    cancelLabel?: string;
    tone?: 'default' | 'danger';
}

interface ConfirmState {
    current: { options: ConfirmOptions; resolve: (value: boolean) => void } | null;
    request: (options: ConfirmOptions) => Promise<boolean>;
    close: (value: boolean) => void;
}

export const useConfirmStore = create<ConfirmState>((set, get) => ({
    current: null,
    request: (options) => new Promise((resolve) => {
        set({ current: { options, resolve } });
    }),
    close: (value) => {
        const current = get().current;
        set({ current: null });
        current?.resolve(value);
    },
}));

/** In-app replacement for `window.confirm`. */
export function requestConfirm(options: ConfirmOptions): Promise<boolean> {
    return useConfirmStore.getState().request(options);
}
