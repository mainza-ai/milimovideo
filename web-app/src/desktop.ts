interface TauriDialog {
    save(options: {
        defaultPath?: string;
        filters?: Array<{ name: string; extensions: string[] }>;
    }): Promise<string | null>;
}

interface TauriGlobal {
    dialog?: TauriDialog;
    core?: {
        invoke<T>(command: string, args?: Record<string, unknown>): Promise<T>;
    };
}

function tauri(): TauriGlobal | null {
    const candidate = (window as unknown as { __TAURI__?: TauriGlobal }).__TAURI__;
    return candidate && candidate.core ? candidate : null;
}

export function isDesktop(): boolean {
    return tauri() !== null;
}

export async function startDesktopBackend(): Promise<string | null> {
    const api = tauri();
    if (!api?.core) return null;
    return api.core.invoke<string>('start_backend');
}

/** Native save in Tauri, browser download everywhere else. */
export async function saveTextFile(
    defaultName: string,
    contents: string,
    filters: Array<{ name: string; extensions: string[] }> = [
        { name: 'JSON', extensions: ['json'] },
    ],
): Promise<'saved' | 'cancelled'> {
    const api = tauri();
    if (api?.dialog && api.core) {
        const path = await api.dialog.save({ defaultPath: defaultName, filters });
        if (!path) return 'cancelled';
        await api.core.invoke('save_text_file', { path, contents });
        return 'saved';
    }

    const url = URL.createObjectURL(new Blob([contents], { type: 'application/json' }));
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = defaultName;
    anchor.click();
    URL.revokeObjectURL(url);
    return 'saved';
}
