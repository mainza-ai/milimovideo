interface TauriDialog {
    save(options: {
        defaultPath?: string;
        filters?: Array<{ name: string; extensions: string[] }>;
    }): Promise<string | null>;
    open(options: {
        multiple?: boolean;
        directory?: boolean;
        filters?: Array<{ name: string; extensions: string[] }>;
    }): Promise<string | string[] | null>;
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

/** Native open in Tauri, browser file picker everywhere else. */
export async function openTextFile(
    filters: Array<{ name: string; extensions: string[] }> = [
        { name: 'JSON', extensions: ['json'] },
    ],
): Promise<string | null> {
    const api = tauri();
    if (api?.dialog && api.core) {
        const selected = await api.dialog.open({ multiple: false, filters });
        const path = Array.isArray(selected) ? selected[0] : selected;
        if (!path) return null;
        return api.core.invoke<string>('read_text_file', { path });
    }

    return new Promise((resolve) => {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = filters
            .flatMap((filter) => filter.extensions.map((extension) => `.${extension}`))
            .join(',');
        input.onchange = async () => {
            resolve(input.files?.[0] ? await input.files[0].text() : null);
        };
        input.oncancel = () => resolve(null);
        input.click();
    });
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
