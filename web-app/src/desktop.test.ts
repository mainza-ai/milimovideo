import { afterEach, describe, expect, it, vi } from 'vitest';
import { openTextFile } from './desktop';

describe('desktop file open', () => {
    afterEach(() => {
        delete (window as unknown as { __TAURI__?: unknown }).__TAURI__;
        vi.restoreAllMocks();
    });

    it('uses the native dialog and reads the selected file', async () => {
        const invoke = vi.fn(async () => '{"prompt":"x"}');
        const open = vi.fn(async () => '/tmp/recipe.json');
        (window as unknown as { __TAURI__?: unknown }).__TAURI__ = {
            dialog: { open, save: vi.fn() },
            core: { invoke },
        };

        await expect(openTextFile()).resolves.toBe('{"prompt":"x"}');
        expect(open).toHaveBeenCalledWith({
            multiple: false,
            filters: [{ name: 'JSON', extensions: ['json'] }],
        });
        expect(invoke).toHaveBeenCalledWith('read_text_file', { path: '/tmp/recipe.json' });
    });

    it('returns null when the native picker is cancelled', async () => {
        const invoke = vi.fn();
        (window as unknown as { __TAURI__?: unknown }).__TAURI__ = {
            dialog: { open: vi.fn(async () => null), save: vi.fn() },
            core: { invoke },
        };

        await expect(openTextFile()).resolves.toBeNull();
        expect(invoke).not.toHaveBeenCalled();
    });
});
