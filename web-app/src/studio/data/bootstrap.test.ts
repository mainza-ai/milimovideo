import { afterEach, describe, expect, it, vi } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { useStudioProjectBootstrap } from './useStudioData';
import { useStudioDocumentStore } from '../state/documentStore';
import { DEFAULT_PROJECT } from '../../stores/slices/projectSlice';

function json(data: unknown): Response {
    return new Response(JSON.stringify(data), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
    });
}

const project = (id: string, name: string, updated_at: number) => ({
    id, name, updated_at, shots: [], scenes: [], captions: [], revision: 0,
});

describe('useStudioProjectBootstrap', () => {
    afterEach(() => {
        vi.restoreAllMocks();
        useStudioDocumentStore.setState({ project: DEFAULT_PROJECT });
    });

    it('opens the newest real project instead of the local placeholder', async () => {
        const loadProject = vi.fn(async () => undefined);
        useStudioDocumentStore.setState({
            project: { ...DEFAULT_PROJECT, id: 'default', shots: [] },
            loadProject: loadProject as never,
        });
        vi.stubGlobal('fetch', vi.fn(async () => json([
            project('default', 'Untitled Project', 300),
            project('newer', 'Newest Film', 200),
            project('older', 'Old Film', 100),
        ])));

        renderHook(() => useStudioProjectBootstrap());

        await waitFor(() => expect(loadProject).toHaveBeenCalledWith('newer'));
    });

    it('keeps an explicitly opened server project', async () => {
        const loadProject = vi.fn(async () => undefined);
        useStudioDocumentStore.setState({
            project: { ...DEFAULT_PROJECT, id: 'mine', shots: [] },
            loadProject: loadProject as never,
        });
        vi.stubGlobal('fetch', vi.fn(async () => json([
            project('mine', 'My Film', 100),
            project('other', 'Other Film', 900),
        ])));

        renderHook(() => useStudioProjectBootstrap());
        await new Promise((resolve) => setTimeout(resolve, 100));

        expect(loadProject).not.toHaveBeenCalled();
    });
});
