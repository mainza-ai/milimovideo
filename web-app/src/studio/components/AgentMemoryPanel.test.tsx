import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { AgentMemoryPanel } from './AgentMemoryPanel';
import { useStudioDocumentStore } from '../state/documentStore';

function json(data: unknown, status = 200): Response {
    return new Response(JSON.stringify(data), {
        status,
        headers: { 'Content-Type': 'application/json' },
    });
}

const entry = {
    id: 'mem-1',
    scope: 'project',
    shot_id: null,
    kind: 'semantic',
    text: 'The rooftops are always wet.',
    metadata: {},
    provenance: 'user',
    trust: 'user',
    pinned: false,
    quarantined: false,
    use_count: 2,
    tokens: 8,
    updated_at: null,
    why: ['match:rooftops'],
};

describe('AgentMemoryPanel', () => {
    beforeEach(() => {
        useStudioDocumentStore.setState((state) => ({
            project: { ...state.project, id: 'proj-1' },
        }));
    });

    afterEach(() => {
        cleanup();
        vi.unstubAllGlobals();
    });

    it('lists project memory with trust and retrieval reasons', async () => {
        vi.stubGlobal('fetch', vi.fn(async () => json({ entries: [entry] })));
        render(<AgentMemoryPanel />);
        expect(await screen.findByText('The rooftops are always wet.')).toBeTruthy();
        expect(screen.getByText(/user · used 2× · 8 tokens/)).toBeTruthy();
        expect(screen.getByText(/match:rooftops/)).toBeTruthy();
    });

    it('shows quarantined entries and exposes approve', async () => {
        const risky = { ...entry, id: 'mem-2', quarantined: true, trust: 'agent', provenance: 'agent' };
        const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
            const url = String(input);
            if (url.includes('/approve')) return json({ ...risky, quarantined: false, trust: 'verified' });
            return json({ entries: [risky] });
        });
        vi.stubGlobal('fetch', fetchMock);
        render(<AgentMemoryPanel />);
        const approve = await screen.findByRole('button', { name: /approve/i });
        fireEvent.click(approve);
        await waitFor(() => {
            expect(fetchMock).toHaveBeenCalledWith(
                expect.stringContaining('/projects/proj-1/memory/mem-2/approve'),
                expect.objectContaining({ method: 'POST' }),
            );
        });
    });

    it('posts a new note with the chosen kind and pin', async () => {
        const fetchMock = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
            if (init?.method === 'POST') return json({ ...entry, id: 'mem-3', quarantined: false });
            return json({ entries: [] });
        });
        vi.stubGlobal('fetch', fetchMock);
        render(<AgentMemoryPanel />);

        fireEvent.change(
            screen.getByPlaceholderText(/rooftops are always wet/i),
            { target: { value: 'Use the amber grade for act two.' } },
        );
        fireEvent.click(screen.getByRole('button', { name: /^save$/i }));

        await waitFor(() => {
            const call = fetchMock.mock.calls.find(([, init]) => init?.method === 'POST');
            expect(call).toBeTruthy();
            expect(JSON.parse(String(call![1]!.body))).toEqual({
                text: 'Use the amber grade for act two.',
                kind: 'semantic',
                pinned: false,
            });
        });
    });
});
