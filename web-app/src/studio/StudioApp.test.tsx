import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen, waitFor } from '@testing-library/react';
import { StudioApp } from './StudioApp';

function json(data: unknown, status = 200): Response {
    return new Response(JSON.stringify(data), {
        status,
        headers: { 'Content-Type': 'application/json' },
    });
}

describe('StudioApp shell', () => {
    beforeEach(() => {
        vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
            const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
            if (url.includes('/health')) {
                return json({
                    status: 'ok',
                    device: 'mps',
                    gpu: { busy: false, queued: 0 },
                    slots: { active_slots: [] },
                    disk_free_bytes: 1,
                    auth_required: false,
                });
            }
            return json([]);
        }));
    });

    afterEach(() => {
        cleanup();
        vi.restoreAllMocks();
    });

    it('renders the story spine and shell controls', async () => {
        render(<StudioApp />);
        await waitFor(() => expect(screen.getByText('Story spine')).toBeTruthy());
        expect(screen.getByText('Milimo')).toBeTruthy();
        expect(screen.getAllByText('Generate').length).toBeGreaterThan(0);
        expect(screen.getAllByText('Deliver').length).toBeGreaterThan(0);
    });
});
