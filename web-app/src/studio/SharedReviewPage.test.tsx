import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen, waitFor } from '@testing-library/react';
import { SharedReviewPage } from './SharedReviewPage';

function json(data: unknown, status = 200): Response {
    return new Response(JSON.stringify(data), {
        status,
        headers: { 'Content-Type': 'application/json' },
    });
}

describe('SharedReviewPage', () => {
    afterEach(() => {
        cleanup();
        vi.restoreAllMocks();
    });

    it('renders a read-only shared cut with reviewer controls', async () => {
        vi.stubGlobal('fetch', vi.fn(async () => json({
            shared: true,
            project: { id: 'p1', name: 'Shared Cut', resolution_w: 768, resolution_h: 512, fps: 25 },
            shots: [{ id: 's1', sequence_index: 0, action: 'Rain street', video_url: null }],
            comments: [],
            approvals: [],
            summary: { comments: 0, open: 0, approved: 0, changes_requested: 0 },
        })));

        render(<SharedReviewPage token="capability-token" />);
        await waitFor(() => expect(screen.getByText('Shared Cut')).toBeTruthy());
        expect(screen.getByText(/1\. Rain street/)).toBeTruthy();
        expect(screen.getByText(/Approve/)).toBeTruthy();
        expect(screen.getByText(/No comments yet/)).toBeTruthy();
    });
});
