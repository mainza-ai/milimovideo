import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { ComfyWorkflowPanel } from './ComfyUIPanel';
import { useStudioJobs } from '../state/jobStore';
import { useStudioDocumentStore } from '../state/documentStore';

function json(data: unknown, status = 200): Response {
    return new Response(JSON.stringify(data), {
        status,
        headers: { 'Content-Type': 'application/json' },
    });
}

function bodyOf(init?: RequestInit): Record<string, unknown> {
    return JSON.parse(String(init?.body || '{}'));
}

const HEALTH = {
    reachable: true,
    url: 'http://127.0.0.1:8001',
    version: '0.45.0',
    queue_running: 0,
    queue_pending: 0,
    ram_free_gb: 40,
    ram_total_gb: 128,
};

function stubFetch(options: { reachable?: boolean } = {}) {
    const calls: Array<{ url: string; init?: RequestInit }> = [];
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
        calls.push({ url, init });
        if (url.includes('/comfy/health')) {
            return json({ ...HEALTH, reachable: options.reachable !== false });
        }
        if (url.includes('/comfy/validate')) {
            return json({
                ok: true, nodes: 2, output_nodes: ['2'], width: 640, height: 360, warnings: [],
            });
        }
        if (url.includes('/comfy/run')) {
            return json({
                job_id: 'job_comfy_test',
                status: 'queued',
                prompt_id: null,
                validation: { nodes: 2, output_nodes: ['2'], width: 640, height: 360, warnings: [] },
                comfy_url: HEALTH.url,
            });
        }
        return json({ detail: 'not found' }, 404);
    }));
    return calls;
}

describe('ComfyWorkflowPanel', () => {
    beforeEach(() => {
        window.localStorage.removeItem('milimo-comfy-workflow');
        useStudioJobs.setState({ activeJobs: [], history: [], metrics: null, lastUpdated: null, unread: 0 });
        useStudioDocumentStore.setState({ toasts: [], updateShot: vi.fn() });
    });

    afterEach(() => {
        cleanup();
        vi.unstubAllGlobals();
    });

    it('validates and queues the workflow against the selected shot', async () => {
        const calls = stubFetch();
        render(<ComfyWorkflowPanel projectId="proj-1" shotId="shot-1" />);

        fireEvent.click(screen.getByText('Validate'));
        await waitFor(() => expect(screen.getByText('2 nodes')).toBeTruthy());
        expect(calls.some((call) => call.url.includes('/comfy/validate'))).toBe(true);

        fireEvent.click(screen.getByText('Run workflow'));
        await waitFor(() => expect(useStudioJobs.getState().activeJobs).toHaveLength(1));

        const run = calls.find((call) => call.url.includes('/comfy/run'));
        const body = bodyOf(run?.init);
        expect(body.project_id).toBe('proj-1');
        expect(body.shot_id).toBe('shot-1');
        expect(body.workflow).toMatchObject({
            '1': { class_type: 'EmptyImage' },
            '2': { class_type: 'SaveImage' },
        });
        expect(useStudioJobs.getState().activeJobs[0]).toMatchObject({
            job_id: 'job_comfy_test', kind: 'comfy', status: 'queued',
        });
        expect(useStudioDocumentStore.getState().updateShot).toHaveBeenCalledWith(
            'shot-1',
            expect.objectContaining({ isGenerating: true, lastJobId: 'job_comfy_test' }),
        );
        expect(window.localStorage.getItem('milimo-comfy-workflow')).toContain('EmptyImage');
    });

    it('disables running when ComfyUI is offline', async () => {
        stubFetch({ reachable: false });
        render(<ComfyWorkflowPanel projectId="proj-1" shotId={null} />);

        await waitFor(() => expect(screen.getByText('offline')).toBeTruthy());
        const run = screen.getByText('Run workflow').closest('button');
        expect(run?.disabled).toBe(true);
    });

    it('reports invalid JSON without calling the backend', async () => {
        const calls = stubFetch();
        render(<ComfyWorkflowPanel projectId="proj-1" />);
        fireEvent.change(screen.getByLabelText('ComfyUI API workflow JSON'), {
            target: { value: '{not json' },
        });
        fireEvent.click(screen.getByText('Validate'));
        await waitFor(() => expect(
            useStudioDocumentStore.getState().toasts.some((toast) => /Invalid JSON/.test(toast.message)),
        ).toBe(true));
        expect(calls.some((call) => call.url.includes('/comfy/validate'))).toBe(false);
    });
});
