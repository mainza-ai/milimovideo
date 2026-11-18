import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { ComfyHealthPanel, ComfyWorkflowPanel } from './ComfyUIPanel';
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
    enabled: true,
    reachable: true,
    url: 'http://127.0.0.1:8001',
    version: '0.45.0',
    queue_running: 0,
    queue_pending: 0,
    ram_free_gb: 40,
    ram_total_gb: 128,
};

function stubFetch(options: { reachable?: boolean; enabled?: boolean } = {}) {
    const calls: Array<{ url: string; init?: RequestInit }> = [];
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
        calls.push({ url, init });
        if (url.includes('/comfy/health')) {
            return json({
                ...HEALTH,
                enabled: options.enabled !== false,
                reachable: options.enabled !== false && options.reachable !== false,
            });
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
        useStudioDocumentStore.setState({
            toasts: [],
            updateShot: vi.fn(),
            project: {
                id: 'proj-1',
                name: 'Test',
                fps: 25,
                seed: 42,
                resolutionW: 768,
                resolutionH: 512,
                shots: [{
                    id: 'shot-1', prompt: 'a prompt', negativePrompt: '', seed: 7,
                    width: 768, height: 512, numFrames: 25, numInferenceSteps: 8,
                    cfgScale: 2, pipelineOverride: 'auto', trackIndex: 0,
                    trimIn: 0, trimOut: 0, timeline: [], status: 'pending',
                }],
                scenes: [],
            },
        } as any);
    });

    afterEach(() => {
        cleanup();
        vi.unstubAllGlobals();
    });

    it('validates and queues the workflow against the selected shot', async () => {
        const calls = stubFetch();
        render(<ComfyWorkflowPanel projectId="proj-1" shotId="shot-1" />);

        await screen.findByText('Validate');
        fireEvent.click(screen.getByText('Validate'));
        await waitFor(() => expect(screen.getByText('2 nodes')).toBeTruthy());
        expect(calls.some((call) => call.url.includes('/comfy/validate'))).toBe(true);

        await waitFor(() => {
            const button = screen.getByText('Run').closest('button');
            expect(button?.disabled).toBe(false);
        });
        fireEvent.click(screen.getByText('Run'));
        await waitFor(() => expect(calls.some((call) => call.url.includes('/comfy/run'))).toBe(true), { timeout: 3000 });

        const run = calls.find((call) => call.url.includes('/comfy/run'));
        const body = bodyOf(run?.init);
        expect(body.project_id).toBe('proj-1');
        expect(body.shot_id).toBe('shot-1');
        expect(body.workflow).toMatchObject({
            '1': { class_type: 'EmptyImage' },
            '2': { class_type: 'SaveImage' },
        });
        expect(body.workflow_id).toBeUndefined();
        expect(useStudioDocumentStore.getState().updateShot).toHaveBeenCalledWith(
            'shot-1',
            expect.objectContaining({ isGenerating: true, lastJobId: 'job_comfy_test' }),
        );
        expect(window.localStorage.getItem('milimo-comfy-workflow:proj-1')).toContain('EmptyImage');
    });

    it('hides the Generate panel while ComfyUI is offline', async () => {
        stubFetch({ reachable: false });
        const { container } = render(<ComfyWorkflowPanel projectId="proj-1" shotId={null} />);

        await waitFor(() => expect(container.textContent).not.toContain('ComfyUI workflow'));
        expect(useStudioJobs.getState().activeJobs).toHaveLength(0);
    });

    it('disappears entirely when the bridge is disabled', async () => {
        stubFetch({ enabled: false });
        const { container } = render(<ComfyWorkflowPanel projectId="proj-1" />);

        await waitFor(() => expect(container.textContent).not.toContain('ComfyUI workflow'));
        expect(useStudioJobs.getState().activeJobs).toHaveLength(0);
    });

    it('reports invalid JSON without calling the backend', async () => {
        const calls = stubFetch();
        render(<ComfyWorkflowPanel projectId="proj-1" />);
        await screen.findByLabelText('ComfyUI API workflow JSON');
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


describe('ComfyHealthPanel', () => {
    beforeEach(() => {
        useStudioDocumentStore.setState({ toasts: [] });
    });

    afterEach(() => {
        cleanup();
        vi.unstubAllGlobals();
    });

    it('shows a disabled state without probing or freeing', async () => {
        const calls = stubFetch({ enabled: false });
        render(<ComfyHealthPanel />);

        await waitFor(() => expect(screen.getByText('disabled')).toBeTruthy());
        expect(screen.getByText(/optional and switched off/i)).toBeTruthy();
        expect(screen.queryByText('Free cached models')).toBeNull();
        expect(screen.queryByText('Probe')).toBeNull();
        expect(calls.some((call) => call.url.includes('/comfy/free'))).toBe(false);
    });
});
