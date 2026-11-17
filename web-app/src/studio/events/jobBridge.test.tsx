import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen } from '@testing-library/react';
import { handleStudioJobEvent, registerStudioJob, stageLabel } from './jobBridge';
import { useStudioJobs } from '../state/jobStore';
import { useStudioDocumentStore } from '../state/documentStore';
import { StudioToastLayer } from '../components/StudioToastLayer';

function reset(): void {
    useStudioJobs.setState({ activeJobs: [], history: [], metrics: null, lastUpdated: null, unread: 0 });
    useStudioDocumentStore.setState({ toasts: [] });
}

describe('jobBridge', () => {
    beforeEach(reset);
    afterEach(() => {
        cleanup();
        vi.restoreAllMocks();
    });

    it('tracks a running job and mirrors progress onto the shot', () => {
        const updateShot = vi.fn();
        useStudioDocumentStore.setState({ updateShot });
        registerStudioJob('job-1', { shotId: 'shot-1', kind: 'video' });

        handleStudioJobEvent({
            event: 'job.stage',
            job_id: 'job-1',
            stage: 'denoising',
            progress: 42,
            step: 4,
            total: 20,
            eta_seconds: 55,
            message: 'Denoising 4/20',
        });

        const state = useStudioJobs.getState();
        expect(state.activeJobs).toHaveLength(1);
        expect(state.activeJobs[0]).toMatchObject({
            job_id: 'job-1',
            stage: 'denoising',
            progress: 42,
            eta_seconds: 55,
        });
        expect(updateShot).toHaveBeenCalledWith(
            'shot-1',
            expect.objectContaining({
                isGenerating: true,
                progress: 42,
                stage: 'denoising',
                etaSeconds: 55,
            }),
        );
    });

    it('fires exactly one terminal toast and moves the job to history', () => {
        registerStudioJob('job-2', { shotId: null, kind: 'video' });

        handleStudioJobEvent({ event: 'job.completed', job_id: 'job-2', progress: 100, message: 'Done' });

        const state = useStudioJobs.getState();
        expect(state.activeJobs).toHaveLength(0);
        expect(state.history[0]).toMatchObject({ job_id: 'job-2', status: 'completed' });
        expect(state.unread).toBe(1);
        expect(useStudioDocumentStore.getState().toasts.at(-1)).toMatchObject({
            message: 'Generation complete',
            type: 'success',
        });
        expect(useStudioDocumentStore.getState().toasts).toHaveLength(1);
    });

    it('reports a failure with the backend reason', () => {
        registerStudioJob('job-3', { kind: 'video' });
        handleStudioJobEvent({
            event: 'job.failed',
            job_id: 'job-3',
            stage: 'admission',
            error: 'Needs ~101.0 GB but only 82.4 GB can be made available',
        });

        const toast = useStudioDocumentStore.getState().toasts.at(-1);
        expect(toast?.type).toBe('error');
        expect(toast?.message).toContain('101.0 GB');
    });

    it('labels stages for the inline progress bar', () => {
        expect(stageLabel('denoising')).toBe('Generating');
        expect(stageLabel('encoding')).toBe('Encoding');
        expect(stageLabel('custom_stage')).toBe('custom stage');
        expect(stageLabel(null)).toBe('');
    });

    it('routes render and download events to toasts', async () => {
        const { handleStudioSystemEvent } = await import('./jobBridge');
        handleStudioSystemEvent('render_completed', { job_id: 'r1' });
        handleStudioSystemEvent('download_error', { error: '404' });

        const toasts = useStudioDocumentStore.getState().toasts;
        expect(toasts.some((toast) => toast.type === 'success' && /Render/.test(toast.message))).toBe(true);
        expect(toasts.some((toast) => toast.type === 'error' && /404/.test(toast.message))).toBe(true);
    });
});

describe('StudioToastLayer', () => {
    beforeEach(reset);
    afterEach(cleanup);

    it('renders every pending toast with a dismiss control', () => {
        useStudioDocumentStore.setState({
            toasts: [
                { id: 'a', message: 'Shot queued', type: 'success' },
                { id: 'b', message: 'Generation failed', type: 'error' },
            ],
        });
        render(<StudioToastLayer />);
        expect(screen.getByTestId('studio-toasts')).toBeTruthy();
        expect(screen.getByText('Shot queued')).toBeTruthy();
        expect(screen.getByText('Generation failed')).toBeTruthy();
        expect(screen.getAllByLabelText('Dismiss notification')).toHaveLength(2);
    });
});
