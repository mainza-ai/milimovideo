import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryPanel } from './MemoryPanel';
import { ActivityCenter } from './ActivityCenter';
import { useStudioJobs } from '../state/jobStore';
import { useStudioDocumentStore } from '../state/documentStore';
import { saveNotificationPreferences } from '../notifications/preferences';
import { handleStudioJobEvent } from '../events/jobBridge';
import { NotificationSettingsPanel } from './NotificationSettingsPanel';

function json(data: unknown, status = 200): Response {
    return new Response(JSON.stringify(data), {
        status,
        headers: { 'Content-Type': 'application/json' },
    });
}

const SNAPSHOT = {
    total_gb: 128,
    reserve_gb: 19.2,
    ceiling_gb: 115.4,
    in_process_gb: 0,
    resident_gb: 36,
    external_gb: 33,
    sam_gb: 0,
    used_gb: 69,
    available_gb: 46.4,
    swap_percent: 42,
    pressure: 'normal',
    enforce: true,
    external_engines: [
        {
            provider_id: 'strata',
            provider_type: 'openai_compatible',
            name: 'Strata',
            base_url: 'http://127.0.0.1:1234',
            resident_gb: 33,
            models: ['strata-70b'],
            can_unload: false,
            source: 'declared',
            unknown: false,
        },
    ],
    slots: {
        image: {
            slot: 'image',
            model_id: 'flux2',
            cost_gb: 36,
            leases: 1,
            keep_warm: false,
            last_used: 0,
            acquired_at: 0,
        },
    },
    active_jobs: [{ job_id: 'job-1', kind: 'image', started_at: 0 }],
    notes: [],
};

describe('MemoryPanel', () => {
    beforeEach(() => {
        useStudioDocumentStore.setState({ toasts: [] });
        vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
            const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
            if (url.includes('/system/memory/unload')) return json({ result: {}, memory: SNAPSHOT });
            return json(SNAPSHOT);
        }));
    });
    afterEach(() => {
        cleanup();
        vi.restoreAllMocks();
    });

    it('shows the measured budget, external engines, and leased slots', async () => {
        render(<MemoryPanel />);
        await waitFor(() => expect(screen.getByText(/69.0 GB of 115.4 GB/)).toBeTruthy());
        expect(screen.getByText(/Strata · 33.0 GB/)).toBeTruthy();
        expect(screen.getByText(/flux2 · 36.0 GB · 1 lease/)).toBeTruthy();
        // A leased slot must not be released from under the job.
        const slotRelease = screen.getByTitle('In use by a running job') as HTMLButtonElement;
        expect(slotRelease.disabled).toBe(true);
        // Strata cannot unload remotely; the control stays disabled.
        const engineRelease = screen.getByTitle('This engine cannot unload remotely') as HTMLButtonElement;
        expect(engineRelease.disabled).toBe(true);
    });

    it('releases everything and refreshes the snapshot', async () => {
        render(<MemoryPanel />);
        await waitFor(() => expect(screen.getByText('Release all')).toBeTruthy());
        fireEvent.click(screen.getByText('Release all'));
        await waitFor(() => expect(useStudioDocumentStore.getState().toasts.length).toBeGreaterThan(0));
        const calls = (fetch as unknown as { mock: { calls: unknown[][] } }).mock.calls;
        expect(calls.some((call) => String(call[0]).includes('/system/memory/unload'))).toBe(true);
    });
});

describe('notification preferences', () => {
    beforeEach(() => {
        useStudioDocumentStore.setState({ toasts: [] });
        saveNotificationPreferences({ toasts: true, os: false, sound: false });
    });
    afterEach(() => {
        cleanup();
        useStudioDocumentStore.setState({ toasts: [] });
        useStudioJobs.setState({ activeJobs: [], history: [], unread: 0 });
    });

    it('suppresses toasts but still records history when disabled', () => {
        saveNotificationPreferences({ toasts: false });
        handleStudioJobEvent({ event: 'job.completed', job_id: 'job-muted', progress: 100 });
        expect(useStudioDocumentStore.getState().toasts).toHaveLength(0);
        expect(useStudioJobs.getState().history).toHaveLength(1);
    });

    it('mutes notifications for one project but keeps history', () => {
        saveNotificationPreferences({ toasts: true, mutedProjects: ['proj-muted'] });
        handleStudioJobEvent({
            event: 'job.completed', job_id: 'job-muted-project', progress: 100,
            project_id: 'proj-muted',
        });
        expect(useStudioDocumentStore.getState().toasts).toHaveLength(0);
        expect(useStudioJobs.getState().history[0]?.job_id).toBe('job-muted-project');
        saveNotificationPreferences({ mutedProjects: [] });
    });

    it('renders the persisted preferences and keeps desktop off without permission', () => {
        render(<NotificationSettingsPanel />);
        const toasts = screen.getByLabelText('In-app toasts') as HTMLInputElement;
        const sound = screen.getByLabelText('Sound cue') as HTMLInputElement;
        const desktop = screen.getByLabelText('Desktop notifications') as HTMLInputElement;
        expect(toasts.checked).toBe(true);
        expect(sound.checked).toBe(false);
        expect(desktop.checked).toBe(false);
        expect(screen.getByText(/Desktop notifications fire only/)).toBeTruthy();
    });
});

describe('ActivityCenter', () => {
    beforeEach(() => {
        useStudioJobs.setState({
            activeJobs: [{
                job_id: 'job-9',
                type: 'video',
                kind: 'video',
                status: 'processing',
                stage: 'denoising',
                progress: 61,
                status_message: 'Denoising 12/20',
                eta_seconds: 90,
            }],
            history: [],
            unread: 0,
            metrics: null,
            lastUpdated: Date.now(),
        });
    });
    afterEach(() => {
        cleanup();
        vi.restoreAllMocks();
        useStudioJobs.setState({ activeJobs: [], history: [], unread: 0 });
    });

    it('retries a finished job from its shot', async () => {
        useStudioJobs.setState({
            activeJobs: [],
            history: [{
                job_id: 'job-done', type: 'video', kind: 'video', shot_id: 'shot-9',
                status: 'failed', stage: 'failed', progress: 0,
                status_message: 'Generation failed', finished_at: Date.now(),
            }],
            unread: 1,
        });
        const fetchMock = vi.fn(async (_input: RequestInfo | URL) => json({ job_id: 'job-new' }));
        vi.stubGlobal('fetch', fetchMock);
        render(<ActivityCenter />);
        fireEvent.click(screen.getByLabelText('Activity'));
        fireEvent.click(screen.getByLabelText('Retry job-done'));
        await waitFor(() => expect(fetchMock).toHaveBeenCalled());
        expect(String(fetchMock.mock.calls[0][0])).toContain('/shots/shot-9/generate');
        expect(useStudioDocumentStore.getState().toasts.at(-1)?.type).toBe('success');
    });

    it('shows the running stage, percent, and ETA in the popover', async () => {
        render(<ActivityCenter />);
        fireEvent.click(screen.getByLabelText('Activity: 1 running'));
        expect(screen.getByText('Generating')).toBeTruthy();
        expect(screen.getByText('61%')).toBeTruthy();
        expect(screen.getByText('2m left')).toBeTruthy();
        expect(screen.getByText('Denoising 12/20')).toBeTruthy();
    });
});
