import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { StudioApp } from './StudioApp';
import { useStudioEditor } from './state/editorStore';

function json(data: unknown, status = 200): Response {
    return new Response(JSON.stringify(data), {
        status,
        headers: { 'Content-Type': 'application/json' },
    });
}

describe('StudioApp shell', () => {
    beforeEach(() => {
        useStudioEditor.setState({
            activeSurface: 'story',
            commandOpen: true,
            commandDraft: '',
        });
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
            if (url.includes('/settings/llm')) {
                return json({
                    schema_version: 1,
                    revision: 0,
                    active_provider: 'gemma',
                    active_model: 'ltx-gemma3',
                    air_gapped: true,
                    temperature: 0.7,
                    max_tokens: 400,
                    cloud_enabled: false,
                    role_catalog: [],
                    providers: {
                        gemma: {
                            type: 'gemma', name: 'Gemma 3 (LTX text encoder)', base_url: '',
                            default_model: 'ltx-gemma3', wire: 'gemma_native', enabled: true,
                            is_local: true, timeout_sec: 120, keep_alive: '0', api_key: null,
                            has_key: false, secret_source: 'none', key_status: 'none',
                            last_success_at: null, last_fault: null,
                        },
                    },
                    roles: {
                        prompt_video: { role: 'prompt_video', provider: 'gemma', model: 'ltx-gemma3', fallback: [] },
                    },
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

    it('opens the review surface with frame comments', async () => {
        render(<StudioApp />);
        await waitFor(() => expect(screen.getByText('Story spine')).toBeTruthy());
        fireEvent.click(screen.getAllByText('Review')[0]);
        await waitFor(() => expect(screen.getByText(/Comment at frame/)).toBeTruthy());
        expect(screen.getByText(/Approve shot/)).toBeTruthy();
    });

    it('opens setup with real device and model status', async () => {
        render(<StudioApp />);
        fireEvent.click(screen.getByText('Setup'));
        await waitFor(() => expect(screen.getByText('Memory slots')).toBeTruthy());
        expect(screen.getByText('Models')).toBeTruthy();
    });
});
