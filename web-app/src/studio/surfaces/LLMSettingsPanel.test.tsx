import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { LLMSettingsPanel } from './LLMSettingsPanel';

function json(data: unknown, status = 200): Response {
    return new Response(JSON.stringify(data), {
        status,
        headers: { 'Content-Type': 'application/json' },
    });
}

function snapshot() {
    return {
        schema_version: 2,
        revision: 3,
        air_gapped: true,
        budget: { daily_usd: 0, eval_gate: false },
        last_eval: null,
        spend: { date: '2026-11-17', total_usd: 0, entries: [] },
        active_provider: 'gemma',
        active_model: 'ltx-gemma3',
        temperature: 0.7,
        max_tokens: 400,
        cloud_enabled: false,
        role_catalog: [
            { role: 'prompt_video', group: 'prompt', label: 'Video prompt', description: 'Camera and action rewrite' },
            { role: 'script_parse', group: 'story', label: 'Script parser', description: 'Script to scenes' },
        ],
        providers: {
            gemma: {
                id: 'gemma', type: 'gemma', name: 'Gemma 3 (LTX text encoder)', base_url: '', default_model: 'ltx-gemma3',
                wire: 'gemma_native', enabled: true, is_local: true, timeout_sec: 120, keep_alive: '0',
                api_key: null, has_key: false, secret_source: 'none', key_status: 'none',
                last_success_at: null, last_fault: null,
            },
            ollama: {
                id: 'ollama', type: 'ollama', name: 'Ollama', base_url: 'http://127.0.0.1:11434', default_model: 'qwen3:8b',
                wire: 'ollama_native', enabled: true, is_local: true, timeout_sec: 120, keep_alive: '0',
                api_key: null, has_key: false, secret_source: 'none', key_status: 'none',
                last_success_at: null, last_fault: null,
            },
            openai: {
                id: 'openai', type: 'openai', name: 'OpenAI', base_url: 'https://api.openai.com/v1', default_model: '',
                wire: 'openai_chat', enabled: false, is_local: false, timeout_sec: 120, keep_alive: '0',
                api_key: null, has_key: false, secret_source: 'none', key_status: 'missing',
                last_success_at: null, last_fault: null,
            },
        },
        roles: {
            prompt_video: { role: 'prompt_video', provider: 'gemma', model: 'ltx-gemma3', fallback: [] },
            script_parse: { role: 'script_parse', provider: 'ollama', model: 'qwen3:8b', fallback: [] },
        },
    };
}

interface Call {
    url: string;
    init?: RequestInit;
}

function stubFetch(overrides: { put?: () => Response } = {}) {
    const calls: Call[] = [];
    let state = snapshot() as ReturnType<typeof snapshot>;
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
        calls.push({ url, init });
        if (url.includes('/settings/llm/models')) {
            return json({ provider: 'ollama', models: [{ id: 'qwen3:8b', name: 'qwen3:8b' }] });
        }
        if (url.includes('/settings/llm/providers/')) {
            return json({ ok: true, provider: 'ollama', wire: 'ollama_native', latency_ms: 12, models: ['qwen3:8b'] });
        }
        if (url.includes('/settings/llm/attach')) {
            const body = JSON.parse(String(init?.body || '{}'));
            state = {
                ...state,
                revision: state.revision + 1,
                providers: {
                    ...state.providers,
                    ollama: { ...state.providers.ollama, base_url: body.base_url },
                },
            };
            return json(state);
        }
        if (url.includes('/settings/llm/discover')) {
            return json({
                servers: [{
                    provider: 'ollama', name: 'Ollama', wire: 'ollama_native',
                    base_url: 'http://localhost:11434', port: 11434, models: ['qwen3:8b'],
                }],
                count: 1,
            });
        }
        if (url.includes('/settings/llm/test')) {
            return json({ reply: 'A rainy street.', provider: 'ollama', model: 'qwen3:8b', role: 'prompt_video', latency_ms: 30 });
        }
        if (init?.method === 'PUT') {
            if (overrides.put) return overrides.put();
            state = { ...state, revision: 4 };
            return json(state);
        }
        return json(state);
    }));
    return calls;
}

describe('LLMSettingsPanel', () => {
    beforeEach(() => {
        stubFetch();
    });

    afterEach(() => {
        cleanup();
        vi.restoreAllMocks();
    });

    it('renders providers, roles, and the air-gap state', async () => {
        render(<LLMSettingsPanel />);
        await waitFor(() => expect(screen.getByText('AI & LLM')).toBeTruthy());
        expect(screen.getAllByText('Ollama').length).toBeGreaterThan(0);
        expect(screen.getByLabelText('Video prompt provider')).toBeTruthy();
        expect(screen.getByText('air-gapped')).toBeTruthy();
        expect(screen.getByText('revision 3')).toBeTruthy();
    });

    it('saves the full document with the revision the client read', async () => {
        const calls = stubFetch();
        render(<LLMSettingsPanel />);
        await waitFor(() => expect(screen.getByLabelText('Ollama default model')).toBeTruthy());

        fireEvent.change(screen.getByLabelText('Ollama default model'), {
            target: { value: 'gemma3:12b' },
        });
        await waitFor(() => expect(screen.getByText('unsaved changes')).toBeTruthy());

        fireEvent.click(screen.getByRole('button', { name: 'Save' }));
        await waitFor(() => expect(calls.some((call) => call.init?.method === 'PUT')).toBe(true));

        const put = calls.find((call) => call.init?.method === 'PUT')!;
        expect(put.url).toContain('expected_revision=3');
        const body = JSON.parse(String(put.init?.body));
        expect(body.providers.ollama.default_model).toBe('gemma3:12b');
        expect(body.roles.prompt_video.provider).toBe('gemma');
    });

    it('surfaces a revision conflict instead of overwriting', async () => {
        stubFetch({
            put: () => json({
                detail: {
                    message: 'settings changed since revision 3 (current 5): re-read and re-apply',
                    expected_revision: 3,
                    revision: 5,
                    settings: { ...snapshot(), revision: 5 },
                },
            }, 409),
        });
        render(<LLMSettingsPanel />);
        await waitFor(() => expect(screen.getByLabelText('Active model')).toBeTruthy());
        fireEvent.change(screen.getByLabelText('Active model'), { target: { value: 'changed' } });
        fireEvent.click(screen.getByRole('button', { name: 'Save' }));
        await waitFor(() => expect(screen.getByText('Revision conflict')).toBeTruthy());
        expect(screen.getByRole('button', { name: 'Load server settings' })).toBeTruthy();
    });

    it('discovers a local server and attaches it to the provider card', async () => {
        render(<LLMSettingsPanel />);
        await waitFor(() => expect(screen.getByRole('button', { name: /Scan local servers/ })).toBeTruthy());
        fireEvent.click(screen.getByRole('button', { name: /Scan local servers/ }));
        await waitFor(() => expect(screen.getByRole('button', { name: 'Attach' })).toBeTruthy());
        fireEvent.click(screen.getByRole('button', { name: 'Attach' }));
        expect(await screen.findByLabelText('Ollama base URL')).toHaveProperty('value', 'http://localhost:11434');
    });

    it('verifies a provider and reports latency', async () => {
        render(<LLMSettingsPanel />);
        await waitFor(() => expect(screen.getAllByRole('button', { name: /Verify/ }).length).toBeGreaterThan(0));
        fireEvent.click(screen.getAllByRole('button', { name: /Verify/ })[1]);
        await waitFor(() => expect(screen.getByText(/ok · 12 ms/)).toBeTruthy());
    });
});
