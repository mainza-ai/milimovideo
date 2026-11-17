import type { Page } from '@playwright/test';

function json(data: unknown) {
    return { status: 200, contentType: 'application/json', body: JSON.stringify(data) };
}

export function llmSettings(revision = 0) {
    return {
        schema_version: 2,
        revision,
        active_provider: 'gemma',
        active_model: 'ltx-gemma3',
        air_gapped: true,
        temperature: 0.7,
        max_tokens: 400,
        cloud_enabled: false,
        budget: { daily_usd: 0, eval_gate: false },
        last_eval: null,
        spend: { date: '2026-11-17', total_usd: 0, entries: [] },
        role_catalog: [
            { role: 'prompt_video', group: 'prompt', label: 'Video prompt', description: 'Camera, action and audio rewrite' },
            { role: 'prompt_image', group: 'prompt', label: 'Image prompt', description: 'Still and preview prompts' },
            { role: 'script_parse', group: 'story', label: 'Script parser', description: 'Script to scenes and shots' },
            { role: 'agent_plan', group: 'agent', label: 'Agent planner', description: 'Command dock plans' },
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
            prompt_image: { role: 'prompt_image', provider: 'ollama', model: 'qwen3:8b', fallback: [] },
            script_parse: { role: 'script_parse', provider: 'ollama', model: 'qwen3:8b', fallback: [] },
            agent_plan: { role: 'agent_plan', provider: 'ollama', model: 'qwen3:8b', fallback: [] },
        },
    };
}

/** API stubs for shell-only tests; no Python backend required. */
export async function stubStudioApi(page: Page) {
    await page.route('**/health', (route) => route.fulfill(json({
        status: 'ok',
        device: 'mps',
        gpu: { busy: false, queued: 0 },
        slots: { active_slots: [] },
        disk_free_bytes: 1,
        auth_required: false,
    })));
    await page.route('**/projects', (route) => route.fulfill(json([
        { id: 'project-test', name: 'Audit Test', shots: [], scenes: [] },
    ])));
    await page.route('**/projects/*/board', (route) => route.fulfill(json({
        layout: { nodes: [], areas: [], links: [] },
        revision: 0,
    })));
    await page.route('**/projects/*/view', (route) => route.fulfill(json({ view: {} })));
    await page.route('**/projects/*/review', (route) => route.fulfill(json({
        comments: [],
        approvals: [],
        summary: { comments: 0, open: 0, approved: 0, changes_requested: 0 },
    })));
    await page.route('**/projects/*/elements', (route) => route.fulfill(json([])));
    await page.route('**/projects/*/takes', (route) => route.fulfill(json({ takes: [], count: 0 })));
    await page.route('**/projects/*/renders', (route) => route.fulfill(json([])));
    await page.route('**/models', (route) => route.fulfill(json({ models: [], count: 0 })));
    await page.route('**/models/pipelines', (route) => route.fulfill(json({})));
    await page.route('**/uploads', (route) => route.fulfill(json([])));
    let llmState = llmSettings();
    await page.route('**/settings/llm**', (route) => {
        const request = route.request();
        const url = request.url();
        const method = request.method();
        if (url.includes('/models')) {
            return route.fulfill(json({
                provider: 'ollama',
                models: [{ id: 'qwen3:8b', name: 'qwen3:8b', parameter_size: '8B', vision: false, chat: true, context_length: 131072 }],
            }));
        }
        if (url.includes('/discover')) {
            return route.fulfill(json({
                servers: [{
                    provider: 'ollama', name: 'Ollama', wire: 'ollama_native',
                    base_url: 'http://localhost:11434', port: 11434, models: ['qwen3:8b'],
                    instance_id: 'ollama', attached: false, needs_key: false,
                }],
                count: 1,
            }));
        }
        if (url.includes('/verify')) {
            return route.fulfill(json({
                ok: true, provider: 'ollama', wire: 'ollama_native', latency_ms: 9,
                models: ['qwen3:8b'], error: null, fault: null, needs_key: false,
            }));
        }
        if (url.includes('/warm')) {
            return route.fulfill(json({ ok: true, provider: 'ollama', model: 'qwen3:8b', load_ms: 1200 }));
        }
        if (url.includes('/eval/apply')) {
            return route.fulfill(json({ ...llmState, revision: llmState.revision + 1 }));
        }
        if (url.includes('/eval')) {
            const summary = {
                completed_at: '2026-11-17T00:00:00+00:00', revision: llmState.revision,
                pass_rate: 1.0, p50_ms: 120, cases: 3,
                roles: {
                    prompt_video: { provider: 'ollama', model: 'qwen3:8b', pass_rate: 1.0, p50_ms: 120, cases: 1 },
                    prompt_image: { provider: 'ollama', model: 'qwen3:8b', pass_rate: 1.0, p50_ms: 110, cases: 1 },
                    script_parse: { provider: 'ollama', model: 'qwen3:8b', pass_rate: 1.0, p50_ms: 130, cases: 1 },
                },
            };
            llmState = { ...llmState, last_eval: summary };
            return route.fulfill(json({ summary, results: [], skipped: [] }));
        }
        if (url.includes('/test')) {
            if (url.includes('/stream')) {
                const body = [
                    'event: start',
                    'data: {"provider":"ollama","model":"qwen3:8b","role":"prompt_video"}',
                    '',
                    'event: delta',
                    'data: {"text":"A slow dolly-in on a rain-soaked street."}',
                    '',
                    'event: done',
                    'data: {"provider":"ollama","model":"qwen3:8b","latency_ms":24}',
                    '',
                ].join('\n');
                return route.fulfill({
                    status: 200,
                    contentType: 'text/event-stream',
                    body,
                });
            }
            return route.fulfill(json({
                reply: 'A slow dolly-in on a rain-soaked street.',
                provider: 'ollama', model: 'qwen3:8b', role: 'prompt_video', latency_ms: 24,
            }));
        }
        if (url.includes('/attach') && method === 'POST') {
            const payload = request.postDataJSON() as { provider: string; base_url: string; name?: string; wire?: string; models?: string[] };
            const id = payload.provider === 'ollama' ? 'ollama' : `${payload.provider}@${payload.base_url.replace(/^https?:\/\//, '').replace(/\/v1$/, '')}`;
            llmState = {
                ...llmState,
                revision: llmState.revision + 1,
                providers: {
                    ...llmState.providers,
                    [id]: {
                        id,
                        type: payload.provider,
                        name: payload.name || id,
                        base_url: payload.base_url,
                        default_model: (payload.models && payload.models[0]) || '',
                        wire: payload.wire || 'openai_chat',
                        enabled: true,
                        is_local: true,
                        timeout_sec: 120,
                        keep_alive: '0',
                        api_key: null,
                        has_key: false,
                        secret_source: 'none',
                        key_status: 'none',
                        last_success_at: null,
                        last_fault: null,
                    },
                },
            };
            return route.fulfill(json(llmState));
        }
        if (url.includes('/profiles') && method === 'POST') {
            llmState = { ...llmState, revision: llmState.revision + 1, air_gapped: true };
            return route.fulfill(json(llmState));
        }
        if (method === 'PUT') {
            const payload = request.postDataJSON() as Record<string, unknown>;
            llmState = { ...llmState, ...payload, revision: llmState.revision + 1 } as typeof llmState;
            return route.fulfill(json(llmState));
        }
        return route.fulfill(json(llmState));
    });
}
