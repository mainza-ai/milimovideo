import { useCallback, useEffect, useMemo, useState } from 'react';
import {
    AlertCircle, CheckCircle2, Cloud, Cpu, Eye, Loader2, Play, Plus, Radar, RefreshCw,
    RotateCcw, Save, ShieldAlert, ShieldCheck, Sparkles, Thermometer, Trash2, TriangleAlert,
} from 'lucide-react';
import { apiUrl } from '../../config';
import { useStudioDocumentStore } from '../state/documentStore';

interface ProviderConfig {
    id: string;
    type: string;
    name: string;
    base_url: string;
    default_model: string;
    wire: string;
    enabled: boolean;
    is_local: boolean;
    timeout_sec: number;
    keep_alive: string;
    api_key?: string | null;
    clear_key?: boolean;
    has_key: boolean;
    secret_source: string;
    key_status: string;
    last_success_at?: string | null;
    last_fault?: string | null;
    telemetry?: {
        last_load_ms?: number;
        last_latency_ms?: number;
        last_tokens_per_sec?: number;
        calls?: number;
        failures?: number;
        last_usage?: Record<string, number>;
    };
}

interface ModelTarget {
    provider: string;
    model: string;
}

interface RoleAssignment {
    role: string;
    provider: string;
    model: string;
    fallback: ModelTarget[];
    temperature?: number | null;
    top_p?: number | null;
    seed?: number | null;
    max_tokens?: number | null;
}

interface RoleMeta {
    role: string;
    group: string;
    label: string;
    description: string;
}

interface ModelRow {
    id: string;
    name: string;
    size?: number | null;
    parameter_size?: string | null;
    context_length?: number | null;
    vision?: boolean;
    chat?: boolean;
    embedding_only?: boolean;
    quantization?: string | null;
}

interface EvalRoleRow {
    provider?: string | null;
    model?: string | null;
    cases?: number;
    passed?: number;
    pass_rate?: number;
    p50_ms?: number;
}

interface BudgetConfig {
    daily_usd: number;
    eval_gate: boolean;
}

interface LLMSettings {
    schema_version: number;
    revision: number;
    active_provider: string;
    active_model: string;
    air_gapped: boolean;
    temperature: number;
    max_tokens: number;
    providers: Record<string, ProviderConfig>;
    roles: Record<string, RoleAssignment>;
    role_catalog: RoleMeta[];
    cloud_enabled: boolean;
    budget: BudgetConfig;
    last_eval?: {
        completed_at: string;
        revision: number;
        pass_rate: number;
        p50_ms: number;
        cases: number;
        roles: Record<string, EvalRoleRow>;
    } | null;
    spend?: { date: string; total_usd: number; entries: { model?: string; usd?: number }[] };
    warnings?: { role: string; provider: string; model: string }[];
}

interface VerifyResult {
    ok: boolean;
    provider: string;
    latency_ms: number;
    models: string[];
    error?: string | null;
    fault?: string | null;
    needs_key?: boolean;
}

interface DiscoveredServer {
    provider: string;
    name: string;
    wire: string;
    base_url: string;
    port: number;
    models: string[];
    instance_id?: string;
    attached?: boolean;
    needs_key?: boolean;
}

const ROLE_GROUPS: Record<string, string> = {
    prompt: 'Prompt rewriting',
    story: 'Story & script',
    agent: 'Agent & review',
    review: 'Review & continuity',
};

const PROFILE_LABELS: Record<string, string> = {
    sovereign: 'Sovereign local',
    local_preferred: 'Local preferred',
    hybrid: 'Hybrid (opt-in)',
};

function payloadFor(settings: LLMSettings): Record<string, unknown> {
    const providers: Record<string, unknown> = {};
    for (const [id, provider] of Object.entries(settings.providers)) {
        providers[id] = {
            id,
            type: provider.type,
            name: provider.name,
            base_url: provider.base_url,
            default_model: provider.default_model,
            wire: provider.wire,
            enabled: provider.enabled,
            is_local: provider.is_local,
            timeout_sec: provider.timeout_sec,
            keep_alive: provider.keep_alive,
            api_key: provider.api_key || null,
            clear_key: Boolean(provider.clear_key),
        };
    }
    const roles: Record<string, unknown> = {};
    for (const [id, role] of Object.entries(settings.roles)) {
        roles[id] = {
            role: role.role,
            provider: role.provider,
            model: role.model,
            fallback: role.fallback,
            temperature: role.temperature ?? null,
            top_p: role.top_p ?? null,
            seed: role.seed ?? null,
            max_tokens: role.max_tokens ?? null,
        };
    }
    return {
        schema_version: settings.schema_version,
        active_provider: settings.active_provider,
        active_model: settings.active_model,
        air_gapped: settings.air_gapped,
        temperature: settings.temperature,
        max_tokens: settings.max_tokens,
        providers,
        roles,
        budget: settings.budget,
        last_eval: settings.last_eval || null,
    };
}

function formatContext(value?: number | null): string {
    if (!value || value <= 0) return 'unknown';
    if (value >= 1000) return `${Math.round(value / 1000)}k`;
    return String(value);
}

function formatBytes(bytes?: number | null): string {
    if (!bytes || bytes <= 0) return '';
    const gb = bytes / 1e9;
    return gb >= 1 ? `${gb.toFixed(1)} GB` : `${Math.round(bytes / 1e6)} MB`;
}

/** Parse one SSE frame from the test stream. */
function parseFrame(chunk: string): { event: string; data: Record<string, unknown> } | null {
    let event = 'message';
    const dataLines: string[] = [];
    for (const line of chunk.split('\n')) {
        if (line.startsWith('event:')) event = line.slice(6).trim();
        else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim());
    }
    if (!dataLines.length) return null;
    try {
        return { event, data: JSON.parse(dataLines.join('\n')) };
    } catch {
        return null;
    }
}

export function LLMSettingsPanel() {
    const addToast = useStudioDocumentStore((state) => state.addToast);
    const [settings, setSettings] = useState<LLMSettings | null>(null);
    const [draft, setDraft] = useState<LLMSettings | null>(null);
    const [loading, setLoading] = useState(true);
    const [saving, setSaving] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [warnings, setWarnings] = useState<LLMSettings['warnings']>([]);
    const [conflict, setConflict] = useState(false);
    const [models, setModels] = useState<Record<string, ModelRow[]>>({});
    const [catalogProvider, setCatalogProvider] = useState<string | null>(null);
    const [busyProvider, setBusyProvider] = useState<string | null>(null);
    const [verified, setVerified] = useState<Record<string, VerifyResult>>({});
    const [discovered, setDiscovered] = useState<DiscoveredServer[] | null>(null);
    const [scanning, setScanning] = useState(false);
    const [newName, setNewName] = useState('');
    const [newUrl, setNewUrl] = useState('');
    const [testRole, setTestRole] = useState('prompt_video');
    const [testPrompt, setTestPrompt] = useState('Describe a slow dolly-in on a rain-soaked street at night.');
    const [testing, setTesting] = useState(false);
    const [testReply, setTestReply] = useState('');
    const [testMeta, setTestMeta] = useState<{ provider?: string; model?: string; latency_ms?: number; role?: string } | null>(null);
    const [testError, setTestError] = useState<string | null>(null);
    const [evalRunning, setEvalRunning] = useState(false);
    const [evalResult, setEvalResult] = useState<{ summary?: LLMSettings['last_eval']; results?: unknown[]; skipped?: unknown[] } | null>(null);
    const [profileBusy, setProfileBusy] = useState<string | null>(null);
    const [missing, setMissing] = useState<{ role: string; provider: string; model: string }[]>([]);
    const [checking, setChecking] = useState(false);

    const load = useCallback(async () => {
        setLoading(true);
        try {
            const response = await fetch(apiUrl('/settings/llm'));
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const data = await response.json() as LLMSettings;
            if (
                !data || typeof data !== 'object'
                || !data.providers || typeof data.providers !== 'object'
                || !data.roles || typeof data.roles !== 'object'
            ) {
                throw new Error('LLM settings response was not a settings document');
            }
            setSettings(data);
            setDraft({ ...data, budget: data.budget || { daily_usd: 0, eval_gate: false } });
            setWarnings([]);
            setError(null);
            setConflict(false);
        } catch (exc) {
            setError(exc instanceof Error ? exc.message : 'Could not load LLM settings');
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => { void load(); }, [load]);

    const dirty = useMemo(
        () => Boolean(settings && draft && JSON.stringify(payloadFor(settings)) !== JSON.stringify(payloadFor(draft))),
        [settings, draft],
    );

    const roleCatalog = useMemo<RoleMeta[]>(() => (
        settings?.role_catalog?.length
            ? settings.role_catalog
            : Object.keys(settings?.roles || {}).map((role) => ({
                role, group: 'prompt', label: role, description: '',
            }))
    ), [settings]);

    const providerIds = draft ? Object.keys(draft.providers) : [];
    const groupedRoles = roleCatalog.reduce<Record<string, RoleMeta[]>>((groups, meta) => {
        const key = meta.group || 'prompt';
        groups[key] = [...(groups[key] || []), meta];
        return groups;
    }, {});
    const evalStale = Boolean(
        draft?.last_eval && settings && draft.last_eval.revision !== settings.revision,
    );

    const patchProvider = (id: string, patch: Partial<ProviderConfig>) => {
        setDraft((current) => current && ({
            ...current,
            providers: { ...current.providers, [id]: { ...current.providers[id], ...patch } },
        }));
    };

    const patchRole = (id: string, patch: Partial<RoleAssignment>) => {
        setDraft((current) => current && ({
            ...current,
            roles: { ...current.roles, [id]: { ...current.roles[id], ...patch } },
        }));
    };

    const patchSettings = (patch: Partial<LLMSettings>) => {
        setDraft((current) => current && ({ ...current, ...patch }));
    };

    const save = async (doc: LLMSettings | null = draft) => {
        if (!doc || !settings) return;
        setSaving(true);
        setError(null);
        setWarnings([]);
        try {
            const response = await fetch(
                apiUrl(`/settings/llm?expected_revision=${settings.revision}`),
                {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payloadFor(doc)),
                },
            );
            const data = await response.json().catch(() => ({}));
            if (response.status === 409) {
                setConflict(true);
                setError(data?.detail?.message || 'Settings changed on the server. Reload before saving.');
                return;
            }
            if (!response.ok) {
                const detail = typeof data?.detail === 'string' ? data.detail : 'Save rejected';
                throw new Error(detail);
            }
            const next = data as LLMSettings;
            setSettings(next);
            setDraft({ ...next, budget: next.budget || { daily_usd: 0, eval_gate: false } });
            setWarnings(next.warnings || []);
            setConflict(false);
            addToast(`LLM settings saved (revision ${next.revision})`, 'success');
        } catch (exc) {
            setError(exc instanceof Error ? exc.message : 'Save failed');
        } finally {
            setSaving(false);
        }
    };

    const loadModels = async (providerId: string) => {
        setBusyProvider(providerId);
        try {
            const response = await fetch(apiUrl(`/settings/llm/models?provider=${encodeURIComponent(providerId)}&deep=true`));
            const data = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Model list failed');
            setModels((current) => ({ ...current, [providerId]: data.models || [] }));
            setCatalogProvider(providerId);
        } catch (exc) {
            addToast(exc instanceof Error ? exc.message : 'Model list failed', 'error');
        } finally {
            setBusyProvider(null);
        }
    };

    const verify = async (providerId: string) => {
        setBusyProvider(providerId);
        try {
            const response = await fetch(apiUrl(`/settings/llm/providers/${encodeURIComponent(providerId)}/verify`), {
                method: 'POST',
            });
            const data = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Verify failed');
            setVerified((current) => ({ ...current, [providerId]: data as VerifyResult }));
            const result = data as VerifyResult;
            addToast(
                result.ok ? `${providerId} reachable (${result.latency_ms} ms, ${result.models.length} model(s))` : `${providerId}: ${result.error}`,
                result.ok ? 'success' : 'error',
            );
        } catch (exc) {
            addToast(exc instanceof Error ? exc.message : 'Verify failed', 'error');
        } finally {
            setBusyProvider(null);
        }
    };

    const warm = async (providerId: string) => {
        setBusyProvider(providerId);
        try {
            const response = await fetch(apiUrl(`/settings/llm/providers/${encodeURIComponent(providerId)}/warm`), {
                method: 'POST',
            });
            const data = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Warm failed');
            addToast(`${providerId} loaded in ${data.load_ms} ms`, 'success');
            await load();
        } catch (exc) {
            addToast(exc instanceof Error ? exc.message : 'Warm failed', 'error');
        } finally {
            setBusyProvider(null);
        }
    };

    const removeProvider = async (providerId: string) => {
        setBusyProvider(providerId);
        try {
            const response = await fetch(apiUrl(`/settings/llm/providers/${encodeURIComponent(providerId)}`), {
                method: 'DELETE',
            });
            const data = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Remove failed');
            await load();
            addToast(`Removed ${providerId}`, 'info');
        } catch (exc) {
            addToast(exc instanceof Error ? exc.message : 'Remove failed', 'error');
        } finally {
            setBusyProvider(null);
        }
    };

    const scan = async () => {
        setScanning(true);
        try {
            const response = await fetch(apiUrl('/settings/llm/discover'), { method: 'POST' });
            const data = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Scan failed');
            setDiscovered(Array.isArray(data.servers) ? data.servers : []);
            if (!data.servers?.length) addToast('No local LLM servers found on the usual ports', 'info');
        } catch (exc) {
            addToast(exc instanceof Error ? exc.message : 'Scan failed', 'error');
        } finally {
            setScanning(false);
        }
    };

    const attach = async (server: DiscoveredServer) => {
        setBusyProvider(server.instance_id || server.base_url);
        try {
            const response = await fetch(apiUrl('/settings/llm/attach'), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    provider: server.provider,
                    base_url: server.base_url,
                    name: server.name,
                    wire: server.wire,
                    models: server.models,
                }),
            });
            const data = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Attach failed');
            await load();
            addToast(`Attached ${server.name}`, 'info');
        } catch (exc) {
            addToast(exc instanceof Error ? exc.message : 'Attach failed', 'error');
        } finally {
            setBusyProvider(null);
        }
    };

    const addEndpoint = async () => {
        if (!newUrl.trim()) return;
        const url = newUrl.trim().replace(/\/$/, '');
        await attach({
            provider: 'openai_compatible',
            name: newName.trim() || url,
            wire: 'openai_chat',
            base_url: url.endsWith('/v1') ? url : `${url}/v1`,
            port: 0,
            models: [],
        });
        setNewName('');
        setNewUrl('');
    };

    const checkPlan = async () => {
        setChecking(true);
        try {
            const response = await fetch(apiUrl('/settings/llm/health?check=true'));
            const data = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Check failed');
            setMissing(Array.isArray(data.missing_models) ? data.missing_models : []);
            const count = (data.missing_models || []).length;
            addToast(count ? `${count} role model(s) missing from a live catalog` : 'Every role model is in a live catalog', count ? 'error' : 'success');
        } catch (exc) {
            addToast(exc instanceof Error ? exc.message : 'Check failed', 'error');
        } finally {
            setChecking(false);
        }
    };

    const runTest = async () => {
        setTesting(true);
        setTestReply('');
        setTestError(null);
        setTestMeta(null);
        try {
            const response = await fetch(apiUrl('/settings/llm/test/stream'), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ role: testRole, prompt: testPrompt }),
            });
            if (!response.ok) {
                const data = await response.json().catch(() => ({}));
                throw new Error(typeof data.detail === 'string' ? data.detail : `HTTP ${response.status}`);
            }
            const reader = response.body?.getReader();
            if (!reader) throw new Error('Streaming response unavailable');
            const decoder = new TextDecoder();
            let buffer = '';
            let text = '';
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });
                const frames = buffer.split('\n\n');
                buffer = frames.pop() || '';
                for (const frame of frames) {
                    const parsed = parseFrame(frame);
                    if (!parsed) continue;
                    if (parsed.event === 'start') {
                        setTestMeta(parsed.data as { provider?: string; model?: string; role?: string });
                    } else if (parsed.event === 'delta') {
                        text += String(parsed.data.text || '');
                        setTestReply(text);
                    } else if (parsed.event === 'error') {
                        setTestError(String(parsed.data.message || 'Test failed'));
                    } else if (parsed.event === 'done') {
                        setTestMeta((current) => ({ ...current, ...(parsed.data as object) }));
                    }
                }
            }
            if (!text) setTestError((current) => current || 'No output returned');
        } catch (exc) {
            setTestError(exc instanceof Error ? exc.message : 'Test failed');
        } finally {
            setTesting(false);
        }
    };

    const runEval = async () => {
        setEvalRunning(true);
        setEvalResult(null);
        try {
            const response = await fetch(apiUrl('/settings/llm/eval'), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ roles: ['prompt_video', 'prompt_image', 'script_parse'], limit: 2 }),
            });
            const data = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Eval failed');
            setEvalResult(data);
            await load();
            addToast(`Eval pass rate ${(Number(data.summary?.pass_rate || 0) * 100).toFixed(0)}%`, 'info');
        } catch (exc) {
            addToast(exc instanceof Error ? exc.message : 'Eval failed', 'error');
        } finally {
            setEvalRunning(false);
        }
    };

    const applyRecommendation = async () => {
        if (!settings) return;
        setProfileBusy('eval');
        try {
            const response = await fetch(apiUrl(`/settings/llm/eval/apply?expected_revision=${settings.revision}`), {
                method: 'POST',
            });
            const data = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Apply failed');
            await load();
            addToast('Applied the recommended local plan', 'success');
        } catch (exc) {
            addToast(exc instanceof Error ? exc.message : 'Apply failed', 'error');
        } finally {
            setProfileBusy(null);
        }
    };

    const applyProfile = async (profile: string) => {
        if (!settings) return;
        setProfileBusy(profile);
        try {
            const response = await fetch(
                apiUrl(`/settings/llm/profiles/${profile}?expected_revision=${settings.revision}`),
                { method: 'POST' },
            );
            const data = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Profile failed');
            await load();
            addToast(`${PROFILE_LABELS[profile] || profile} applied`, 'success');
        } catch (exc) {
            addToast(exc instanceof Error ? exc.message : 'Profile failed', 'error');
        } finally {
            setProfileBusy(null);
        }
    };

    const catalogRows = catalogProvider ? models[catalogProvider] || [] : [];

    return (
        <div className="st-panel" style={{ marginBottom: 14 }} aria-labelledby="llm-panel-title" data-testid="llm-panel">
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 10, flexWrap: 'wrap' }}>
                <div>
                    <h3 className="st-panel-title" id="llm-panel-title" style={{ margin: 0 }}>
                        <Cpu size={14} style={{ verticalAlign: -2, marginRight: 6 }} />
                        AI &amp; LLM
                    </h3>
                    <p className="st-value" style={{ margin: '4px 0 0', color: 'var(--st-muted)', fontSize: 11 }}>
                        Local-first. Each engine is its own instance, roles route independently, and cloud
                        stays unreachable unless <span className="st-mono">MILIMO_LLM_CLOUD=1</span>, a budget,
                        and the air-gap switch allow it.
                    </p>
                </div>
                <div className="st-button-row">
                    {draft && (
                        <span className="st-chip" data-tone={dirty ? 'warn' : 'good'}>
                            {dirty ? 'unsaved changes' : 'saved'}
                        </span>
                    )}
                    {draft && <span className="st-chip">revision {draft.revision}</span>}
                    <span className="st-status-pill" data-testid="llm-air-gap" data-tone={draft?.air_gapped === false ? 'warn' : 'good'}>
                        {draft?.air_gapped === false ? <ShieldAlert size={11} /> : <ShieldCheck size={11} />}
                        {draft?.air_gapped === false ? 'air-gap off' : 'air-gapped'}
                    </span>
                    <button type="button" className="st-button" data-size="sm" onClick={() => void load()} disabled={loading}>
                        <RefreshCw size={12} className={loading ? 'animate-spin' : ''} /> Reload
                    </button>
                    <button type="button" className="st-button" data-size="sm" onClick={() => void checkPlan()} disabled={checking}>
                        {checking ? <Loader2 size={12} className="animate-spin" /> : <ShieldCheck size={12} />} Check plan
                    </button>
                    <button
                        type="button"
                        className="st-button"
                        data-size="sm"
                        data-variant="primary"
                        disabled={!dirty || saving || !draft}
                        onClick={() => void save()}
                    >
                        {saving ? <Loader2 size={12} className="animate-spin" /> : <Save size={12} />}
                        {saving ? 'Saving…' : 'Save'}
                    </button>
                    <button
                        type="button"
                        className="st-button"
                        data-size="sm"
                        data-variant="ghost"
                        disabled={!dirty || saving}
                        onClick={() => { setDraft(settings); setConflict(false); setError(null); }}
                    >
                        <RotateCcw size={12} /> Revert
                    </button>
                </div>
            </div>

            {error && (
                <div
                    role="alert"
                    className="st-panel"
                    style={{ marginTop: 10, borderColor: 'rgba(255,93,108,.35)', background: 'rgba(255,93,108,.06)' }}
                >
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: 'var(--st-danger)' }}>
                        <AlertCircle size={13} /> <strong>{conflict ? 'Revision conflict' : 'LLM settings error'}</strong>
                    </div>
                    <div className="st-value" style={{ marginTop: 4, fontSize: 12 }}>{error}</div>
                    {conflict && (
                        <div className="st-button-row" style={{ marginTop: 8 }}>
                            <button type="button" className="st-button" data-size="sm" onClick={() => void load()}>
                                Load server settings
                            </button>
                        </div>
                    )}
                </div>
            )}

            {missing.length > 0 && (
                <div className="st-button-row" style={{ marginTop: 8 }} role="alert">
                    <span className="st-chip" data-tone="bad">
                        <TriangleAlert size={10} /> {missing.length} role model(s) missing from a live catalog
                    </span>
                    {missing.slice(0, 4).map((row) => (
                        <span className="st-chip" key={`${row.role}-${row.model}`}>
                            {row.role}: {row.model}
                        </span>
                    ))}
                </div>
            )}

            {warnings && warnings.length > 0 && (
                <div className="st-button-row" style={{ marginTop: 8 }} role="status">
                    <span className="st-chip" data-tone="warn">
                        <TriangleAlert size={10} /> {warnings.length} role model(s) not in the live catalog
                    </span>
                    {warnings.slice(0, 3).map((warning) => (
                        <span className="st-chip" key={`${warning.role}-${warning.model}`}>
                            {warning.role}: {warning.model}
                        </span>
                    ))}
                </div>
            )}

            {loading && !draft ? (
                <div className="st-value" style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 12 }}>
                    <Loader2 size={14} className="animate-spin" /> Loading LLM settings…
                </div>
            ) : draft && (
                <>
                    <div className="st-button-row" style={{ marginTop: 12 }}>
                        <label style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: 11, color: 'var(--st-muted)' }}>
                            Active provider
                            <select
                                className="st-input"
                                aria-label="Active provider"
                                value={draft.active_provider}
                                onChange={(event) => {
                                    const next = event.target.value;
                                    patchSettings({
                                        active_provider: next,
                                        active_model: draft.providers[next]?.default_model || draft.active_model,
                                    });
                                }}
                            >
                                {providerIds.map((id) => (
                                    <option key={id} value={id}>{draft.providers[id].name}</option>
                                ))}
                            </select>
                        </label>
                        <label style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: 11, color: 'var(--st-muted)' }}>
                            Active model
                            <input
                                className="st-input st-mono"
                                aria-label="Active model"
                                value={draft.active_model}
                                onChange={(event) => patchSettings({ active_model: event.target.value })}
                            />
                        </label>
                        <label style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: 11, color: 'var(--st-muted)' }}>
                            Temperature
                            <input
                                className="st-input"
                                aria-label="Temperature"
                                type="number"
                                min={0}
                                max={2}
                                step={0.1}
                                value={draft.temperature}
                                onChange={(event) => patchSettings({ temperature: Number(event.target.value) })}
                            />
                        </label>
                        <label style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: 11, color: 'var(--st-muted)' }}>
                            Max tokens
                            <input
                                className="st-input"
                                aria-label="Max tokens"
                                type="number"
                                min={0}
                                step={50}
                                value={draft.max_tokens}
                                onChange={(event) => patchSettings({ max_tokens: Number(event.target.value) })}
                            />
                        </label>
                        <label style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 14, fontSize: 12 }}>
                            <input
                                type="checkbox"
                                aria-label="Air-gapped"
                                checked={draft.air_gapped}
                                onChange={(event) => patchSettings({ air_gapped: event.target.checked })}
                            />
                            Air-gapped
                        </label>
                        <span className="st-chip" data-tone={draft.cloud_enabled ? 'warn' : undefined}>
                            <Cloud size={10} /> cloud {draft.cloud_enabled ? 'enabled' : 'disabled'}
                        </span>
                    </div>

                    <h4 className="st-panel-title" style={{ marginTop: 16 }}>Providers</h4>
                    <div className="st-grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(270px, 1fr))' }}>
                        {providerIds.map((id) => {
                            const provider = draft.providers[id];
                            const result = verified[id];
                            const telemetry = provider.telemetry || {};
                            return (
                                <div key={id} className="st-panel" data-tone={provider.enabled ? undefined : 'muted'}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8 }}>
                                        <strong>{provider.name}</strong>
                                        <span className="st-chip" data-tone={provider.is_local ? 'good' : 'warn'}>
                                            {provider.is_local ? 'local' : 'cloud'}
                                        </span>
                                    </div>
                                    <div className="st-value st-mono" style={{ fontSize: 10, color: 'var(--st-faint)', marginTop: 2 }}>
                                        {id} · {provider.wire} · {provider.type}
                                    </div>
                                    <label style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 8, fontSize: 11 }}>
                                        <input
                                            type="checkbox"
                                            aria-label={`Enable ${provider.name}`}
                                            checked={provider.enabled}
                                            onChange={(event) => patchProvider(id, { enabled: event.target.checked })}
                                        />
                                        enabled
                                    </label>
                                    <label style={{ display: 'block', marginTop: 8, fontSize: 11, color: 'var(--st-muted)' }}>
                                        Base URL
                                        <input
                                            className="st-input st-mono"
                                            aria-label={`${provider.name} base URL`}
                                            value={provider.base_url}
                                            placeholder={provider.type === 'gemma' ? 'in-process' : 'http://127.0.0.1:11434'}
                                            disabled={provider.type === 'gemma'}
                                            onChange={(event) => patchProvider(id, { base_url: event.target.value })}
                                        />
                                    </label>
                                    <label style={{ display: 'block', marginTop: 6, fontSize: 11, color: 'var(--st-muted)' }}>
                                        Default model
                                        <input
                                            className="st-input st-mono"
                                            aria-label={`${provider.name} default model`}
                                            list={models[id] ? `llm-models-${id}` : undefined}
                                            value={provider.default_model}
                                            disabled={provider.type === 'gemma'}
                                            onChange={(event) => patchProvider(id, { default_model: event.target.value })}
                                        />
                                        {models[id] && (
                                            <datalist id={`llm-models-${id}`}>
                                                {models[id].filter((row) => row.chat !== false).map((row) => (
                                                    <option key={row.id} value={row.id}>
                                                        {row.parameter_size
                                                            ? `${row.id} (${row.parameter_size})`
                                                            : row.size ? `${row.id} (${formatBytes(row.size)})` : row.id}
                                                    </option>
                                                ))}
                                            </datalist>
                                        )}
                                    </label>
                                    <div className="st-button-row" style={{ marginTop: 6, gap: 8 }}>
                                        <label style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 10, color: 'var(--st-muted)' }}>
                                            timeout
                                            <input
                                                className="st-input"
                                                aria-label={`${provider.name} timeout`}
                                                type="number"
                                                min={5}
                                                step={5}
                                                style={{ width: 70 }}
                                                value={provider.timeout_sec}
                                                onChange={(event) => patchProvider(id, { timeout_sec: Number(event.target.value) })}
                                            />
                                        </label>
                                        <label style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 10, color: 'var(--st-muted)' }}>
                                            keep-alive
                                            <input
                                                className="st-input st-mono"
                                                aria-label={`${provider.name} keep alive`}
                                                style={{ width: 64 }}
                                                value={provider.keep_alive}
                                                onChange={(event) => patchProvider(id, { keep_alive: event.target.value })}
                                            />
                                        </label>
                                    </div>
                                    {!provider.is_local && (
                                        <label style={{ display: 'block', marginTop: 6, fontSize: 11, color: 'var(--st-muted)' }}>
                                            API key
                                            <input
                                                className="st-input st-mono"
                                                aria-label={`${provider.name} API key`}
                                                type="password"
                                                value={provider.api_key || ''}
                                                placeholder={provider.has_key ? 'stored — type to replace' : 'not set'}
                                                onChange={(event) => patchProvider(id, { api_key: event.target.value, clear_key: false })}
                                            />
                                        </label>
                                    )}
                                    <div className="st-button-row" style={{ marginTop: 8 }}>
                                        <button
                                            type="button"
                                            className="st-button"
                                            data-size="sm"
                                            disabled={busyProvider === id || provider.type === 'gemma'}
                                            onClick={() => void loadModels(id)}
                                        >
                                            {busyProvider === id ? <Loader2 size={11} className="animate-spin" /> : null} List models
                                        </button>
                                        <button
                                            type="button"
                                            className="st-button"
                                            data-size="sm"
                                            disabled={busyProvider === id}
                                            onClick={() => void verify(id)}
                                        >
                                            {busyProvider === id ? <Loader2 size={11} className="animate-spin" /> : <ShieldCheck size={11} />} Verify
                                        </button>
                                        <button
                                            type="button"
                                            className="st-button"
                                            data-size="sm"
                                            disabled={busyProvider === id || provider.type === 'gemma' || !provider.default_model}
                                            onClick={() => void warm(id)}
                                        >
                                            <Thermometer size={11} /> Warm
                                        </button>
                                        {provider.has_key && !provider.is_local && (
                                            <button
                                                type="button"
                                                className="st-button"
                                                data-size="sm"
                                                data-variant="danger"
                                                onClick={() => patchProvider(id, { api_key: null, clear_key: true })}
                                            >
                                                <Trash2 size={11} /> Clear key
                                            </button>
                                        )}
                                        {id !== 'gemma' && (
                                            <button
                                                type="button"
                                                className="st-button"
                                                data-size="sm"
                                                data-variant="ghost"
                                                aria-label={`Remove ${provider.name}`}
                                                disabled={busyProvider === id}
                                                onClick={() => void removeProvider(id)}
                                            >
                                                <Trash2 size={11} /> Remove
                                            </button>
                                        )}
                                    </div>
                                    <div className="st-button-row" style={{ marginTop: 6 }}>
                                        <span className="st-chip" data-tone={provider.key_status === 'missing' ? 'bad' : undefined}>
                                            key {provider.key_status}
                                        </span>
                                        {provider.has_key && <span className="st-chip">source {provider.secret_source}</span>}
                                        {result && (
                                            <span className="st-chip" data-tone={result.ok ? 'good' : 'bad'}>
                                                {result.ok
                                                    ? `ok · ${result.latency_ms} ms · ${result.models.length} model(s)`
                                                    : `${result.fault || 'failed'}: ${result.error}`}
                                            </span>
                                        )}
                                        {!result && provider.last_fault && (
                                            <span className="st-chip" data-tone="bad" title={provider.last_fault}>
                                                last fault
                                            </span>
                                        )}
                                        {telemetry.last_load_ms ? (
                                            <span className="st-chip">load {telemetry.last_load_ms} ms</span>
                                        ) : null}
                                        {telemetry.last_tokens_per_sec ? (
                                            <span className="st-chip">{telemetry.last_tokens_per_sec} tok/s</span>
                                        ) : null}
                                    </div>
                                </div>
                            );
                        })}
                    </div>

                    <div className="st-button-row" style={{ marginTop: 10 }}>
                        <input
                            className="st-input"
                            aria-label="New endpoint name"
                            placeholder="Name (optional)"
                            value={newName}
                            onChange={(event) => setNewName(event.target.value)}
                        />
                        <input
                            className="st-input st-mono"
                            aria-label="New endpoint URL"
                            placeholder="http://127.0.0.1:8081"
                            style={{ flex: 1, minWidth: 220 }}
                            value={newUrl}
                            onChange={(event) => setNewUrl(event.target.value)}
                        />
                        <button type="button" className="st-button" data-size="sm" onClick={() => void addEndpoint()} disabled={!newUrl.trim()}>
                            <Plus size={12} /> Add endpoint
                        </button>
                    </div>

                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8, marginTop: 16 }}>
                        <h4 className="st-panel-title" style={{ margin: 0 }}>Local discovery</h4>
                        <button type="button" className="st-button" data-size="sm" disabled={scanning} onClick={() => void scan()}>
                            {scanning ? <Loader2 size={12} className="animate-spin" /> : <Radar size={12} />}
                            {scanning ? 'Scanning…' : 'Scan local servers'}
                        </button>
                    </div>
                    {discovered && (
                        <div className="st-link-list" style={{ marginTop: 8 }}>
                            {discovered.length === 0 && (
                                <div className="st-value" style={{ color: 'var(--st-muted)' }}>
                                    No servers answered with a model catalog on the usual ports.
                                </div>
                            )}
                            {discovered.map((server) => (
                                <div className="st-link-row" key={`${server.provider}-${server.port}-${server.base_url}`}>
                                    <span className="st-chip" data-tone="good">{server.provider}</span>
                                    <span style={{ flex: 1 }}>
                                        {server.name}
                                        <span className="st-mono" style={{ color: 'var(--st-faint)', marginLeft: 8 }}>{server.base_url}</span>
                                        {server.models.length > 0 && (
                                            <span style={{ color: 'var(--st-muted)' }}> · {server.models.length} model(s)</span>
                                        )}
                                        {server.needs_key && <span style={{ color: 'var(--st-warn)' }}> · needs key</span>}
                                        {server.attached && <span style={{ color: 'var(--st-success)' }}> · attached</span>}
                                    </span>
                                    <button
                                        type="button"
                                        className="st-button"
                                        data-size="sm"
                                        disabled={busyProvider === (server.instance_id || server.base_url)}
                                        onClick={() => void attach(server)}
                                    >
                                        {server.attached ? 'Refresh' : 'Attach'}
                                    </button>
                                </div>
                            ))}
                        </div>
                    )}

                    {catalogRows.length > 0 && catalogProvider && (
                        <div style={{ marginTop: 12 }}>
                            <h4 className="st-panel-title" style={{ marginBottom: 6 }}>
                                Catalog · {draft.providers[catalogProvider]?.name || catalogProvider}
                            </h4>
                            <table className="st-table">
                                <thead>
                                    <tr>
                                        <th scope="col">Model</th>
                                        <th scope="col">Context</th>
                                        <th scope="col">Params</th>
                                        <th scope="col">Vision</th>
                                        <th scope="col">Chat</th>
                                        <th scope="col">Use</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {catalogRows.map((row) => (
                                        <tr key={row.id}>
                                            <td className="st-mono">{row.id}</td>
                                            <td>{formatContext(row.context_length)}</td>
                                            <td>{row.parameter_size || row.quantization || '—'}</td>
                                            <td>{row.vision ? <Eye size={11} /> : '—'}</td>
                                            <td>
                                                {row.chat === false ? (
                                                    <span className="st-chip" data-tone="warn">
                                                        {row.embedding_only ? 'embedding' : 'tool/audio'}
                                                    </span>
                                                ) : 'chat'}
                                            </td>
                                            <td>
                                                <button
                                                    type="button"
                                                    className="st-button"
                                                    data-size="sm"
                                                    disabled={row.chat === false}
                                                    onClick={() => patchProvider(catalogProvider, { default_model: row.id })}
                                                >
                                                    Set default
                                                </button>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    )}

                    <h4 className="st-panel-title" style={{ marginTop: 16 }}>Role routing</h4>
                    <p className="st-value" style={{ margin: '0 0 8px', color: 'var(--st-muted)', fontSize: 11 }}>
                        Each filmmaking job resolves independently and can be left unassigned. Gemma only
                        serves video prompts inside a loaded LTX pipeline; script parsing and image prompts
                        run on a local HTTP model.
                    </p>
                    {Object.entries(groupedRoles).map(([group, metas]) => (
                        <div key={group} style={{ marginBottom: 10 }}>
                            <div className="st-value" style={{ fontSize: 11, color: 'var(--st-faint)', textTransform: 'uppercase', letterSpacing: '.06em' }}>
                                {ROLE_GROUPS[group] || group}
                            </div>
                            <table className="st-table">
                                <thead>
                                    <tr>
                                        <th scope="col">Role</th>
                                        <th scope="col">Provider</th>
                                        <th scope="col">Model</th>
                                        <th scope="col">Fallback</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {metas.map((meta) => {
                                        const role = draft.roles[meta.role];
                                        if (!role) return null;
                                        const provider = draft.providers[role.provider];
                                        const context = role.model && models[role.provider]
                                            ? models[role.provider].find((row) => row.id === role.model)?.context_length
                                            : null;
                                        return (
                                            <tr key={meta.role}>
                                                <td>
                                                    <strong>{meta.label}</strong>
                                                    <div style={{ color: 'var(--st-faint)', fontSize: 10 }}>{meta.description}</div>
                                                </td>
                                                <td>
                                                    <select
                                                        className="st-input"
                                                        aria-label={`${meta.label} provider`}
                                                        value={role.provider}
                                                        onChange={(event) => {
                                                            const providerId = event.target.value;
                                                            patchRole(meta.role, {
                                                                provider: providerId,
                                                                model: providerId
                                                                    ? (draft.providers[providerId]?.default_model || '')
                                                                    : '',
                                                                fallback: providerId ? role.fallback : [],
                                                            });
                                                        }}
                                                    >
                                                        <option value="">unassigned</option>
                                                        {providerIds.map((id) => (
                                                            <option key={id} value={id}>{id}</option>
                                                        ))}
                                                    </select>
                                                </td>
                                                <td>
                                                    <input
                                                        className="st-input st-mono"
                                                        aria-label={`${meta.label} model`}
                                                        list={models[role.provider] ? `llm-models-${role.provider}` : undefined}
                                                        value={role.model}
                                                        disabled={!role.provider}
                                                        onChange={(event) => patchRole(meta.role, { model: event.target.value })}
                                                    />
                                                    {provider && context ? (
                                                        <div style={{ color: 'var(--st-faint)', fontSize: 10 }}>context {formatContext(context)}</div>
                                                    ) : null}
                                                </td>
                                                <td>
                                                    <div className="st-button-row">
                                                        {role.fallback.map((hop, index) => (
                                                            <span className="st-chip" key={`${hop.provider}-${index}`}>
                                                                {hop.provider}/{hop.model || '…'}
                                                                <button
                                                                    type="button"
                                                                    className="st-button"
                                                                    data-size="sm"
                                                                    data-variant="ghost"
                                                                    aria-label={`Remove fallback ${index + 1} for ${meta.label}`}
                                                                    onClick={() => patchRole(meta.role, {
                                                                        fallback: role.fallback.filter((_, i) => i !== index),
                                                                    })}
                                                                >
                                                                    ×
                                                                </button>
                                                            </span>
                                                        ))}
                                                        <button
                                                            type="button"
                                                            className="st-button"
                                                            data-size="sm"
                                                            data-variant="ghost"
                                                            aria-label={`Add fallback for ${meta.label}`}
                                                            disabled={!role.provider}
                                                            onClick={() => {
                                                                const nextProvider = providerIds.find((id) => id !== role.provider) || role.provider;
                                                                if (!nextProvider) return;
                                                                patchRole(meta.role, {
                                                                    fallback: [
                                                                        ...role.fallback,
                                                                        {
                                                                            provider: nextProvider,
                                                                            model: draft.providers[nextProvider]?.default_model || '',
                                                                        },
                                                                    ],
                                                                });
                                                            }}
                                                        >
                                                            + fallback
                                                        </button>
                                                    </div>
                                                </td>
                                            </tr>
                                        );
                                    })}
                                </tbody>
                            </table>
                        </div>
                    ))}

                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8, marginTop: 16, flexWrap: 'wrap' }}>
                        <h4 className="st-panel-title" style={{ margin: 0 }}>
                            <Sparkles size={13} style={{ verticalAlign: -2, marginRight: 5 }} />
                            Local role eval
                        </h4>
                        <div className="st-button-row">
                            <span className="st-chip" data-tone={evalStale ? 'warn' : undefined}>
                                {draft.last_eval
                                    ? `last ${(draft.last_eval.pass_rate * 100).toFixed(0)}% · ${draft.last_eval.cases} cases${evalStale ? ' · stale' : ''}`
                                    : 'no eval on record'}
                            </span>
                            <button type="button" className="st-button" data-size="sm" disabled={evalRunning} onClick={() => void runEval()}>
                                {evalRunning ? <Loader2 size={12} className="animate-spin" /> : <Play size={12} />}
                                {evalRunning ? 'Running…' : 'Run eval'}
                            </button>
                            <button
                                type="button"
                                className="st-button"
                                data-size="sm"
                                data-variant="primary"
                                disabled={!draft.last_eval || profileBusy === 'eval'}
                                onClick={() => void applyRecommendation()}
                            >
                                Apply recommended
                            </button>
                        </div>
                    </div>
                    <p className="st-value" style={{ margin: '4px 0 0', color: 'var(--st-faint)', fontSize: 10 }}>
                        Runs shot, image, and script fixtures through the saved plan. It can take minutes on a
                        large local model.
                    </p>
                    {evalResult?.summary && (
                        <table className="st-table" style={{ marginTop: 8 }}>
                            <thead>
                                <tr>
                                    <th scope="col">Role</th>
                                    <th scope="col">Provider</th>
                                    <th scope="col">Model</th>
                                    <th scope="col">Pass</th>
                                    <th scope="col">p50</th>
                                </tr>
                            </thead>
                            <tbody>
                                {Object.entries(evalResult.summary.roles || {}).map(([role, row]) => (
                                    <tr key={role}>
                                        <td>{role}</td>
                                        <td className="st-mono">{row.provider || '—'}</td>
                                        <td className="st-mono">{row.model || '—'}</td>
                                        <td>{Math.round(Number(row.pass_rate || 0) * 100)}%</td>
                                        <td>{row.p50_ms} ms</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    )}

                    <h4 className="st-panel-title" style={{ marginTop: 16 }}>Budget &amp; profiles</h4>
                    <div className="st-button-row">
                        <label style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 11, color: 'var(--st-muted)' }}>
                            daily cloud budget $
                            <input
                                className="st-input"
                                aria-label="Daily cloud budget"
                                type="number"
                                min={0}
                                step={1}
                                style={{ width: 90 }}
                                value={draft.budget.daily_usd}
                                onChange={(event) => patchSettings({
                                    budget: { ...draft.budget, daily_usd: Number(event.target.value) },
                                })}
                            />
                        </label>
                        <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11 }}>
                            <input
                                type="checkbox"
                                aria-label="Eval gate for cloud roles"
                                checked={draft.budget.eval_gate}
                                onChange={(event) => patchSettings({
                                    budget: { ...draft.budget, eval_gate: event.target.checked },
                                })}
                            />
                            eval gate
                        </label>
                        <span className="st-chip">
                            spent today ${(draft.spend?.total_usd ?? 0).toFixed(4)}
                        </span>
                        {Object.entries(PROFILE_LABELS).map(([profile, label]) => (
                            <button
                                key={profile}
                                type="button"
                                className="st-button"
                                data-size="sm"
                                disabled={Boolean(profileBusy)}
                                onClick={() => void applyProfile(profile)}
                            >
                                {profileBusy === profile ? <Loader2 size={11} className="animate-spin" /> : null}
                                {label}
                            </button>
                        ))}
                    </div>

                    <h4 className="st-panel-title" style={{ marginTop: 16 }}>Test saved configuration</h4>
                    <div className="st-button-row">
                        <select
                            className="st-input"
                            aria-label="Test role"
                            value={testRole}
                            onChange={(event) => setTestRole(event.target.value)}
                        >
                            {roleCatalog.map((meta) => (
                                <option key={meta.role} value={meta.role}>{meta.label}</option>
                            ))}
                        </select>
                        <input
                            className="st-input"
                            aria-label="Test prompt"
                            style={{ flex: 1, minWidth: 220 }}
                            value={testPrompt}
                            onChange={(event) => setTestPrompt(event.target.value)}
                        />
                        <button type="button" className="st-button" data-size="sm" disabled={testing} onClick={() => void runTest()}>
                            {testing ? <Loader2 size={12} className="animate-spin" /> : <Play size={12} />}
                            {testing ? 'Streaming…' : 'Run test'}
                        </button>
                    </div>
                    <p className="st-value" style={{ margin: '6px 0 0', color: 'var(--st-faint)', fontSize: 10 }}>
                        Tests the saved configuration, not the unsaved draft. Streams when the engine supports
                        it and falls back to a single response when it does not.
                    </p>
                    {(testReply || testError || testMeta) && (
                        <div className="st-panel" style={{ marginTop: 8 }} role="status">
                            {testError ? (
                                <div style={{ color: 'var(--st-danger)', display: 'flex', gap: 6, alignItems: 'flex-start' }}>
                                    <TriangleAlert size={13} /> <span>{testError}</span>
                                </div>
                            ) : (
                                <>
                                    <div className="st-button-row">
                                        {testMeta?.provider && (
                                            <span className="st-chip" data-tone="good">
                                                <CheckCircle2 size={10} /> {testMeta.provider}/{testMeta.model}
                                            </span>
                                        )}
                                        {testMeta?.latency_ms ? <span className="st-chip">{testMeta.latency_ms} ms</span> : null}
                                        {testing && <span className="st-chip">streaming…</span>}
                                    </div>
                                    <div className="st-value" style={{ marginTop: 6, whiteSpace: 'pre-wrap', fontSize: 12 }}>
                                        {testReply}
                                    </div>
                                </>
                            )}
                        </div>
                    )}
                </>
            )}
        </div>
    );
}
