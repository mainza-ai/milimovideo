import { useCallback, useEffect, useMemo, useState } from 'react';
import {
    AlertCircle, CheckCircle2, Cloud, Cpu, Eye, Loader2, Play, Radar, RefreshCw, RotateCcw,
    Save, ShieldAlert, ShieldCheck, Trash2, TriangleAlert,
} from 'lucide-react';
import { apiUrl } from '../../config';
import { useStudioDocumentStore } from '../state/documentStore';

interface ProviderConfig {
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
}

interface ModelRow {
    id: string;
    name: string;
    size?: number | null;
    parameter_size?: string | null;
    vision?: boolean;
    context_length?: number | null;
}

interface VerifyResult {
    ok: boolean;
    provider: string;
    latency_ms: number;
    models: string[];
    error?: string | null;
    fault?: string | null;
}

interface DiscoveredServer {
    provider: string;
    name: string;
    wire: string;
    base_url: string;
    port: number;
    models: string[];
}

const ROLE_GROUPS: Record<string, string> = {
    prompt: 'Prompt rewriting',
    story: 'Story & script',
    agent: 'Agent & review',
    review: 'Review & continuity',
};

function payloadFor(settings: LLMSettings): Record<string, unknown> {
    const providers: Record<string, unknown> = {};
    for (const [id, provider] of Object.entries(settings.providers)) {
        providers[id] = {
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
    };
}

export function LLMSettingsPanel() {
    const addToast = useStudioDocumentStore((state) => state.addToast);
    const [settings, setSettings] = useState<LLMSettings | null>(null);
    const [draft, setDraft] = useState<LLMSettings | null>(null);
    const [loading, setLoading] = useState(true);
    const [saving, setSaving] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [conflict, setConflict] = useState(false);
    const [models, setModels] = useState<Record<string, ModelRow[]>>({});
    const [busyProvider, setBusyProvider] = useState<string | null>(null);
    const [verified, setVerified] = useState<Record<string, VerifyResult>>({});
    const [discovered, setDiscovered] = useState<DiscoveredServer[] | null>(null);
    const [scanning, setScanning] = useState(false);
    const [testRole, setTestRole] = useState('prompt_video');
    const [testPrompt, setTestPrompt] = useState('Describe a slow dolly-in on a rain-soaked street at night.');
    const [testing, setTesting] = useState(false);
    const [testResult, setTestResult] = useState<{ reply?: string; provider?: string; model?: string; latency_ms?: number; error?: string } | null>(null);

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
            setDraft(data);
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

    const save = async () => {
        if (!draft || !settings) return;
        setSaving(true);
        setError(null);
        try {
            const response = await fetch(
                apiUrl(`/settings/llm?expected_revision=${settings.revision}`),
                {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payloadFor(draft)),
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
            setSettings(data as LLMSettings);
            setDraft(data as LLMSettings);
            setConflict(false);
            addToast(`LLM settings saved (revision ${(data as LLMSettings).revision})`, 'success');
        } catch (exc) {
            setError(exc instanceof Error ? exc.message : 'Save failed');
        } finally {
            setSaving(false);
        }
    };

    const loadModels = async (providerId: string) => {
        setBusyProvider(providerId);
        try {
            const response = await fetch(apiUrl(`/settings/llm/models?provider=${encodeURIComponent(providerId)}`));
            const data = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Model list failed');
            setModels((current) => ({ ...current, [providerId]: data.models || [] }));
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

    const attach = (server: DiscoveredServer) => {
        setDraft((current) => {
            if (!current) return current;
            const providerId = current.providers[server.provider] ? server.provider : 'openai_compatible';
            const provider = current.providers[providerId];
            if (!provider) return current;
            return {
                ...current,
                providers: {
                    ...current.providers,
                    [providerId]: {
                        ...provider,
                        base_url: server.base_url,
                        wire: server.wire,
                        enabled: true,
                        default_model: provider.default_model || server.models[0] || '',
                    },
                },
            };
        });
        addToast(`Attached ${server.name} at ${server.base_url}`, 'info');
    };

    const runTest = async () => {
        setTesting(true);
        setTestResult(null);
        try {
            const response = await fetch(apiUrl('/settings/llm/test'), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ role: testRole, prompt: testPrompt }),
            });
            const data = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Test failed');
            setTestResult(data);
        } catch (exc) {
            setTestResult({ error: exc instanceof Error ? exc.message : 'Test failed' });
        } finally {
            setTesting(false);
        }
    };

    const providerIds = draft ? Object.keys(draft.providers) : [];
    const groupedRoles = roleCatalog.reduce<Record<string, RoleMeta[]>>((groups, meta) => {
        const key = meta.group || 'prompt';
        groups[key] = [...(groups[key] || []), meta];
        return groups;
    }, {});

    return (
        <div className="st-panel" style={{ marginBottom: 14 }} aria-labelledby="llm-panel-title" data-testid="llm-panel">
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 10, flexWrap: 'wrap' }}>
                <div>
                    <h3 className="st-panel-title" id="llm-panel-title" style={{ margin: 0 }}>
                        <Cpu size={14} style={{ verticalAlign: -2, marginRight: 6 }} />
                        AI &amp; LLM
                    </h3>
                    <p className="st-value" style={{ margin: '4px 0 0', color: 'var(--st-muted)', fontSize: 11 }}>
                        Persisted provider and role routing. Local-first: cloud providers stay disabled
                        unless <span className="st-mono">MILIMO_LLM_CLOUD=1</span> and the air-gap is off.
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
                    <div className="st-grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))' }}>
                        {providerIds.map((id) => {
                            const provider = draft.providers[id];
                            const result = verified[id];
                            const rows = models[id];
                            return (
                                <div key={id} className="st-panel" data-tone={provider.enabled ? undefined : 'muted'}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8 }}>
                                        <strong>{provider.name}</strong>
                                        <span className="st-chip" data-tone={provider.is_local ? 'good' : 'warn'}>
                                            {provider.is_local ? 'local' : 'cloud'}
                                        </span>
                                    </div>
                                    <div className="st-value st-mono" style={{ fontSize: 10, color: 'var(--st-faint)', marginTop: 2 }}>
                                        {id} · {provider.wire}
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
                                            list={rows ? `llm-models-${id}` : undefined}
                                            value={provider.default_model}
                                            disabled={provider.type === 'gemma'}
                                            onChange={(event) => patchProvider(id, { default_model: event.target.value })}
                                        />
                                        {rows && (
                                            <datalist id={`llm-models-${id}`}>
                                                {rows.map((row) => (
                                                    <option key={row.id} value={row.id}>
                                                        {row.parameter_size
                                                            ? `${row.id} (${row.parameter_size})`
                                                            : row.size ? `${row.id} (${Math.round(row.size / 1e6)} MB)` : row.id}
                                                    </option>
                                                ))}
                                            </datalist>
                                        )}
                                    </label>
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
                                        {!result && provider.last_success_at && (
                                            <span className="st-chip" data-tone="good">last ok</span>
                                        )}
                                    </div>
                                </div>
                            );
                        })}
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
                                    No servers answered on 11434, 1234, 8080, 8081, 8899, 8000, 8787, or 47831.
                                </div>
                            )}
                            {discovered.map((server) => (
                                <div className="st-link-row" key={`${server.provider}-${server.port}`}>
                                    <span className="st-chip" data-tone="good">{server.provider}</span>
                                    <span style={{ flex: 1 }}>
                                        {server.name}
                                        <span className="st-mono" style={{ color: 'var(--st-faint)', marginLeft: 8 }}>{server.base_url}</span>
                                        {server.models.length > 0 && (
                                            <span style={{ color: 'var(--st-muted)' }}> · {server.models.length} model(s)</span>
                                        )}
                                    </span>
                                    <button type="button" className="st-button" data-size="sm" onClick={() => attach(server)}>
                                        Attach
                                    </button>
                                </div>
                            ))}
                        </div>
                    )}

                    <h4 className="st-panel-title" style={{ marginTop: 16 }}>Role routing</h4>
                    <p className="st-value" style={{ margin: '0 0 8px', color: 'var(--st-muted)', fontSize: 11 }}>
                        Each filmmaking job resolves independently. Gemma only serves video prompts inside a
                        loaded LTX pipeline; script parsing and image prompts run happily on a local HTTP model.
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
                                                                model: draft.providers[providerId]?.default_model || role.model,
                                                            });
                                                        }}
                                                    >
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
                                                        onChange={(event) => patchRole(meta.role, { model: event.target.value })}
                                                    />
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
                                                            onClick={() => {
                                                                const nextProvider = providerIds.find((id) => id !== role.provider) || role.provider;
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
                            {testing ? 'Running…' : 'Run test'}
                        </button>
                    </div>
                    <p className="st-value" style={{ margin: '6px 0 0', color: 'var(--st-faint)', fontSize: 10 }}>
                        Tests the saved configuration, not the unsaved draft. Save first to test edits.
                    </p>
                    {testResult && (
                        <div className="st-panel" style={{ marginTop: 8 }} role="status">
                            {testResult.error ? (
                                <div style={{ color: 'var(--st-danger)', display: 'flex', gap: 6, alignItems: 'flex-start' }}>
                                    <TriangleAlert size={13} /> <span>{testResult.error}</span>
                                </div>
                            ) : (
                                <>
                                    <div className="st-button-row">
                                        <span className="st-chip" data-tone="good">
                                            <CheckCircle2 size={10} /> {testResult.provider}/{testResult.model}
                                        </span>
                                        <span className="st-chip">{testResult.latency_ms} ms</span>
                                    </div>
                                    <div className="st-value" style={{ marginTop: 6, whiteSpace: 'pre-wrap', fontSize: 12 }}>
                                        {testResult.reply}
                                    </div>
                                </>
                            )}
                        </div>
                    )}

                    {models[Object.keys(models)[0]]?.some((row) => row.vision) && (
                        <p className="st-value" style={{ marginTop: 10, color: 'var(--st-faint)', fontSize: 10 }}>
                            <Eye size={10} style={{ verticalAlign: -1 }} /> Vision-capable models are listed, but
                            Milimo only sends image input on the video-prompt path.
                        </p>
                    )}
                </>
            )}
        </div>
    );
}
