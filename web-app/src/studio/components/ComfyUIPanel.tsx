import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
    CheckCircle2, Copy, Download, ExternalLink, Loader2, Play, PowerOff, RefreshCw,
    Trash2, TriangleAlert, Upload, Workflow,
} from 'lucide-react';
import { apiFetch, apiUrl, getAssetUrl } from '../../config';
import { saveTextFile } from '../../desktop';
import { useStudioDocumentStore } from '../state/documentStore';
import { useStudioJobs } from '../state/jobStore';
import { useStudioActiveJobs, useStudioAssets } from '../data/useStudioData';
import { registerStudioJob } from '../events/jobBridge';
import { useMediaPreview } from '../state/mediaPreviewStore';
import { MediaThumb } from '../components/MediaThumb';
import {
    BINDING_FIELDS,
    applyBindings,
    detectBindings,
    formatWorkflow,
    outputKind,
    parseWorkflowText,
    scalarInputs,
    type ComfyBinding,
    type ComfyBindings,
    type ComfyWorkflow,
} from '../comfy/workflow';

export interface ComfyHealth {
    enabled?: boolean;
    reachable: boolean;
    url?: string;
    version?: string | null;
    compatible?: boolean;
    min_version?: string;
    queue_running?: number | null;
    queue_pending?: number | null;
    ram_free_gb?: number | null;
    ram_total_gb?: number | null;
    declared_gb?: number;
    has_token?: boolean;
    hard_disabled?: boolean;
    error?: string;
}

interface ComfySettings {
    revision: number;
    url: string;
    enabled: boolean;
    declared_gb: number;
    has_token: boolean;
    hard_disabled: boolean;
}

interface ValidationError {
    node_id: string;
    class_type: string;
    field?: string | null;
    message: string;
}

interface Validation {
    nodes?: number;
    node_count?: number;
    output_nodes: string[];
    width?: number | null;
    height?: number | null;
    warnings: string[];
    errors?: ValidationError[];
    live?: boolean;
    ok?: boolean;
    error?: string;
}

interface SavedWorkflow {
    id: string;
    name: string;
    description: string;
    workflow: ComfyWorkflow;
    bindings: ComfyBindings;
    revision: number;
    is_default: boolean;
}

interface CatalogNode {
    class_type: string;
    display_name: string;
    category: string;
    output_node: boolean;
}

interface NodeSpec {
    input?: { required?: Record<string, any>; optional?: Record<string, any> };
    output?: string[];
    output_node?: boolean;
    category?: string;
    display_name?: string;
}

const SMOKE_WORKFLOW = JSON.stringify({
    '1': {
        class_type: 'EmptyImage',
        inputs: { width: 640, height: 360, batch_size: 1, color: 0 },
    },
    '2': {
        class_type: 'SaveImage',
        inputs: { images: ['1', 0], filename_prefix: 'milimo' },
    },
}, null, 2);

const WORKFLOW_STORAGE_KEY = 'milimo-comfy-workflow';

export function useComfyHealth(pollMs = 15000) {
    const [health, setHealth] = useState<ComfyHealth | null>(null);
    const [loading, setLoading] = useState(true);
    const refresh = useCallback(async () => {
        try {
            const response = await apiFetch('/comfy/health');
            const data = await response.json().catch(() => ({}));
            setHealth(response.ok ? data : { reachable: false, error: data.detail || `HTTP ${response.status}` });
        } catch (error) {
            setHealth({ reachable: false, error: error instanceof Error ? error.message : 'ComfyUI probe failed' });
        } finally {
            setLoading(false);
        }
    }, []);
    useEffect(() => {
        void refresh();
        if (pollMs <= 0) return undefined;
        const timer = window.setInterval(() => void refresh(), pollMs);
        return () => window.clearInterval(timer);
    }, [pollMs, refresh]);
    return { health, loading, refresh };
}

function useComfySettings() {
    const [settings, setSettings] = useState<ComfySettings | null>(null);
    const refresh = useCallback(async () => {
        const response = await apiFetch('/comfy/settings');
        const data = await response.json().catch(() => ({}));
        if (response.ok) setSettings(data);
        return data;
    }, []);
    useEffect(() => { void refresh(); }, [refresh]);
    return { settings, setSettings, refresh };
}

function ComfyStatus({ health }: { health: ComfyHealth | null }) {
    if (!health) {
        return <span className="st-status-pill" data-tone="warn"><Loader2 size={10} className="animate-spin" /> checking</span>;
    }
    if (health.enabled === false) {
        return (
            <span className="st-status-pill" data-tone="warn" title="ComfyUI bridge is disabled">
                <PowerOff size={10} /> disabled
            </span>
        );
    }
    if (!health.reachable) {
        return (
            <span className="st-status-pill" data-tone="bad" title={health.error || ''}>
                <TriangleAlert size={10} /> offline
            </span>
        );
    }
    return (
        <span className="st-status-pill" data-tone="good" title={health.url || ''}>
            <CheckCircle2 size={10} /> {health.version || 'ComfyUI'}
        </span>
    );
}

export function ComfyHealthPanel() {
    const { health, loading, refresh } = useComfyHealth();
    const { settings, setSettings, refresh: refreshSettings } = useComfySettings();
    const addToast = useStudioDocumentStore((state) => state.addToast);
    const [freeing, setFreeing] = useState(false);
    const [saving, setSaving] = useState(false);
    const [draft, setDraft] = useState<{ url: string; enabled: boolean; declared_gb: number; token: string }>({
        url: '', enabled: true, declared_gb: 2, token: '',
    });
    const [tokenDirty, setTokenDirty] = useState(false);

    useEffect(() => {
        if (!settings) return;
        setDraft({
            url: settings.url,
            enabled: settings.enabled,
            declared_gb: settings.declared_gb,
            token: '',
        });
        setTokenDirty(false);
    }, [settings]);

    const free = async () => {
        setFreeing(true);
        try {
            const response = await apiFetch('/comfy/free', { method: 'POST' });
            const data = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
            addToast('ComfyUI released its cached models', 'success');
            await refresh();
        } catch (error) {
            addToast(error instanceof Error ? error.message : 'ComfyUI free failed', 'error');
        } finally {
            setFreeing(false);
        }
    };

    const save = async () => {
        setSaving(true);
        try {
            const payload: Record<string, unknown> = {
                url: draft.url,
                enabled: draft.enabled,
                declared_gb: draft.declared_gb,
                expected_revision: settings?.revision ?? 0,
            };
            if (tokenDirty) payload.token = draft.token;
            const response = await apiFetch('/comfy/settings', {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            const data = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(data.detail?.message || data.detail || `HTTP ${response.status}`);
            setSettings(data);
            setTokenDirty(false);
            addToast('ComfyUI settings saved', 'success');
            await refresh();
            await refreshSettings();
        } catch (error) {
            addToast(error instanceof Error ? error.message : 'ComfyUI settings failed', 'error');
        } finally {
            setSaving(false);
        }
    };

    return (
        <div className="st-panel" style={{ marginBottom: 14 }} data-testid="comfy-setup">
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 10 }}>
                <h3 className="st-panel-title" style={{ margin: 0 }}>
                    <Workflow size={13} style={{ verticalAlign: -2, marginRight: 6 }} />
                    ComfyUI bridge
                </h3>
                <div className="st-button-row">
                    <ComfyStatus health={health} />
                    {health?.enabled !== false && (
                        <button type="button" className="st-button" data-size="sm" disabled={loading} onClick={() => void refresh()}>
                            <RefreshCw size={12} className={loading ? 'animate-spin' : ''} /> Probe
                        </button>
                    )}
                </div>
            </div>
            <dl className="st-meta-grid" style={{ marginTop: 8 }}>
                <dt>URL</dt><dd className="st-mono">{health?.url || '—'}</dd>
                <dt>Queue</dt>
                <dd>{health?.enabled !== false && health?.reachable ? `${health.queue_running ?? 0} running · ${health.queue_pending ?? 0} pending` : '—'}</dd>
                <dt>Free RAM</dt>
                <dd>{health?.enabled !== false && health?.reachable && health.ram_free_gb != null ? `${health.ram_free_gb.toFixed(1)} / ${(health.ram_total_gb || 0).toFixed(1)} GB` : '—'}</dd>
            </dl>
            {health?.enabled === false && (
                <p className="st-value" style={{ marginTop: 6, color: 'var(--st-muted)', fontSize: 11 }}>
                    The ComfyUI bridge is optional and switched off. Core generation does not use it.
                    {health.hard_disabled ? ' It is hard-disabled by MILIMO_COMFY_ENABLED=0.' : ' Enable it below or remove the setting.'}
                </p>
            )}
            {health && health.enabled !== false && !health.reachable && (
                <p className="st-value" style={{ marginTop: 6, color: 'var(--st-muted)', fontSize: 11 }}>
                    {health.error || 'ComfyUI did not answer.'} Start Comfy Desktop or change the URL below.
                </p>
            )}
            <details style={{ marginTop: 8 }}>
                <summary className="st-label" style={{ cursor: 'pointer' }}>Connection settings</summary>
                <div className="st-row" style={{ marginTop: 8, alignItems: 'end', flexWrap: 'wrap' }}>
                    <label style={{ flex: 2, minWidth: 220 }}>
                        <span className="st-label">URL</span>
                        <input
                            className="st-input"
                            value={draft.url}
                            disabled={settings?.hard_disabled}
                            onChange={(event) => setDraft({ ...draft, url: event.target.value })}
                        />
                    </label>
                    <label style={{ minWidth: 90 }}>
                        <span className="st-label">Declared GB</span>
                        <input
                            className="st-input"
                            type="number"
                            min={0}
                            max={256}
                            step={0.5}
                            value={draft.declared_gb}
                            disabled={settings?.hard_disabled}
                            onChange={(event) => setDraft({ ...draft, declared_gb: Number(event.target.value) })}
                        />
                    </label>
                    <label style={{ minWidth: 170 }}>
                        <span className="st-label">Bearer token</span>
                        <input
                            className="st-input"
                            type="password"
                            placeholder={settings?.has_token ? 'stored — type to replace' : 'optional'}
                            value={draft.token}
                            disabled={settings?.hard_disabled}
                            onChange={(event) => {
                                setDraft({ ...draft, token: event.target.value });
                                setTokenDirty(true);
                            }}
                        />
                    </label>
                    <label className="st-chip" style={{ textTransform: 'none', letterSpacing: 0 }}>
                        <input
                            type="checkbox"
                            checked={draft.enabled}
                            disabled={settings?.hard_disabled}
                            onChange={(event) => setDraft({ ...draft, enabled: event.target.checked })}
                        />
                        enabled
                    </label>
                    <button
                        type="button"
                        className="st-button"
                        data-size="sm"
                        data-variant="primary"
                        disabled={saving || settings?.hard_disabled}
                        onClick={() => void save()}
                    >
                        {saving ? <Loader2 size={12} className="animate-spin" /> : <CheckCircle2 size={12} />}
                        Save
                    </button>
                </div>
            </details>
            {health?.enabled !== false && (
                <div className="st-button-row" style={{ marginTop: 8 }}>
                    <button
                        type="button"
                        className="st-button"
                        data-size="sm"
                        disabled={freeing || !health?.reachable}
                        onClick={() => void free()}
                    >
                        {freeing ? <Loader2 size={12} className="animate-spin" /> : <RefreshCw size={12} />}
                        Free cached models
                    </button>
                    {health?.url && (
                        <a className="st-button" data-size="sm" href={health.url} target="_blank" rel="noreferrer">
                            <ExternalLink size={12} /> Open ComfyUI
                        </a>
                    )}
                </div>
            )}
            <p className="st-value" style={{ marginTop: 8, color: 'var(--st-faint)', fontSize: 11 }}>
                ComfyUI runs as a separate GPU process. Its declared residency is admitted before a workflow,
                it is charged to the same queue, and <code>/free</code> releases its cache. SAM defaults to port 8002.
            </p>
        </div>
    );
}

function bindingPath(binding?: ComfyBinding): string {
    return binding ? `${binding.node}.${binding.input}` : '';
}

function findBinding(workflow: ComfyWorkflow, path: string): ComfyBinding | null {
    if (!path) return null;
    const [node, input] = path.split('.');
    return workflow[node] ? { node, input } : null;
}

function NodeInputs({
    workflow,
    nodeId,
    spec,
    onChange,
}: {
    workflow: ComfyWorkflow;
    nodeId: string;
    spec: NodeSpec | undefined;
    onChange: (nodeId: string, input: string, value: unknown) => void;
}) {
    const node = workflow[nodeId];
    if (!node) return null;
    const inputs = node.inputs || {};
    const required = spec?.input?.required || {};
    const optional = spec?.input?.optional || {};
    const fields = Array.from(new Set([...Object.keys(required), ...Object.keys(optional), ...Object.keys(inputs)]));

    return (
        <div className="st-link-list" style={{ marginTop: 8 }}>
            {fields.map((field) => {
                const value = inputs[field];
                if (Array.isArray(value) && value.length === 2) {
                    return (
                        <div key={field} className="st-link-row">
                            <span className="st-mono">{field}</span>
                            <span className="st-chip">linked from {String(value[0])}</span>
                        </div>
                    );
                }
                const rule = required[field] || optional[field];
                const type = Array.isArray(rule) ? rule[0] : undefined;
                const options = Array.isArray(type) ? type : null;
                return (
                    <label key={field} className="st-field">
                        <span>{field}{required[field] ? ' *' : ''}</span>
                        {options ? (
                            <select
                                className="st-select"
                                value={String(value ?? '')}
                                onChange={(event) => onChange(nodeId, field, event.target.value)}
                            >
                                {!options.includes(value) && <option value={String(value ?? '')}>{String(value ?? '—')}</option>}
                                {options.map((option: unknown) => (
                                    <option key={String(option)} value={String(option)}>{String(option)}</option>
                                ))}
                            </select>
                        ) : typeof value === 'boolean' || type === 'BOOLEAN' ? (
                            <input
                                type="checkbox"
                                checked={Boolean(value)}
                                onChange={(event) => onChange(nodeId, field, event.target.checked)}
                            />
                        ) : type === 'INT' || type === 'FLOAT' || typeof value === 'number' ? (
                            <input
                                className="st-input"
                                type="number"
                                value={Number(value ?? 0)}
                                onChange={(event) => onChange(nodeId, field, Number(event.target.value))}
                            />
                        ) : (
                            <input
                                className="st-input"
                                value={String(value ?? '')}
                                onChange={(event) => onChange(nodeId, field, event.target.value)}
                            />
                        )}
                    </label>
                );
            })}
        </div>
    );
}

export function ComfyWorkflowPanel({ projectId, shotId }: { projectId: string; shotId?: string | null }) {
    const addToast = useStudioDocumentStore((state) => state.addToast);
    const project = useStudioDocumentStore((state) => state.project);
    const shot = useStudioDocumentStore((state) =>
        state.project.shots.find((item) => item.id === shotId) || null,
    );
    const { health } = useComfyHealth();
    const { assets } = useStudioAssets();
    const { jobs: activeJobs, refresh: refreshJobs } = useStudioActiveJobs(projectId);
    const openMediaPreview = useMediaPreview((state) => state.open);
    const [text, setText] = useState(() => {
        try {
            return window.localStorage.getItem(`${WORKFLOW_STORAGE_KEY}:${projectId}`)
                || window.localStorage.getItem(WORKFLOW_STORAGE_KEY)
                || SMOKE_WORKFLOW;
        } catch {
            return SMOKE_WORKFLOW;
        }
    });
    const [bindings, setBindings] = useState<ComfyBindings>({});
    const [validation, setValidation] = useState<Validation | null>(null);
    const [busy, setBusy] = useState<'validate' | 'run' | null>(null);
    const [jobId, setJobId] = useState<string | null>(null);
    const [outputs, setOutputs] = useState<Array<Record<string, unknown>>>([]);
    const [workflows, setWorkflows] = useState<SavedWorkflow[]>([]);
    const [current, setCurrent] = useState<SavedWorkflow | null>(null);
    const [catalog, setCatalog] = useState<{ node_count: number; nodes: CatalogNode[]; templates: Record<string, string[]> } | null>(null);
    const [schemas, setSchemas] = useState<Record<string, NodeSpec>>({});
    const [selectedNode, setSelectedNode] = useState<string | null>(null);
    const [modelOptions, setModelOptions] = useState<string[]>([]);
    const [estimateSeconds, setEstimateSeconds] = useState(60);
    const [batchBusy, setBatchBusy] = useState(false);
    const [batchShots, setBatchShots] = useState<string[]>([]);
    const fileInput = useRef<HTMLInputElement>(null);
    const packInput = useRef<HTMLInputElement>(null);
    const loadedWorkflow = useRef(false);
    const parsed = useMemo(() => {
        try {
            return { workflow: parseWorkflowText(text), error: null as string | null };
        } catch (error) {
            return { workflow: null as ComfyWorkflow | null, error: error instanceof Error ? error.message : 'Invalid JSON' };
        }
    }, [text]);

    const activeComfy = activeJobs.find((job) =>
        job.kind === 'comfy' && (!shotId || job.shot_id === shotId),
    ) || null;
    const currentJob = activeComfy || (jobId ? activeJobs.find((job) => job.job_id === jobId) : null);

    const updateText = useCallback((next: ComfyWorkflow) => {
        setText(formatWorkflow(next));
        setValidation(null);
    }, []);

    const refreshWorkflows = useCallback(async () => {
        try {
            const response = await apiFetch(`/projects/${projectId}/comfy/workflows`);
            const data = response.ok ? await response.json() : [];
            setWorkflows(Array.isArray(data) ? data : []);
            return Array.isArray(data) ? data as SavedWorkflow[] : [];
        } catch {
            setWorkflows([]);
            return [];
        }
    }, [projectId]);

    useEffect(() => {
        void refreshWorkflows();
    }, [refreshWorkflows]);

    useEffect(() => {
        if (!health?.reachable) return;
        let alive = true;
        const load = async () => {
            try {
                const response = await apiFetch('/comfy/catalog');
                if (!response.ok) return;
                const data = await response.json();
                if (alive) setCatalog(data);
            } catch {
                // Discovery is optional; the editor still works with raw JSON.
            }
            try {
                const [checkpoints, diffusion] = await Promise.all([
                    apiFetch('/comfy/models?folder=checkpoints').then((r) => r.ok ? r.json() : { models: [] }),
                    apiFetch('/comfy/models?folder=diffusion_models').then((r) => r.ok ? r.json() : { models: [] }),
                ]);
                if (alive) setModelOptions([...(checkpoints.models || []), ...(diffusion.models || [])]);
            } catch {
                if (alive) setModelOptions([]);
            }
        };
        void load();
        return () => { alive = false; };
    }, [health?.reachable]);

    const classKey = parsed.workflow
        ? Array.from(new Set(Object.values(parsed.workflow).map((node: any) => String(node.class_type || '')))).sort().join(',')
        : '';
    useEffect(() => {
        if (!classKey || !health?.reachable) return;
        let alive = true;
        apiFetch(`/comfy/schema?classes=${encodeURIComponent(classKey)}`)
            .then((response) => response.ok ? response.json() : { classes: {} })
            .then((data) => { if (alive) setSchemas(data.classes || {}); })
            .catch(() => { if (alive) setSchemas({}); });
        return () => { alive = false; };
    }, [classKey, health?.reachable]);

    useEffect(() => {
        const workflow = parsed.workflow;
        if (!workflow) return;
        setSelectedNode((currentNode) => (
            currentNode && workflow[currentNode]
                ? currentNode
                : Object.keys(workflow)[0] || null
        ));
    }, [parsed.workflow]);

    useEffect(() => {
        if (activeComfy) setJobId(activeComfy.job_id);
    }, [activeComfy]);

    useEffect(() => {
        const handler = (event: Event) => {
            const detail = (event as CustomEvent).detail || {};
            if (!jobId || detail.job_id !== jobId) return;
            if (detail.event === 'job.completed') {
                setOutputs(Array.isArray(detail.outputs) ? detail.outputs : []);
                setJobId(null);
            } else if (detail.event === 'job.failed' || detail.event === 'job.cancelled') {
                setJobId(null);
            }
        };
        window.addEventListener('milimo:job-event', handler);
        return () => window.removeEventListener('milimo:job-event', handler);
    }, [jobId]);

    useEffect(() => {
        if (!currentJob) return;
        setJobId((current) => current || currentJob.job_id);
    }, [currentJob]);

    useEffect(() => {
        if (loadedWorkflow.current || !workflows.length) return;
        loadedWorkflow.current = true;
        const preferred = workflows.find((item) => item.is_default) || workflows[0];
        setCurrent(preferred);
        setText(formatWorkflow(preferred.workflow));
        setBindings(preferred.bindings || detectBindings(preferred.workflow));
    }, [workflows]);

    useEffect(() => {
        try {
            window.localStorage.setItem(`${WORKFLOW_STORAGE_KEY}:${projectId}`, text);
        } catch {
            // best effort
        }
    }, [projectId, text]);

    const loadWorkflow = (workflow: SavedWorkflow) => {
        setCurrent(workflow);
        setText(formatWorkflow(workflow.workflow));
        setBindings(workflow.bindings || detectBindings(workflow.workflow));
        setValidation(null);
        setOutputs([]);
    };

    const saveAs = async () => {
        if (!parsed.workflow) return;
        const name = window.prompt('Workflow name', current?.name || 'Comfy workflow');
        if (!name?.trim()) return;
        try {
            const response = await apiFetch(`/projects/${projectId}/comfy/workflows`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    name: name.trim(),
                    description: '',
                    workflow: parsed.workflow,
                    bindings,
                    is_default: workflows.length === 0,
                }),
            });
            const data = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
            setCurrent(data);
            await refreshWorkflows();
            addToast(`Saved workflow “${data.name}”`, 'success');
        } catch (error) {
            addToast(error instanceof Error ? error.message : 'Could not save workflow', 'error');
        }
    };

    const updateWorkflow = async () => {
        if (!current || !parsed.workflow) return;
        try {
            const response = await apiFetch(`/projects/${projectId}/comfy/workflows/${current.id}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    name: current.name,
                    description: current.description,
                    workflow: parsed.workflow,
                    bindings,
                    is_default: current.is_default,
                    expected_revision: current.revision,
                }),
            });
            const data = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(data.detail?.message || data.detail || `HTTP ${response.status}`);
            setCurrent(data);
            await refreshWorkflows();
            addToast('Workflow updated', 'success');
        } catch (error) {
            addToast(error instanceof Error ? error.message : 'Could not update workflow', 'error');
            await refreshWorkflows();
        }
    };

    const renameWorkflow = async () => {
        if (!current || !parsed.workflow) return;
        const name = window.prompt('Rename workflow', current.name);
        if (!name?.trim() || name.trim() === current.name) return;
        try {
            const response = await apiFetch(`/projects/${projectId}/comfy/workflows/${current.id}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    name: name.trim(),
                    description: current.description,
                    workflow: parsed.workflow,
                    bindings,
                    is_default: current.is_default,
                    expected_revision: current.revision,
                }),
            });
            const data = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(data.detail?.message || data.detail || `HTTP ${response.status}`);
            setCurrent(data);
            await refreshWorkflows();
            addToast('Workflow renamed', 'success');
        } catch (error) {
            addToast(error instanceof Error ? error.message : 'Rename failed', 'error');
        }
    };

    const deleteWorkflow = async () => {
        if (!current || !window.confirm(`Delete workflow “${current.name}”?`)) return;
        await apiFetch(`/projects/${projectId}/comfy/workflows/${current.id}`, { method: 'DELETE' });
        setCurrent(null);
        setOutputs([]);
        const rows = await refreshWorkflows();
        const next = rows.find((row) => row.is_default) || rows[0] || null;
        if (next) loadWorkflow(next);
        addToast('Workflow deleted', 'info');
    };

    const makeDefault = async () => {
        if (!current) return;
        const response = await apiFetch(`/projects/${projectId}/comfy/workflows/${current.id}/default`, { method: 'POST' });
        if (response.ok) {
            await refreshWorkflows();
            addToast('Default workflow set', 'success');
        }
    };

    const validate = async (live = true): Promise<Validation | null> => {
        setBusy('validate');
        try {
            const workflow = parseWorkflowText(text);
            const response = await apiFetch('/comfy/validate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ workflow, live }),
            });
            const data = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
            const result = data as Validation;
            setValidation(result);
            if (result.errors?.length) {
                addToast(`${result.errors.length} workflow error(s)`, 'error');
            } else {
                addToast(
                    result.warnings.length ? result.warnings[0] : `${result.nodes || result.node_count || 0} nodes · ${result.output_nodes.length} output(s)`,
                    result.warnings.length ? 'info' : 'success',
                );
            }
            return result;
        } catch (error) {
            setValidation(null);
            addToast(error instanceof Error ? error.message : 'Workflow validation failed', 'error');
            return null;
        } finally {
            setBusy(null);
        }
    };

    const run = async () => {
        if (!shot) {
            addToast('Select a shot before running a ComfyUI workflow', 'error');
            return;
        }
        if (currentJob) {
            addToast('A ComfyUI workflow is already running for this shot', 'info');
            return;
        }
        setBusy('run');
        try {
            const workflow = parseWorkflowText(text);
            const live = await validate(true);
            if (!live || !live.ok) {
                if (live?.errors?.length) addToast(live.errors[0].message, 'error');
                return;
            }
            const resolved = applyBindings(workflow, bindings, shot);
            const response = await apiFetch('/comfy/run', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    project_id: projectId,
                    shot_id: shotId || undefined,
                    workflow: resolved,
                    bindings,
                    workflow_id: current?.id,
                    workflow_name: current?.name,
                    prompt: shot.prompt,
                    estimate_seconds: estimateSeconds,
                }),
            });
            const data = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
            setJobId(data.job_id as string);
            setOutputs([]);
            setValidation(data.validation || live);
            registerStudioJob(data.job_id, { shotId, projectId, kind: 'comfy' });
            useStudioJobs.getState().upsertActive({
                job_id: data.job_id,
                type: 'comfy',
                kind: 'comfy',
                shot_id: shotId || null,
                status: 'queued',
                stage: 'queued',
                progress: 0,
                status_message: 'Queued ComfyUI workflow…',
            });
            if (shotId) {
                useStudioDocumentStore.getState().updateShot(shotId, {
                    isGenerating: true,
                    status: 'generating',
                    progress: 0,
                    stage: 'queued',
                    statusMessage: 'Queued ComfyUI workflow…',
                    lastJobId: data.job_id,
                });
            }
            addToast('ComfyUI workflow queued', 'success');
            void refreshJobs();
        } catch (error) {
            addToast(error instanceof Error ? error.message : 'ComfyUI run failed', 'error');
        } finally {
            setBusy(null);
        }
    };

    const runBatch = async () => {
        if (!batchShots.length) return;
        if (!parsed.workflow) return;
        setBatchBusy(true);
        try {
            const workflow = parseWorkflowText(text);
            let queued = 0;
            for (const target of project.shots.filter((item) => batchShots.includes(item.id) && item.trackIndex !== 2)) {
                const response = await apiFetch('/comfy/run', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        project_id: projectId,
                        shot_id: target.id,
                        workflow: applyBindings(workflow, bindings, target),
                        bindings,
                        workflow_id: current?.id,
                        workflow_name: current?.name,
                        prompt: target.prompt,
                        estimate_seconds: estimateSeconds,
                    }),
                });
                const data = await response.json().catch(() => ({}));
                if (!response.ok) {
                    addToast(`${target.prompt || target.id}: ${data.detail?.message || data.detail || 'queue failed'}`, 'error');
                    continue;
                }
                queued += 1;
                registerStudioJob(data.job_id, { shotId: target.id, projectId, kind: 'comfy' });
                useStudioJobs.getState().upsertActive({
                    job_id: data.job_id, type: 'comfy', kind: 'comfy', shot_id: target.id,
                    status: 'queued', stage: 'queued', progress: 0,
                    status_message: 'Queued ComfyUI workflow…',
                });
                useStudioDocumentStore.getState().updateShot(target.id, {
                    isGenerating: true, status: 'generating', progress: 0,
                    stage: 'queued', statusMessage: 'Queued ComfyUI workflow…', lastJobId: data.job_id,
                });
            }
            if (queued) addToast(`${queued} ComfyUI workflow(s) queued`, 'success');
            setBatchShots([]);
            void refreshJobs();
        } catch (error) {
            addToast(error instanceof Error ? error.message : 'Batch queue failed', 'error');
        } finally {
            setBatchBusy(false);
        }
    };

    const exportPack = async () => {
        const rows = workflows.length ? workflows : current ? [current] : [];
        if (!rows.length) {
            addToast('Save a workflow before exporting a pack', 'info');
            return;
        }
        await saveTextFile(
            `comfy-pack-${projectId}.json`,
            JSON.stringify({
                name: `${project.name || 'Project'} Comfy pack`,
                workflows: rows.map((row) => ({
                    name: row.name, description: row.description,
                    workflow: row.workflow, bindings: row.bindings, is_default: row.is_default,
                })),
            }, null, 2),
        );
        addToast(`Exported ${rows.length} workflow(s)`, 'success');
    };

    const importPack = async (file: File) => {
        try {
            const data = JSON.parse(await file.text());
            const rows = Array.isArray(data) ? data : data.workflows;
            if (!Array.isArray(rows) || rows.length === 0) throw new Error('Pack has no workflows');
            let imported = 0;
            for (const [index, row] of rows.entries()) {
                const response = await apiFetch(`/projects/${projectId}/comfy/workflows`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        name: String(row.name || `Imported ${index + 1}`).slice(0, 100),
                        description: String(row.description || ''),
                        workflow: row.workflow || row,
                        bindings: row.bindings || {},
                        is_default: Boolean(row.is_default),
                    }),
                });
                if (response.ok) imported += 1;
            }
            await refreshWorkflows();
            addToast(`Imported ${imported} workflow(s)`, imported ? 'success' : 'error');
        } catch (error) {
            addToast(error instanceof Error ? error.message : 'Pack import failed', 'error');
        }
    };

    const cancel = async () => {
        if (!jobId) return;
        await apiFetch(`/jobs/${jobId}/cancel`, { method: 'POST' }).catch(() => undefined);
        addToast('Cancellation requested', 'info');
        void refreshJobs();
    };

    const importFile = (file: File) => {
        const reader = new FileReader();
        reader.onload = () => {
            setText(String(reader.result || ''));
            setValidation(null);
            setOutputs([]);
            try {
                const parsedFile = parseWorkflowText(String(reader.result || ''));
                setBindings(detectBindings(parsedFile));
            } catch {
                // The parse error is shown by the textarea state.
            }
            addToast(`Loaded ${file.name}`, 'info');
        };
        reader.readAsText(file);
    };

    const copy = async () => {
        try {
            await navigator.clipboard.writeText(text);
            addToast('Workflow JSON copied', 'success');
        } catch {
            addToast('Clipboard unavailable', 'error');
        }
    };

    const applyNodeInput = (nodeId: string, input: string, value: unknown) => {
        if (!parsed.workflow) return;
        const next = JSON.parse(JSON.stringify(parsed.workflow)) as ComfyWorkflow;
        next[nodeId].inputs = next[nodeId].inputs || {};
        next[nodeId].inputs[input] = value;
        updateText(next);
    };

    const uploadImage = async (nodeId: string, url: string) => {
        try {
            const response = await apiFetch('/comfy/upload', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ project_id: projectId, url }),
            });
            const data = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
            applyNodeInput(nodeId, 'image', data.name);
            addToast('Image uploaded to ComfyUI', 'success');
        } catch (error) {
            addToast(error instanceof Error ? error.message : 'ComfyUI upload failed', 'error');
        }
    };

    const attachOutput = (record: Record<string, unknown>) => {
        if (!shotId) return;
        const url = String(record.url || '');
        if (!url) return;
        useStudioDocumentStore.getState().updateShot(shotId, {
            videoUrl: url,
            thumbnailUrl: String(record.thumbnail_url || url),
            status: 'completed',
            isGenerating: false,
            progress: 100,
        });
        void useStudioDocumentStore.getState().saveProject();
        addToast('Output attached to shot', 'success');
    };

    if (!health || health.enabled === false || !health.reachable) return null;

    const scalar = parsed.workflow ? scalarInputs(parsed.workflow) : [];
    const selectedSpec = selectedNode && parsed.workflow ? schemas[parsed.workflow[selectedNode]?.class_type] : undefined;
    const images = assets.filter((asset) => /\.(png|jpe?g|webp|gif|avif)$/i.test(asset.filename));

    return (
        <details className="st-panel" style={{ marginTop: 14 }} data-testid="comfy-workflow">
            <summary className="st-panel-title" style={{ cursor: 'pointer', margin: 0 }}>
                ComfyUI workflow
            </summary>
            <div className="st-button-row" style={{ marginTop: 8, flexWrap: 'wrap' }}>
                <ComfyStatus health={health} />
                {health.compatible === false && (
                    <span className="st-chip" data-tone="warn">
                        Requires ComfyUI {health.min_version || '0.40'}+
                    </span>
                )}
                <span className="st-value" style={{ color: 'var(--st-faint)', fontSize: 11 }}>
                    {catalog ? `${catalog.node_count} node classes · ${Object.values(catalog.templates).flat().length} templates` : 'API-format workflow'}
                </span>
            </div>

            <div className="st-row" style={{ marginTop: 8, alignItems: 'end' }}>
                <label style={{ flex: 2 }}>
                    <span className="st-label">Workflow</span>
                    <select
                        className="st-select"
                        value={current?.id || ''}
                        onChange={(event) => {
                            if (!event.target.value) {
                                setCurrent(null);
                                return;
                            }
                            const chosen = workflows.find((row) => row.id === event.target.value);
                            if (chosen) loadWorkflow(chosen);
                        }}
                    >
                        <option value="">Draft</option>
                        {workflows.map((workflow) => (
                            <option key={workflow.id} value={workflow.id}>
                                {workflow.name}{workflow.is_default ? ' · default' : ''}
                            </option>
                        ))}
                    </select>
                </label>
                <button type="button" className="st-button" data-size="sm" onClick={() => void saveAs()}>
                    Save as…
                </button>
                {current && (
                    <>
                        <button type="button" className="st-button" data-size="sm" onClick={() => void updateWorkflow()}>
                            Update
                        </button>
                        <button type="button" className="st-button" data-size="sm" onClick={() => void renameWorkflow()}>
                            Rename
                        </button>
                        <button type="button" className="st-button" data-size="sm" onClick={() => void makeDefault()}>
                            Make default
                        </button>
                        <button type="button" className="st-button" data-size="sm" data-variant="danger" onClick={() => void deleteWorkflow()}>
                            <Trash2 size={12} /> Delete
                        </button>
                    </>
                )}
            </div>

            <textarea
                className="st-textarea st-mono"
                aria-label="ComfyUI API workflow JSON"
                style={{ minHeight: 190, marginTop: 8, fontSize: 11 }}
                spellCheck={false}
                value={text}
                onChange={(event) => {
                    setText(event.target.value);
                    setValidation(null);
                }}
                onKeyDown={(event) => {
                    if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
                        event.preventDefault();
                        void run();
                    }
                }}
            />
            <div className="st-button-row" style={{ marginTop: 8, flexWrap: 'wrap' }}>
                <input
                    ref={fileInput}
                    type="file"
                    accept=".json,application/json"
                    style={{ display: 'none' }}
                    aria-label="Import ComfyUI workflow JSON"
                    onChange={(event) => {
                        const file = event.target.files?.[0];
                        event.target.value = '';
                        if (file) importFile(file);
                    }}
                />
                <button type="button" className="st-button" data-size="sm" onClick={() => fileInput.current?.click()}>
                    <Upload size={12} /> Import
                </button>
                <button type="button" className="st-button" data-size="sm" onClick={() => void exportPack()}>
                    Export pack
                </button>
                <input
                    ref={packInput}
                    type="file"
                    accept=".json,application/json"
                    style={{ display: 'none' }}
                    aria-label="Import ComfyUI workflow pack"
                    onChange={(event) => {
                        const file = event.target.files?.[0];
                        event.target.value = '';
                        if (file) void importPack(file);
                    }}
                />
                <button type="button" className="st-button" data-size="sm" onClick={() => packInput.current?.click()}>
                    Import pack
                </button>
                <button type="button" className="st-button" data-size="sm" onClick={() => void copy()}>
                    <Copy size={12} /> Copy
                </button>
                <button
                    type="button"
                    className="st-button"
                    data-size="sm"
                    onClick={() => void saveTextFile(`comfy-workflow-${current?.name || 'draft'}.json`, text)}
                >
                    <Download size={12} /> Download
                </button>
                <button
                    type="button"
                    className="st-button"
                    data-size="sm"
                    onClick={() => {
                        try {
                            setText(formatWorkflow(parseWorkflowText(text)));
                        } catch (error) {
                            addToast(error instanceof Error ? error.message : 'Invalid JSON', 'error');
                        }
                    }}
                >
                    Format
                </button>
                <button
                    type="button"
                    className="st-button"
                    data-size="sm"
                    onClick={() => {
                        setText(SMOKE_WORKFLOW);
                        setBindings(detectBindings(parseWorkflowText(SMOKE_WORKFLOW)));
                        setValidation(null);
                    }}
                    title="Model-free EmptyImage → SaveImage graph"
                >
                    Smoke test
                </button>
                <button type="button" className="st-button" data-size="sm" disabled={busy !== null} onClick={() => void validate(true)}>
                    {busy === 'validate' ? <Loader2 size={12} className="animate-spin" /> : <CheckCircle2 size={12} />}
                    Validate
                </button>
                <button
                    type="button"
                    className="st-button"
                    data-size="sm"
                    data-variant="primary"
                    disabled={busy !== null || !shot || Boolean(currentJob) || health.compatible === false}
                    onClick={() => void run()}
                >
                    {busy === 'run' ? <Loader2 size={12} className="animate-spin" /> : <Play size={12} />}
                    Run
                </button>
                {outputs.length > 0 && !currentJob && (
                    <button type="button" className="st-button" data-size="sm" onClick={() => void run()}>
                        <RefreshCw size={12} /> Retry last
                    </button>
                )}
                {jobId && (
                    <button type="button" className="st-button" data-size="sm" data-variant="danger" onClick={() => void cancel()}>
                        Cancel
                    </button>
                )}
                <a className="st-button" data-size="sm" data-variant="ghost" href={apiUrl('/comfy/health')} target="_blank" rel="noreferrer">
                    <ExternalLink size={12} /> Health
                </a>
            </div>
            <div className="st-button-row" style={{ marginTop: 6 }}>
                <span className="st-value" style={{ color: 'var(--st-faint)', fontSize: 11 }}>
                    {text.split('\n').length} lines · {text.length.toLocaleString()} chars · ⌘/Ctrl+Enter run
                </span>
                {current && text !== formatWorkflow(current.workflow) && (
                    <span className="st-chip" data-tone="warn">unsaved changes</span>
                )}
            </div>

            {parsed.error && <div className="st-lens-warnings" role="alert"><TriangleAlert size={11} /> {parsed.error}</div>}
            {validation && (
                <div style={{ marginTop: 8 }}>
                    <div className="st-button-row" style={{ flexWrap: 'wrap' }}>
                        <span className="st-chip">{validation.nodes || validation.node_count || 0} nodes</span>
                        <span className="st-chip">{validation.output_nodes.length} output(s)</span>
                        {validation.width && validation.height && (
                            <span className="st-chip">{validation.width}×{validation.height}</span>
                        )}
                        {validation.live && <span className="st-chip" data-tone="good">live preflight</span>}
                        {validation.error && <span className="st-chip" data-tone="bad">{validation.error}</span>}
                        {validation.warnings.map((warning) => (
                            <span key={warning} className="st-chip" data-tone="warn">{warning}</span>
                        ))}
                    </div>
                    {validation.errors?.map((error) => (
                        <div key={`${error.node_id}-${error.field || ''}-${error.message}`} className="st-lens-warnings" role="alert">
                            <TriangleAlert size={11} />
                            <span>
                                <strong>{error.node_id} · {error.class_type}</strong>
                                {error.field ? ` · ${error.field}` : ''}: {error.message}
                            </span>
                        </div>
                    ))}
                </div>
            )}

            {currentJob && (
                <div style={{ marginTop: 8 }}>
                    <div className="st-value" style={{ fontSize: 11 }}>
                        {currentJob.status_message || currentJob.status} · {currentJob.progress}%
                        {currentJob.position ? ` · #${currentJob.position}` : ''}
                    </div>
                    <progress value={currentJob.progress} max={100} style={{ width: '100%' }} />
                </div>
            )}

            {parsed.workflow && (
                <details style={{ marginTop: 10 }} open>
                    <summary className="st-label" style={{ cursor: 'pointer' }}>Shot bindings</summary>
                    <div className="st-button-row" style={{ marginTop: 6 }}>
                        <button
                            type="button"
                            className="st-button"
                            data-size="sm"
                            onClick={() => setBindings(detectBindings(parsed.workflow!))}
                        >
                            Auto-detect
                        </button>
                        <span className="st-value" style={{ color: 'var(--st-faint)', fontSize: 11 }}>
                            Prompt/seed/size replaced at run time.
                        </span>
                    </div>
                    <div className="st-input-grid" style={{ marginTop: 6 }}>
                        {BINDING_FIELDS.map((field) => {
                            const binding = bindings[field.key];
                            return (
                                <label key={field.key}>
                                    <span className="st-label">{field.label}</span>
                                    <select
                                        className="st-select"
                                        value={bindingPath(binding)}
                                        onChange={(event) => {
                                            const found = findBinding(parsed.workflow!, event.target.value);
                                            setBindings((rows) => ({ ...rows, [field.key]: found || undefined }));
                                        }}
                                    >
                                        <option value="">— unbound —</option>
                                        {scalar
                                            .filter((row) => {
                                                if (field.key === 'prompt' || field.key === 'negative') return typeof row.value === 'string';
                                                if (field.key === 'model') return /name$|ckpt|unet|model/i.test(row.input);
                                                return typeof row.value === 'number' || /seed|width|height|step|cfg|guidance/i.test(row.input);
                                            })
                                            .map((row) => (
                                                <option key={`${row.node}.${row.input}`} value={`${row.node}.${row.input}`}>
                                                    {row.node}.{row.input} · {row.classType}
                                                </option>
                                            ))}
                                    </select>
                                    {field.key === 'model' && binding && (
                                        <select
                                            className="st-select"
                                            value={String(binding.value ?? '')}
                                            onChange={(event) => setBindings((rows) => ({
                                                ...rows,
                                                model: { ...binding, value: event.target.value },
                                            }))}
                                        >
                                            <option value={String(binding.value ?? '')}>{String(binding.value ?? '—')}</option>
                                            {modelOptions.map((model) => (
                                                <option key={model} value={model}>{model}</option>
                                            ))}
                                        </select>
                                    )}
                                </label>
                            );
                        })}
                    </div>
                </details>
            )}

            {parsed.workflow && (
                <details style={{ marginTop: 10 }}>
                    <summary className="st-label" style={{ cursor: 'pointer' }}>Batch across shots</summary>
                    <div className="st-link-list" style={{ marginTop: 6 }}>
                        {project.shots.filter((item) => item.trackIndex !== 2).map((item) => (
                            <label key={item.id} className="st-chip" style={{ textTransform: 'none', letterSpacing: 0 }}>
                                <input
                                    type="checkbox"
                                    checked={batchShots.includes(item.id)}
                                    onChange={(event) => setBatchShots((rows) => (
                                        event.target.checked
                                            ? [...rows, item.id]
                                            : rows.filter((id) => id !== item.id)
                                    ))}
                                />
                                {item.prompt?.slice(0, 60) || item.id}
                            </label>
                        ))}
                        {project.shots.length === 0 && (
                            <span className="st-value" style={{ color: 'var(--st-muted)' }}>No shots in this project.</span>
                        )}
                    </div>
                    <div className="st-button-row" style={{ marginTop: 6 }}>
                        <button
                            type="button"
                            className="st-button"
                            data-size="sm"
                            disabled={batchBusy || batchShots.length === 0}
                            onClick={() => void runBatch()}
                        >
                            {batchBusy ? <Loader2 size={12} className="animate-spin" /> : <Play size={12} />}
                            Queue {batchShots.length || ''} batch
                        </button>
                    </div>
                </details>
            )}

            {parsed.workflow && (
                <details style={{ marginTop: 10 }}>
                    <summary className="st-label" style={{ cursor: 'pointer' }}>Node inputs</summary>
                    <div className="st-row" style={{ marginTop: 6, alignItems: 'end' }}>
                        <label style={{ flex: 1 }}>
                            <span className="st-label">Node</span>
                            <select
                                className="st-select"
                                value={selectedNode || ''}
                                onChange={(event) => setSelectedNode(event.target.value)}
                            >
                                {Object.entries(parsed.workflow).map(([nodeId, node]: [string, any]) => (
                                    <option key={nodeId} value={nodeId}>
                                        {nodeId} · {node.class_type}
                                    </option>
                                ))}
                            </select>
                        </label>
                        <label style={{ flex: 1 }}>
                            <span className="st-label">Estimate (s)</span>
                            <input
                                className="st-input"
                                type="number"
                                min={1}
                                value={estimateSeconds}
                                onChange={(event) => setEstimateSeconds(Number(event.target.value))}
                            />
                        </label>
                    </div>
                    {selectedNode && (
                        <NodeInputs
                            workflow={parsed.workflow}
                            nodeId={selectedNode}
                            spec={selectedSpec}
                            onChange={applyNodeInput}
                        />
                    )}
                    {selectedNode && parsed.workflow[selectedNode]?.class_type === 'LoadImage' && (
                        <div style={{ marginTop: 8 }}>
                            <span className="st-label">Project image</span>
                            <div className="st-button-row" style={{ marginTop: 4 }}>
                                <select
                                    className="st-select"
                                    value=""
                                    onChange={(event) => {
                                        if (event.target.value) void uploadImage(selectedNode, event.target.value);
                                    }}
                                >
                                    <option value="">Choose an image…</option>
                                    {images.map((asset) => (
                                        <option key={asset.id} value={asset.url}>{asset.filename}</option>
                                    ))}
                                </select>
                            </div>
                        </div>
                    )}
                </details>
            )}

            {outputs.length > 0 && (
                <div style={{ marginTop: 10 }}>
                    <span className="st-label">Latest outputs · {outputs.length}</span>
                    <div className="st-grid" style={{ marginTop: 6 }}>
                        {outputs.map((record, index) => {
                            const kind = outputKind(record as any);
                            const url = String(record.url || '');
                            const thumb = String(record.thumbnail_url || record.url || '');
                            return (
                                <article className="st-take" key={`${url}-${index}`}>
                                    <button
                                        type="button"
                                        className="st-take-media st-media-open"
                                        aria-label={`Preview ${record.source_filename || index + 1}`}
                                        onClick={() => openMediaPreview({
                                            url: getAssetUrl(url) || url,
                                            title: String(record.source_filename || 'ComfyUI output'),
                                            subtitle: `${kind} · node ${record.node_id || '—'}`,
                                            kind: kind === 'text' ? undefined : kind,
                                        })}
                                    >
                                        {(kind === 'image' || kind === 'video') ? (
                                            <MediaThumb src={url} poster={thumb} alt={String(record.source_filename || 'output')} />
                                        ) : (
                                            <div className="st-stage-empty">{kind}</div>
                                        )}
                                    </button>
                                    <div className="st-take-body">
                                        <strong>{String(record.source_filename || `output ${index + 1}`)}</strong>
                                        <div>{kind} · node {String(record.node_id || '—')}</div>
                                        <button type="button" className="st-button" data-size="sm" onClick={() => attachOutput(record)}>
                                            Use as shot media
                                        </button>
                                    </div>
                                </article>
                            );
                        })}
                    </div>
                </div>
            )}
        </details>
    );
}
