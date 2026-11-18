import { useCallback, useEffect, useRef, useState } from 'react';
import {
    CheckCircle2, ExternalLink, Loader2, Play, PowerOff, RefreshCw, TriangleAlert, Upload, Workflow,
} from 'lucide-react';
import { apiFetch, apiUrl } from '../../config';
import { useStudioDocumentStore } from '../state/documentStore';
import { useStudioJobs } from '../state/jobStore';
import { registerStudioJob } from '../events/jobBridge';

export interface ComfyHealth {
    enabled?: boolean;
    reachable: boolean;
    url?: string;
    version?: string | null;
    queue_running?: number | null;
    queue_pending?: number | null;
    ram_free_gb?: number | null;
    ram_total_gb?: number | null;
    error?: string;
}

interface Validation {
    nodes: number;
    output_nodes: string[];
    width?: number | null;
    height?: number | null;
    warnings: string[];
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

function ComfyStatus({ health }: { health: ComfyHealth | null }) {
    if (!health) {
        return <span className="st-status-pill" data-tone="warn"><Loader2 size={10} className="animate-spin" /> checking</span>;
    }
    if (health.enabled === false) {
        return (
            <span className="st-status-pill" data-tone="warn" title="MILIMO_COMFY_ENABLED=0">
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
    const addToast = useStudioDocumentStore((state) => state.addToast);
    const [freeing, setFreeing] = useState(false);

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
                    Remove <code>MILIMO_COMFY_ENABLED=0</code> to enable the workflow panel.
                </p>
            )}
            {health && health.enabled !== false && !health.reachable && (
                <p className="st-value" style={{ marginTop: 6, color: 'var(--st-muted)', fontSize: 11 }}>
                    {health.error || 'ComfyUI did not answer.'} Start Comfy Desktop or set <code>MILIMO_COMFY_URL</code>.
                </p>
            )}
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
                it is charged to the same queue, and <code>/free</code> releases its cache. Comfy Desktop and
                SAM defaults to port 8002, so both can run; set <code>MILIMO_COMFY_URL</code> or <code>MILIMO_SAM_PORT</code> to override.
            </p>
        </div>
    );
}

export function ComfyWorkflowPanel({ projectId, shotId }: { projectId: string; shotId?: string | null }) {
    const addToast = useStudioDocumentStore((state) => state.addToast);
    const { health, refresh: refreshHealth } = useComfyHealth();
    const [text, setText] = useState(() => {
        try {
            return window.localStorage.getItem(WORKFLOW_STORAGE_KEY) || SMOKE_WORKFLOW;
        } catch {
            return SMOKE_WORKFLOW;
        }
    });
    const [validation, setValidation] = useState<Validation | null>(null);
    const [busy, setBusy] = useState<'validate' | 'run' | null>(null);
    const [jobId, setJobId] = useState<string | null>(null);
    const fileInput = useRef<HTMLInputElement>(null);

    const parse = (): unknown => {
        try {
            return JSON.parse(text);
        } catch (error) {
            throw new Error(error instanceof Error ? `Invalid JSON: ${error.message}` : 'Invalid JSON');
        }
    };

    const validate = async (): Promise<Validation | null> => {
        setBusy('validate');
        try {
            const workflow = parse();
            const response = await apiFetch('/comfy/validate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ workflow }),
            });
            const data = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
            const result = data as Validation;
            setValidation(result);
            addToast(
                result.warnings.length ? result.warnings[0] : `${result.nodes} nodes · ${result.output_nodes.length} output node(s)`,
                result.warnings.length ? 'info' : 'success',
            );
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
        setBusy('run');
        try {
            const workflow = parse();
            const response = await apiFetch('/comfy/run', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    project_id: projectId,
                    shot_id: shotId || undefined,
                    workflow,
                }),
            });
            const data = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
            try {
                window.localStorage.setItem(WORKFLOW_STORAGE_KEY, text);
            } catch {
                // Storage is best-effort; the workflow still ran.
            }
            setJobId(data.job_id as string);
            setValidation(data.validation || null);
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
            if (!health?.reachable) void refreshHealth();
        } catch (error) {
            addToast(error instanceof Error ? error.message : 'ComfyUI run failed', 'error');
        } finally {
            setBusy(null);
        }
    };

    const cancel = async () => {
        if (!jobId) return;
        await apiFetch(`/jobs/${jobId}/cancel`, { method: 'POST' }).catch(() => undefined);
        addToast('Cancellation requested', 'info');
    };

    const importFile = (file: File) => {
        const reader = new FileReader();
        reader.onload = () => {
            setText(String(reader.result || ''));
            setValidation(null);
            addToast(`Loaded ${file.name}`, 'info');
        };
        reader.readAsText(file);
    };

    // The bridge is optional: with MILIMO_COMFY_ENABLED=0 the surface disappears,
    // and while ComfyUI is offline the Generate rail stays clean (Setup still
    // documents the optional engine).
    if (!health || health.enabled === false || !health.reachable) return null;

    return (
        <details className="st-panel" style={{ marginTop: 14 }} data-testid="comfy-workflow">
            <summary className="st-panel-title" style={{ cursor: 'pointer', margin: 0 }}>
                ComfyUI workflow
            </summary>
            <div className="st-button-row" style={{ marginTop: 8 }}>
                <ComfyStatus health={health} />
                <span className="st-value" style={{ color: 'var(--st-faint)', fontSize: 11 }}>
                    Workflow → Save (API Format), then paste or import here.
                </span>
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
            />
            <div className="st-button-row" style={{ marginTop: 8 }}>
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
                    <Upload size={12} /> Import JSON
                </button>
                <button
                    type="button"
                    className="st-button"
                    data-size="sm"
                    onClick={() => {
                        setText(SMOKE_WORKFLOW);
                        setValidation(null);
                    }}
                    title="Model-free EmptyImage → SaveImage graph; proves the bridge end to end"
                >
                    Smoke test
                </button>
                <button type="button" className="st-button" data-size="sm" disabled={busy !== null} onClick={() => void validate()}>
                    {busy === 'validate' ? <Loader2 size={12} className="animate-spin" /> : <CheckCircle2 size={12} />}
                    Validate
                </button>
                <button
                    type="button"
                    className="st-button"
                    data-size="sm"
                    data-variant="primary"
                    disabled={busy !== null}
                    onClick={() => void run()}
                >
                    {busy === 'run' ? <Loader2 size={12} className="animate-spin" /> : <Play size={12} />}
                    Run workflow
                </button>
                {jobId && (
                    <button type="button" className="st-button" data-size="sm" data-variant="danger" onClick={() => void cancel()}>
                        Cancel
                    </button>
                )}
                <a className="st-button" data-size="sm" data-variant="ghost" href={apiUrl('/comfy/health')} target="_blank" rel="noreferrer">
                    <ExternalLink size={12} /> Health
                </a>
            </div>
            {validation && (
                <div className="st-button-row" style={{ marginTop: 8 }}>
                    <span className="st-chip">{validation.nodes} nodes</span>
                    <span className="st-chip">{validation.output_nodes.length} output(s)</span>
                    {validation.width && validation.height && (
                        <span className="st-chip">{validation.width}×{validation.height}</span>
                    )}
                    {validation.warnings.map((warning) => (
                        <span key={warning} className="st-chip" data-tone="warn">{warning}</span>
                    ))}
                </div>
            )}
            <p className="st-value" style={{ marginTop: 8, color: 'var(--st-faint)', fontSize: 11 }}>
                Outputs are copied into this project and attached to the selected shot; progress streams in the
                activity center. The quick smoke graph has no model dependency.
            </p>
        </details>
    );
}
