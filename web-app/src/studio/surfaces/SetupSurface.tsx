import { useMemo, useState } from 'react';
import {
    CheckCircle2, Cpu, HardDrive, Loader2, RefreshCw, ShieldCheck, TriangleAlert,
} from 'lucide-react';
import {
    useStudioHealth, useStudioIntegrity, useStudioLoras, useStudioMetrics, useStudioModels, useStudioProject,
} from '../data/useStudioData';
import { useStudioDocumentStore } from '../state/documentStore';
import { apiUrl } from '../../config';

function formatBytes(bytes?: number | null): string {
    if (!bytes || bytes <= 0) return '—';
    const gb = bytes / 1e9;
    return gb >= 1000 ? `${(gb / 1000).toFixed(1)} TB` : `${gb.toFixed(1)} GB`;
}

export function SetupSurface() {
    const { health, error } = useStudioHealth();
    const { models, pipelines, loading, refresh, setActive } = useStudioModels();
    const { loras, refresh: refreshLoras, setStrength } = useStudioLoras('video');
    const metrics = useStudioMetrics(7);
    const project = useStudioProject();
    const { report: integrity, loading: integrityLoading, scan: scanIntegrity } = useStudioIntegrity(project.id);
    const addToast = useStudioDocumentStore((state) => state.addToast);

    const downloadModel = async (modelId: string, name: string) => {
        setDownloading(modelId);
        try {
            const response = await fetch(apiUrl(`/models/${modelId}/download`), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ force: false }),
            });
            const data = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(data.detail || 'Download failed');
            addToast(data.message || `Queued ${name}`, 'success');
            await refresh();
        } catch (exc) {
            addToast(exc instanceof Error ? exc.message : 'Download failed', 'error');
        } finally {
            setDownloading(null);
        }
    };
    const repairProject = async () => {
        setRepairing(true);
        try {
            const response = await fetch(apiUrl(`/projects/${project.id}/repair`), { method: 'POST' });
            const data = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(data.detail || 'Repair failed');
            const total = Object.values(data.counts || {}).reduce(
                (sum: number, value) => sum + Number(value || 0),
                0,
            );
            addToast(`Repaired ${total} issue(s)`, 'success');
            await scanIntegrity();
        } catch (exc) {
            addToast(exc instanceof Error ? exc.message : 'Repair failed', 'error');
        } finally {
            setRepairing(false);
        }
    };

    const toggleQueue = async (paused: boolean) => {
        setQueueBusy(true);
        try {
            const response = await fetch(apiUrl(`/jobs/queue/${paused ? 'pause' : 'resume'}`), { method: 'POST' });
            const data = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(data.detail || 'Queue update failed');
            addToast(paused ? 'GPU queue paused after the current job' : 'GPU queue resumed', 'info');
        } catch (exc) {
            addToast(exc instanceof Error ? exc.message : 'Queue update failed', 'error');
        } finally {
            setQueueBusy(false);
        }
    };

    const [filter, setFilter] = useState<'all' | 'video' | 'image' | 'segmentation' | 'depth'>('all');
    const [kind, setKind] = useState<'base' | 'lora'>('base');
    const [busy, setBusy] = useState<string | null>(null);
    const [downloading, setDownloading] = useState<string | null>(null);
    const [queueBusy, setQueueBusy] = useState(false);
    const [repairing, setRepairing] = useState(false);

    const visible = useMemo(() => {
        const matching = models.filter((model) => (
            kind === 'base' ? model.type === 'base' : model.type !== 'base'
        ));
        return filter === 'all' ? matching : matching.filter((model) => model.pipeline === filter);
    }, [filter, kind, models]);

    const activeCount = models.filter((model) => model.status === 'active').length;

    return (
        <div className="st-page">
            <div className="st-page-head">
                <div>
                    <h2 className="st-page-title">Setup</h2>
                    <p className="st-page-subtitle">
                        Real machine state: device, memory slots, pipeline readiness, and disk.
                        Activation loads a model into the one heavy GPU slot; it is never implicit.
                    </p>
                </div>
                <div className="st-button-row">
                    <span className="st-status-pill" data-tone={health ? 'good' : 'bad'}>
                        <Cpu size={11} /> {health ? health.device.toUpperCase() : 'offline'}
                    </span>
                    <button type="button" className="st-button" data-size="sm" onClick={() => void refresh()}>
                        <RefreshCw size={12} className={loading ? 'animate-spin' : ''} /> Rescan models
                    </button>
                </div>
            </div>

            {error && (
                <div className="st-panel" style={{ marginBottom: 12, borderColor: 'rgba(255,93,108,.3)' }}>
                    <strong style={{ color: 'var(--st-danger)' }}>Backend offline</strong>
                    <div className="st-value" style={{ color: 'var(--st-muted)', marginTop: 4 }}>{error}</div>
                </div>
            )}

            <div className="st-grid" style={{ marginBottom: 14 }}>
                <div className="st-panel">
                    <h3 className="st-panel-title">Memory slots</h3>
                    <dl className="st-meta-grid">
                        <dt>Device</dt><dd>{health?.device || '—'}</dd>
                        <dt>Active</dt><dd>{health?.slots?.active_slots?.length ? health.slots.active_slots.join(', ') : 'none'}</dd>
                        <dt>GPU</dt><dd>{health?.gpu?.busy ? 'busy' : 'free'}</dd>
                        <dt>Queue</dt><dd>{health?.gpu?.queued ?? 0}</dd>
                        <dt>Auth</dt><dd>{health?.auth_required ? 'token required' : 'loopback only'}</dd>
                        <dt>Dispatch</dt><dd>{health?.queue_paused ? 'paused' : 'running'}</dd>
                    </dl>
                    <div className="st-button-row" style={{ marginTop: 8 }}>
                        <button
                            type="button"
                            className="st-button"
                            data-size="sm"
                            disabled={queueBusy}
                            onClick={() => void toggleQueue(!(health?.queue_paused ?? false))}
                        >
                            {health?.queue_paused ? 'Resume queue' : 'Pause queue'}
                        </button>
                    </div>
                    <p className="st-value" style={{ marginTop: 8, color: 'var(--st-faint)', fontSize: 10 }}>
                        Only one heavy model runs at a time; conflicting pipelines wait in the durable queue.
                        Pausing stops new dispatch after the current job finishes.
                    </p>
                </div>
                <div className="st-panel">
                    <h3 className="st-panel-title">Disk</h3>
                    <div className="st-value" style={{ fontSize: 20, fontWeight: 700 }}>
                        <HardDrive size={16} style={{ verticalAlign: -2, marginRight: 6 }} />
                        {formatBytes(health?.disk_free_bytes)} free
                    </div>
                    <p className="st-value" style={{ marginTop: 6, color: 'var(--st-muted)', fontSize: 10 }}>
                        Checkpoints are large. Rescan before downloading; the library lists real downloaded state.
                    </p>
                </div>
                <div className="st-panel">
                    <h3 className="st-panel-title">Pipelines</h3>
                    <div className="st-button-row">
                        {Object.entries(pipelines).map(([name, readiness]) => (
                            <span
                                key={name}
                                className="st-status-pill"
                                data-tone={readiness.ready ? 'good' : 'warn'}
                                title={readiness.missing?.join(', ') || 'ready'}
                            >
                                {readiness.ready ? <CheckCircle2 size={10} /> : <TriangleAlert size={10} />}
                                {name} {readiness.ready ? 'ready' : `${readiness.missing?.length || 0} missing`}
                            </span>
                        ))}
                        {Object.keys(pipelines).length === 0 && <span className="st-value">No readiness data.</span>}
                    </div>
                    <p className="st-value" style={{ marginTop: 8, color: 'var(--st-faint)', fontSize: 10 }}>
                        {activeCount} base/LoRA active. Activate a downloaded base below.
                    </p>
                </div>
            </div>

            <div className="st-panel" style={{ marginBottom: 14 }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 10 }}>
                    <h3 className="st-panel-title" style={{ margin: 0 }}>Project integrity</h3>
                    <div className="st-button-row">
                        <button type="button" className="st-button" data-size="sm" disabled={integrityLoading} onClick={() => void scanIntegrity()}>
                            {integrityLoading ? <Loader2 size={12} className="animate-spin" /> : null}
                            {integrityLoading ? 'Scanning…' : 'Scan project'}
                        </button>
                        {integrity && integrity.issues.length > 0 && (
                            <button
                                type="button"
                                className="st-button"
                                data-size="sm"
                                data-variant="primary"
                                disabled={repairing}
                                onClick={() => void repairProject()}
                            >
                                {repairing ? <Loader2 size={12} className="animate-spin" /> : null}
                                {repairing ? 'Repairing…' : 'Repair safe issues'}
                            </button>
                        )}
                    </div>
                </div>
                {!integrity ? (
                    <p className="st-value" style={{ color: 'var(--st-muted)', marginTop: 8 }}>
                        Scan for missing media, orphan shots/scenes, orphan board links, and bad caption ranges.
                    </p>
                ) : (
                    <>
                        <div className="st-button-row" style={{ marginTop: 8 }}>
                            <span className="st-chip" data-tone={integrity.ok ? 'good' : 'bad'}>
                                {integrity.ok ? 'no blocking issues' : `${integrity.counts.errors} error(s)`}
                            </span>
                            <span className="st-chip">{integrity.counts.shots} shots</span>
                            <span className="st-chip">{integrity.counts.scenes} scenes</span>
                            <span className="st-chip">{integrity.counts.issues} issue(s)</span>
                        </div>
                        {integrity.issues.length > 0 && (
                            <div className="st-link-list" style={{ marginTop: 8 }}>
                                {integrity.issues.slice(0, 20).map((issue, index) => (
                                    <div className="st-link-row" key={`${issue.kind}-${index}`}>
                                        <span className="st-chip" data-tone={issue.severity === 'error' ? 'bad' : undefined}>
                                            {issue.kind}
                                        </span>
                                        <span style={{ flex: 1 }}>{issue.message}</span>
                                    </div>
                                ))}
                            </div>
                        )}
                    </>
                )}
            </div>

            <div className="st-panel" style={{ marginBottom: 14 }}>
                <h3 className="st-panel-title">Error budget · last {metrics?.window_days ?? 7} days</h3>
                {!metrics ? (
                    <span className="st-value" style={{ color: 'var(--st-muted)' }}>No metrics yet.</span>
                ) : (
                    <>
                        <div className="st-button-row">
                            <span className="st-chip">{metrics.total} jobs</span>
                            {Object.entries(metrics.by_status).map(([status, count]) => (
                                <span className="st-chip" key={status}>{status} {count}</span>
                            ))}
                            {(metrics.recent_failures || []).length > 0 && (
                                <span className="st-chip" style={{ color: 'var(--st-danger)' }}>
                                    {metrics.recent_failures.length} recent failure(s)
                                </span>
                            )}
                        </div>
                        {(metrics.recent_failures || []).length > 0 && (
                            <div className="st-link-list" style={{ marginTop: 8 }}>
                                {metrics.recent_failures.slice(0, 5).map((failure) => (
                                    <div className="st-link-row" key={failure.job_id}>
                                        <span className="st-mono">{failure.job_id.slice(0, 10)}</span>
                                        <span className="st-chip">{failure.type}</span>
                                        <span style={{ flex: 1, color: 'var(--st-danger)' }}>{failure.error || 'failed'}</span>
                                    </div>
                                ))}
                            </div>
                        )}
                    </>
                )}
            </div>

            <div className="st-panel">
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 10, marginBottom: 10 }}>
                    <h3 className="st-panel-title" style={{ margin: 0 }}>Models</h3>
                    <div className="st-button-row">
                        <button
                            type="button"
                            className="st-button"
                            data-size="sm"
                            data-variant={kind === 'base' ? 'primary' : 'ghost'}
                            onClick={() => setKind('base')}
                        >
                            Base
                        </button>
                        <button
                            type="button"
                            className="st-button"
                            data-size="sm"
                            data-variant={kind === 'lora' ? 'primary' : 'ghost'}
                            onClick={() => setKind('lora')}
                        >
                            LoRA
                        </button>
                        {(['all', 'video', 'image', 'segmentation', 'depth'] as const).map((item) => (
                            <button
                                key={item}
                                type="button"
                                className="st-button"
                                data-size="sm"
                                data-variant={filter === item ? 'primary' : 'ghost'}
                                onClick={() => setFilter(item)}
                            >
                                {item}
                            </button>
                        ))}
                    </div>
                </div>

                <table className="st-table">
                    <thead>
                        <tr>
                            <th scope="col">Model</th>
                            <th scope="col">Pipeline</th>
                            <th scope="col">Status</th>
                            <th scope="col">Size</th>
                            <th scope="col">VRAM est.</th>
                            <th scope="col">Action</th>
                        </tr>
                    </thead>
                    <tbody>
                        {visible.map((model) => (
                            <tr key={model.id}>
                                <td>
                                    <strong>{model.name}</strong>
                                    <div style={{ color: 'var(--st-faint)', fontSize: 9 }}>{model.id}{model.required ? ' · required' : ''}</div>
                                </td>
                                <td>{model.pipeline} · {model.family}</td>
                                <td>
                                    <span
                                        className="st-status-pill"
                                        data-tone={model.status === 'active' ? 'good' : model.status === 'downloaded' ? 'warn' : undefined}
                                    >
                                        {model.status}
                                    </span>
                                </td>
                                <td className="st-mono">{formatBytes(model.size_bytes)}</td>
                                <td className="st-mono">{model.vram_estimate_gb ? `${model.vram_estimate_gb} GB` : '—'}</td>
                                <td>
                                    {model.status === 'downloaded' && model.family === 'depth' && (
                                        <span style={{ color: 'var(--st-success)', fontSize: 9 }}>ready · loads on demand</span>
                                    )}
                                    {model.status === 'downloaded' && model.family !== 'depth' && (
                                        <button
                                            type="button"
                                            className="st-button"
                                            data-size="sm"
                                            disabled={busy === model.id}
                                            onClick={async () => {
                                                setBusy(model.id);
                                                try { if (await setActive(model, true)) await refreshLoras(); } finally { setBusy(null); }
                                            }}
                                        >
                                            {busy === model.id ? <Loader2 size={12} className="animate-spin" /> : <ShieldCheck size={12} />}
                                            {busy === model.id ? 'Loading…' : 'Activate'}
                                        </button>
                                    )}
                                    {model.status === 'active' && (
                                        <>
                                            <button
                                                type="button"
                                                className="st-button"
                                                data-size="sm"
                                                data-variant="danger"
                                                disabled={busy === model.id}
                                                onClick={async () => {
                                                    setBusy(model.id);
                                                    try { if (await setActive(model, false)) await refreshLoras(); } finally { setBusy(null); }
                                                }}
                                            >
                                                {busy === model.id ? <Loader2 size={12} className="animate-spin" /> : null}
                                                Unload
                                            </button>
                                            {kind === 'lora' && loras.find((lora) => lora.model_id === model.id) && (
                                                <label style={{ display: 'block', marginTop: 5, color: 'var(--st-muted)', fontSize: 9 }}>
                                                    Strength · {Math.round((loras.find((lora) => lora.model_id === model.id)?.strength ?? 1) * 100)}%
                                                    <input
                                                        type="range"
                                                        min={0}
                                                        max={200}
                                                        defaultValue={Math.round((loras.find((lora) => lora.model_id === model.id)?.strength ?? 1) * 100)}
                                                        style={{ width: '100%', accentColor: 'var(--st-accent)' }}
                                                        aria-label={`${model.name} strength`}
                                                        onPointerUp={(event) => void setStrength(
                                                            model.id,
                                                            Number(event.currentTarget.value) / 100,
                                                        )}
                                                    />
                                                </label>
                                            )}
                                        </>
                                    )}
                                    {model.status !== 'downloaded' && model.status !== 'active' && model.status !== 'downloading' && (
                                        model.status === 'incompatible' ? (
                                            <span style={{ color: 'var(--st-faint)', fontSize: 9 }}>not for this device</span>
                                        ) : (
                                            <button
                                                type="button"
                                                className="st-button"
                                                data-size="sm"
                                                disabled={downloading === model.id}
                                                onClick={() => void downloadModel(model.id, model.name)}
                                            >
                                                {downloading === model.id ? <Loader2 size={12} className="animate-spin" /> : null}
                                                {downloading === model.id ? 'Queueing…' : 'Download'}
                                            </button>
                                        )
                                    )}
                                    {model.status === 'downloading' && (
                                        <span style={{ color: 'var(--st-warning)', fontSize: 9 }}>downloading…</span>
                                    )}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
                {visible.length === 0 && !loading && (
                    <div className="st-empty" style={{ minHeight: 120 }}>
                        <strong>No models reported</strong>
                        <span>Check the backend registry and disk scan.</span>
                    </div>
                )}
            </div>
        </div>
    );
}
