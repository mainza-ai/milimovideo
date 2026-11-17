import { useMemo, useState } from 'react';
import {
    CheckCircle2, Cpu, HardDrive, Loader2, RefreshCw, ShieldCheck, TriangleAlert,
} from 'lucide-react';
import { useStudioHealth, useStudioModels } from '../data/useStudioData';

function formatBytes(bytes?: number | null): string {
    if (!bytes || bytes <= 0) return '—';
    const gb = bytes / 1e9;
    return gb >= 1000 ? `${(gb / 1000).toFixed(1)} TB` : `${gb.toFixed(1)} GB`;
}

export function SetupSurface() {
    const { health, error } = useStudioHealth();
    const { models, pipelines, loading, refresh, setActive } = useStudioModels();
    const [filter, setFilter] = useState<'all' | 'video' | 'image' | 'segmentation'>('all');
    const [busy, setBusy] = useState<string | null>(null);

    const visible = useMemo(() => {
        const bases = models.filter((model) => model.type === 'base');
        return filter === 'all' ? bases : bases.filter((model) => model.pipeline === filter);
    }, [filter, models]);

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
                    <div className="st-meta-grid">
                        <dt>Device</dt><dd>{health?.device || '—'}</dd>
                        <dt>Active</dt><dd>{health?.slots?.active_slots?.length ? health.slots.active_slots.join(', ') : 'none'}</dd>
                        <dt>GPU</dt><dd>{health?.gpu?.busy ? 'busy' : 'free'}</dd>
                        <dt>Queue</dt><dd>{health?.gpu?.queued ?? 0}</dd>
                        <dt>Auth</dt><dd>{health?.auth_required ? 'token required' : 'loopback only'}</dd>
                    </div>
                    <p className="st-value" style={{ marginTop: 8, color: 'var(--st-faint)', fontSize: 10 }}>
                        Only one heavy model runs at a time; conflicting pipelines wait in the durable queue.
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

            <div className="st-panel">
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 10, marginBottom: 10 }}>
                    <h3 className="st-panel-title" style={{ margin: 0 }}>Models</h3>
                    <div className="st-button-row">
                        {(['all', 'video', 'image', 'segmentation'] as const).map((item) => (
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
                                    {model.status === 'downloaded' && (
                                        <button
                                            type="button"
                                            className="st-button"
                                            data-size="sm"
                                            disabled={busy === model.id}
                                            onClick={async () => {
                                                setBusy(model.id);
                                                try { await setActive(model, true); } finally { setBusy(null); }
                                            }}
                                        >
                                            {busy === model.id ? <Loader2 size={12} className="animate-spin" /> : <ShieldCheck size={12} />}
                                            {busy === model.id ? 'Loading…' : 'Activate'}
                                        </button>
                                    )}
                                    {model.status === 'active' && (
                                        <button
                                            type="button"
                                            className="st-button"
                                            data-size="sm"
                                            data-variant="danger"
                                            disabled={busy === model.id}
                                            onClick={async () => {
                                                setBusy(model.id);
                                                try { await setActive(model, false); } finally { setBusy(null); }
                                            }}
                                        >
                                            {busy === model.id ? <Loader2 size={12} className="animate-spin" /> : null}
                                            Unload
                                        </button>
                                    )}
                                    {model.status !== 'downloaded' && model.status !== 'active' && (
                                        <span style={{ color: 'var(--st-faint)', fontSize: 9 }}>
                                            {model.status === 'incompatible' ? 'not for this device' : 'download in legacy Library'}
                                        </span>
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
