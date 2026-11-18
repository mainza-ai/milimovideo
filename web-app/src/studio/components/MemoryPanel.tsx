import { HardDrive, Loader2, Pin, Snowflake, TriangleAlert } from 'lucide-react';
import { useStudioMemory } from '../data/useStudioData';
import { useStudioDocumentStore } from '../state/documentStore';

function gb(value?: number | null): string {
    if (value === null || value === undefined) return '—';
    return `${value.toFixed(1)} GB`;
}

const SOURCE_LABEL: Record<string, string> = {
    measured: 'measured',
    declared: 'declared',
    estimated: 'estimated',
    unknown: 'unknown',
};

const STAGE_LABEL: Record<string, string> = {
    normal: 'Normal',
    warning: 'Elevated',
    critical: 'Critical',
};

/**
 * Measured memory budget: the same numbers admission control uses to refuse a
 * job. Unload and keep-warm are real controls, not display-only.
 */
export function MemoryPanel() {
    const { status, error, busy, refresh, unload, keepWarm } = useStudioMemory();
    const addToast = useStudioDocumentStore((state) => state.addToast);

    const run = async (label: string, action: () => Promise<unknown>) => {
        try {
            await action();
            addToast(label, 'success');
            await refresh();
        } catch (exc) {
            addToast(exc instanceof Error ? exc.message : 'Memory action failed', 'error');
        }
    };

    const usedPercent = status && status.ceiling_gb > 0
        ? Math.min(100, Math.round((status.used_gb / status.ceiling_gb) * 100))
        : 0;

    return (
        <div className="st-panel" data-testid="memory-panel">
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 10 }}>
                <h3 className="st-panel-title" style={{ margin: 0 }}>
                    <HardDrive size={12} style={{ verticalAlign: -2, marginRight: 6 }} />
                    Memory budget
                </h3>
                <div className="st-button-row">
                    <span
                        className="st-status-pill"
                        data-tone={status
                            ? (status.pressure !== 'normal' || status.notes.length > 0 || status.swap_percent >= 90 ? 'warn' : 'good')
                            : undefined}
                        title={status?.notes.join(' · ') || undefined}
                    >
                        {status ? STAGE_LABEL[status.pressure] || status.pressure : 'unknown'}
                    </span>
                    {status?.used_gb ? (
                        <button
                            type="button"
                            className="st-button"
                            data-size="sm"
                            data-variant="ghost"
                            disabled={busy}
                            onClick={() => void run('Released resident models', () => unload({ all: true }))}
                        >
                            {busy ? <Loader2 size={11} className="animate-spin" /> : <Snowflake size={11} />}
                            Release all
                        </button>
                    ) : null}
                </div>
            </div>

            {error && (
                <p className="st-value" style={{ color: 'var(--st-danger)', marginTop: 6, fontSize: 11 }}>
                    {error}
                </p>
            )}

            {!status ? (
                <p className="st-value" style={{ color: 'var(--st-muted)', marginTop: 8 }}>
                    Reading measured memory…
                </p>
            ) : (
                <>
                    <div className="st-job-progress-bar" style={{ height: 6, marginTop: 10 }}>
                        <span
                            style={{
                                width: `${usedPercent}%`,
                                background: status.pressure === 'critical'
                                    ? 'var(--st-danger)'
                                    : status.pressure === 'warning'
                                        ? 'var(--st-warning, #d9a521)'
                                        : 'var(--st-accent)',
                            }}
                        />
                    </div>
                    <dl className="st-meta-grid" style={{ marginTop: 8 }}>
                        <dt>Used</dt>
                        <dd>{gb(status.used_gb)} of {gb(status.ceiling_gb)} ceiling</dd>
                        <dt>Available</dt><dd>{gb(status.available_gb)}</dd>
                        <dt>Machine</dt><dd>{gb(status.total_gb)} total · {gb(status.reserve_gb)} reserve</dd>
                        <dt>In process</dt><dd>{gb(status.in_process_gb)}</dd>
                        <dt>External LLM</dt><dd>{gb(status.external_gb)}</dd>
                        <dt>SAM</dt><dd>{gb(status.sam_gb)}</dd>
                        <dt>Swap</dt><dd>{status.swap_percent.toFixed(0)}%</dd>
                        <dt>Enforcement</dt><dd>{status.enforce ? 'on' : 'off'}</dd>
                    </dl>
                    {status.counters && (
                        <div className="st-button-row" style={{ marginTop: 8 }}>
                            <span className="st-chip">{status.counters.loads} load(s)</span>
                            <span className="st-chip">{status.counters.unloads} release(s)</span>
                            <span className="st-chip">{status.counters.evictions} eviction(s)</span>
                            <span className="st-chip" data-tone={status.counters.refusals > 0 ? 'warn' : undefined}>
                                {status.counters.refusals} refusal(s)
                            </span>
                        </div>
                    )}

                    {status.notes.length > 0 && (
                        <div className="st-warning-list" role="status">
                            {status.notes.map((note) => (
                                <div className="st-warning-note" key={note} data-tone="warn" title={note}>
                                    <TriangleAlert size={11} />
                                    <span>{note}</span>
                                </div>
                            ))}
                        </div>
                    )}

                    <div className="st-link-list" style={{ marginTop: 10 }}>
                        {Object.values(status.slots).map((slot) => (
                            <div className="st-link-row" key={slot.slot}>
                                <span className="st-chip" data-tone={slot.leases > 0 ? 'good' : undefined}>
                                    {slot.slot}
                                </span>
                                <span className="st-mono" style={{ flex: 1, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                                    {slot.model_id} · {gb(slot.cost_gb)}
                                    {slot.leases > 0 ? ` · ${slot.leases} lease(s)` : ''}
                                </span>
                                <button
                                    type="button"
                                    className="st-button"
                                    data-size="sm"
                                    data-variant={slot.keep_warm ? 'primary' : 'ghost'}
                                    disabled={busy}
                                    title={slot.keep_warm ? 'Pinned: skip idle eviction' : 'Pin this model'}
                                    aria-label={`${slot.keep_warm ? 'Unpin' : 'Pin'} ${slot.slot}`}
                                    onClick={() => void run(
                                        slot.keep_warm ? `Unpinned ${slot.slot}` : `Pinned ${slot.slot}`,
                                        () => keepWarm(slot.slot, !slot.keep_warm),
                                    )}
                                >
                                    <Pin size={11} />
                                </button>
                                <button
                                    type="button"
                                    className="st-button"
                                    data-size="sm"
                                    data-variant="ghost"
                                    disabled={busy || slot.leases > 0}
                                    title={slot.leases > 0 ? 'In use by a running job' : 'Unload this slot'}
                                    onClick={() => void run(`Unloaded ${slot.slot}`, () => unload({ slot: slot.slot }))}
                                >
                                    Release
                                </button>
                            </div>
                        ))}
                        {Object.keys(status.slots).length === 0 && (
                            <div className="st-link-row">
                                <span className="st-value" style={{ color: 'var(--st-muted)', fontSize: 11 }}>
                                    No model loaded in this process.
                                </span>
                            </div>
                        )}
                    </div>

                    {status.external_engines.length > 0 && (
                        <>
                            <h4 className="st-panel-title" style={{ margin: '12px 0 6px' }}>External LLM engines</h4>
                            <div className="st-link-list">
                                {status.external_engines.map((engine) => (
                                    <div className="st-link-row" key={engine.provider_id}>
                                        <span className="st-chip" data-tone={engine.unknown ? 'warn' : undefined}>
                                            {engine.provider_type}
                                        </span>
                                        <span style={{ flex: 1, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                                            {engine.name} · {gb(engine.resident_gb)}
                                            <span style={{ color: 'var(--st-faint)' }}>
                                                {' '}· {SOURCE_LABEL[engine.source] || engine.source}
                                            </span>
                                        </span>
                                        <button
                                            type="button"
                                            className="st-button"
                                            data-size="sm"
                                            data-variant="ghost"
                                            disabled={busy || !engine.can_unload || engine.resident_gb <= 0}
                                            title={engine.can_unload ? 'Unload resident models' : 'This engine cannot unload remotely'}
                                            onClick={() => void run(
                                                `Unloaded ${engine.name}`,
                                                () => unload({ providerId: engine.provider_id }),
                                            )}
                                        >
                                            Release
                                        </button>
                                    </div>
                                ))}
                            </div>
                        </>
                    )}

                    <p className="st-value" style={{ marginTop: 8, color: 'var(--st-faint)', fontSize: 11 }}>
                        Admission refuses a job when weights + workspace + external engines exceed the ceiling.
                        Pin a model to skip idle eviction; release it to make room.
                    </p>
                </>
            )}
        </div>
    );
}
