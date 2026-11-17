import { useEffect, useMemo, useState } from 'react';
import { Activity, AlertCircle, CheckCircle2, Loader2, RotateCcw, X } from 'lucide-react';
import { apiUrl } from '../../config';
import { useStudioJobs } from '../state/jobStore';
import { stageLabel } from '../events/jobBridge';
import { useStudioDocumentStore } from '../state/documentStore';
import { isDesktop, setDesktopBadge, setDesktopTitle } from '../../desktop';

function formatEta(seconds?: number | null): string {
    if (!seconds || seconds <= 0) return '';
    if (seconds < 60) return `${Math.round(seconds)}s left`;
    return `${Math.round(seconds / 60)}m left`;
}

export function ActivityCenter() {
    const active = useStudioJobs((state) => state.activeJobs);
    const history = useStudioJobs((state) => state.history);
    const unread = useStudioJobs((state) => state.unread);
    const clearUnread = useStudioJobs((state) => state.clearUnread);
    const addToast = useStudioDocumentStore((state) => state.addToast);
    const [open, setOpen] = useState(false);
    const busy = active.length > 0;

    const current = useMemo(() => active[0] || null, [active]);

    useEffect(() => {
        if (typeof document === 'undefined') return;
        const title = current && busy
            ? `${current.progress ?? 0}% · ${stageLabel(current.stage) || 'Working'} · Milimo`
            : 'Milimo Video';
        document.title = title;
        if (isDesktop()) {
            void setDesktopTitle(title);
            void setDesktopBadge(
                current && busy ? `${Math.max(0, Math.min(100, current.progress ?? 0))}%` : null,
                current && busy ? Math.max(1, active.length) : 0,
            );
        }
        return () => {
            document.title = 'Milimo Video';
            if (isDesktop()) {
                void setDesktopTitle('Milimo Video');
                void setDesktopBadge(null, 0);
            }
        };
    }, [active.length, busy, current]);

    const cancel = async (jobId: string) => {
        await fetch(apiUrl(`/jobs/${jobId}/cancel`), { method: 'POST' }).catch(() => undefined);
    };

    const retry = async (job: { shot_id?: string | null; job_id: string }) => {
        if (!job.shot_id) {
            addToast('This job has no shot to retry', 'info');
            return;
        }
        try {
            const response = await fetch(apiUrl(`/shots/${job.shot_id}/generate`), { method: 'POST' });
            const data = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
            addToast(`Retrying shot (job ${String(data.job_id || '').slice(0, 12)})`, 'success');
        } catch (exc) {
            addToast(exc instanceof Error ? exc.message : 'Retry failed', 'error');
        }
    };

    return (
        <div className="st-activity-wrap">
            <button
                type="button"
                className="st-icon-button"
                aria-label={busy ? `Activity: ${active.length} running` : 'Activity'}
                aria-expanded={open}
                title={busy ? `${current?.status_message || stageLabel(current?.stage)}` : 'Activity'}
                onClick={() => {
                    setOpen(!open);
                    if (!open) clearUnread();
                }}
            >
                {busy ? <Loader2 size={15} className="animate-spin" /> : <Activity size={15} />}
                {(active.length + unread) > 0 && (
                    <span className="st-chip" style={{ marginLeft: 4, padding: '0 5px', fontSize: 9 }}>
                        {active.length + unread}
                    </span>
                )}
            </button>
            {open && (
                <div className="st-activity-panel" role="dialog" aria-label="Job activity">
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                        <strong style={{ fontSize: 11 }}>Activity</strong>
                        <button type="button" className="st-icon-button" aria-label="Close activity" onClick={() => setOpen(false)}>
                            <X size={12} />
                        </button>
                    </div>
                    {active.length === 0 && history.length === 0 && (
                        <div className="st-value" style={{ color: 'var(--st-muted)', fontSize: 11 }}>
                            No jobs yet. Generate a shot to see progress here.
                        </div>
                    )}
                    {active.map((job) => (
                        <div className="st-activity-row" key={job.job_id} role="status">
                            <Loader2 size={12} className="animate-spin" />
                            <div style={{ flex: 1, minWidth: 0 }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', gap: 6 }}>
                                    <span>{stageLabel(job.stage) || job.status_message || 'Working'}</span>
                                    <span className="st-mono">{job.progress ?? 0}%</span>
                                </div>
                                <div className="st-job-progress-bar" style={{ marginTop: 4 }}>
                                    <span style={{ width: `${Math.max(0, Math.min(100, job.progress ?? 0))}%` }} />
                                </div>
                                <div className="st-job-progress-meta">
                                    <span>{job.status_message || job.kind || job.type}</span>
                                    <span>{formatEta(job.eta_seconds)}</span>
                                </div>
                            </div>
                            <button
                                type="button"
                                className="st-button"
                                data-size="sm"
                                data-variant="ghost"
                                aria-label={`Cancel ${job.job_id}`}
                                onClick={() => void cancel(job.job_id)}
                            >
                                <X size={11} />
                            </button>
                        </div>
                    ))}
                    {history.slice(0, 6).map((job) => (
                        <div className="st-activity-row" key={`${job.job_id}-${job.finished_at}`}>
                            {job.status === 'failed'
                                ? <AlertCircle size={12} style={{ color: 'var(--st-danger)' }} />
                                : <CheckCircle2 size={12} style={{ color: 'var(--st-success)' }} />}
                            <div style={{ flex: 1, minWidth: 0 }}>
                                <div style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                    {job.status_message || job.status}
                                </div>
                                <div className="st-mono" style={{ color: 'var(--st-faint)', fontSize: 10 }}>
                                    {job.kind || job.type} · {new Date(job.finished_at).toLocaleTimeString()}
                                </div>
                            </div>
                            {job.shot_id && (
                                <button
                                    type="button"
                                    className="st-button"
                                    data-size="sm"
                                    data-variant="ghost"
                                    aria-label={`Retry ${job.job_id}`}
                                    title="Queue the same shot again"
                                    onClick={() => void retry(job)}
                                >
                                    <RotateCcw size={11} />
                                </button>
                            )}
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}
