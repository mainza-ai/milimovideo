import { useStudioData } from '../data/useStudioData';
import { useStudioDocumentStore } from '../state/documentStore';
import { useStudioJobs } from '../state/jobStore';
import { stageLabel } from '../events/jobBridge';

function formatBytes(bytes: number | null | undefined): string {
    if (!bytes || bytes <= 0) return '—';
    const gb = bytes / 1e9;
    return `${gb >= 1000 ? (gb / 1000).toFixed(1) + ' TB' : gb.toFixed(0) + ' GB'} free`;
}

export function StatusStrip() {
    const { health } = useStudioData();
    const saveStatus = useStudioDocumentStore((state) => state.saveStatus);
    const currentJob = useStudioJobs((state) => state.activeJobs[0]);

    return (
        <footer className="st-status-strip">
            <span><strong>{health?.device?.toUpperCase() || 'DEVICE'}</strong></span>
            <span><strong>Slots</strong> {health?.slots?.active_slots?.length ? health.slots.active_slots.join(', ') : 'none'}</span>
            {currentJob ? (
                <span data-testid="status-current-job">
                    <strong>{stageLabel(currentJob.stage) || 'Job'}</strong> {currentJob.progress ?? 0}%
                    {currentJob.eta_seconds ? ` · ${Math.round(currentJob.eta_seconds)}s` : ''}
                </span>
            ) : (
                <span><strong>Queue</strong> {health?.gpu?.queued ?? 0}</span>
            )}
            <span><strong>Disk</strong> {formatBytes(health?.disk_free_bytes)}</span>
            <span><strong>Save</strong> {saveStatus}</span>
            <span style={{ marginLeft: 'auto' }}>Studio · <strong>local-first</strong></span>
        </footer>
    );
}
