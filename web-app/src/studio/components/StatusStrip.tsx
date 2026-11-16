import { useStudioData } from '../data/useStudioData';
import { useTimelineStore } from '../../stores/timelineStore';

function formatBytes(bytes: number | null | undefined): string {
    if (!bytes || bytes <= 0) return '—';
    const gb = bytes / 1e9;
    return `${gb >= 1000 ? (gb / 1000).toFixed(1) + ' TB' : gb.toFixed(0) + ' GB'} free`;
}

export function StatusStrip() {
    const { health } = useStudioData();
    const saveStatus = useTimelineStore((state) => state.saveStatus);

    return (
        <footer className="st-status-strip">
            <span><strong>{health?.device?.toUpperCase() || 'DEVICE'}</strong></span>
            <span><strong>Slots</strong> {health?.slots?.active_slots?.length ? health.slots.active_slots.join(', ') : 'none'}</span>
            <span><strong>Queue</strong> {health?.gpu?.queued ?? 0}</span>
            <span><strong>Disk</strong> {formatBytes(health?.disk_free_bytes)}</span>
            <span><strong>Save</strong> {saveStatus}</span>
            <span style={{ marginLeft: 'auto' }}>Studio shell F0 · <strong>legacy editor preserved</strong></span>
        </footer>
    );
}
