import { getAssetUrl } from '../../config';
import { useStudioJobs, type JobHistoryEntry } from '../state/jobStore';
import { useStudioDocumentStore } from '../state/documentStore';
import type { ActiveJob } from '../data/useStudioData';
import { isProjectMuted, notifyOs, notificationPreferences, playCue } from '../notifications/preferences';

export interface StudioJobEvent {
    event: string;
    job_id: string;
    shot_id?: string | null;
    project_id?: string | null;
    kind?: string | null;
    stage?: string | null;
    progress?: number;
    step?: number | null;
    total?: number | null;
    eta_seconds?: number | null;
    message?: string | null;
    error?: string | null;
    url?: string | null;
    thumbnail_url?: string | null;
    actual_frames?: number | null;
    type?: string | null;
    asset_id?: string | null;
    enhanced_prompt?: string | null;
    [key: string]: unknown;
}

interface JobIdentity {
    shotId?: string | null;
    projectId?: string | null;
    kind?: string | null;
}

const jobIndex = new Map<string, JobIdentity>();

const STAGE_LABELS: Record<string, string> = {
    queued: 'Queued',
    waiting: 'Waiting',
    starting: 'Starting',
    enhancing: 'Enhancing prompt',
    loading: 'Loading model',
    conditioning: 'Preparing conditioning',
    denoising: 'Generating',
    executing: 'Executing nodes',
    inpainting: 'Inpainting',
    upscaling: 'Upscaling',
    encoding: 'Encoding',
    saving: 'Saving',
    completed: 'Complete',
    failed: 'Failed',
    cancelled: 'Cancelled',
};

export function stageLabel(stage?: string | null): string {
    if (!stage) return '';
    return STAGE_LABELS[stage] || stage.replace(/_/g, ' ');
}

export function registerStudioJob(
    jobId: string,
    meta: { shotId?: string | null; projectId?: string | null; kind?: string | null },
): void {
    if (!jobId) return;
    const current = jobIndex.get(jobId) || {};
    jobIndex.set(jobId, {
        shotId: meta.shotId ?? current.shotId,
        projectId: meta.projectId ?? current.projectId,
        kind: meta.kind ?? current.kind,
    });
}

export function knownJob(jobId: string): JobIdentity | undefined {
    return jobIndex.get(jobId);
}

function notifyWindow(name: string, detail: unknown): void {
    if (typeof window !== 'undefined') {
        window.dispatchEvent(new CustomEvent(name, { detail }));
    }
}

function toast(message: string, type: 'success' | 'error' | 'info' = 'info'): void {
    if (!notificationPreferences().toasts) return;
    useStudioDocumentStore.getState().addToast(message, type);
}

function isTerminal(event: string): boolean {
    return event === 'job.completed' || event === 'job.failed' || event === 'job.cancelled';
}

function currentEntry(event: StudioJobEvent, identity: JobIdentity): ActiveJob {
    return {
        job_id: event.job_id,
        type: event.type || identity.kind || 'job',
        kind: identity.kind ?? undefined,
        shot_id: event.shot_id ?? identity.shotId ?? null,
        status: event.event.replace('job.', ''),
        stage: event.stage ?? null,
        step: event.step ?? null,
        total: event.total ?? null,
        progress: typeof event.progress === 'number' ? event.progress : 0,
        status_message: event.message || stageLabel(event.stage),
        eta_seconds: event.eta_seconds ?? null,
    };
}

function terminalMessage(event: StudioJobEvent): string {
    const kind = event.kind;
    if (event.event === 'job.failed') {
        return `Generation failed: ${event.error || event.message || 'unknown error'}`;
    }
    if (event.event === 'job.cancelled') return 'Generation cancelled';
    if (kind === 'thumbnail' || event.type === 'thumbnail') return 'Concept art ready';
    if (kind === 'element') return 'Element visual ready';
    if (kind === 'ai_parse') return 'Script parse complete';
    if (kind === 'depth') return 'Depth overlay ready';
    if (kind === 'comfy') return 'ComfyUI workflow complete';
    if (kind === 'inpaint') return 'Inpainted take ready';
    return 'Generation complete';
}

export function handleStudioJobEvent(event: StudioJobEvent): void {
    if (!event?.job_id || !event.event?.startsWith('job.')) return;
    const identity = jobIndex.get(event.job_id) || {};
    if (event.shot_id || event.project_id || event.kind) {
        registerStudioJob(event.job_id, {
            shotId: event.shot_id,
            projectId: event.project_id,
            kind: event.kind,
        });
    }
    const shotId = event.shot_id || identity.shotId || null;
    const entry = currentEntry(event, identity);
    const jobs = useStudioJobs.getState();
    const shotStore = useStudioDocumentStore.getState();

    notifyWindow('milimo:job-event', event);

    if (!isTerminal(event.event)) {
        jobs.upsertActive(entry);
        if (shotId && event.kind !== 'ai_parse' && event.kind !== 'depth') {
            const stage = stageLabel(event.stage);
            shotStore.updateShot(shotId, {
                isGenerating: true,
                status: 'generating',
                progress: entry.progress,
                stage: event.stage || undefined,
                statusMessage: event.message || stage || 'Working…',
                etaSeconds: event.eta_seconds ?? undefined,
                currentPrompt: typeof event.enhanced_prompt === 'string' ? event.enhanced_prompt : undefined,
                lastJobId: event.job_id,
            });
        }
        return;
    }

    const status: JobHistoryEntry = {
        ...entry,
        status_message: event.message || stageLabel(event.stage) || entry.status_message,
        progress: event.progress ?? (event.event === 'job.completed' ? 100 : entry.progress),
        finished_at: Date.now(),
        error: event.error || null,
    };
    jobs.finish(status);

    if (shotId) {
        if (event.event === 'job.completed') {
            const url = typeof event.url === 'string' ? event.url : null;
            shotStore.updateShot(shotId, {
                isGenerating: false,
                status: 'completed',
                progress: 100,
                stage: 'completed',
                statusMessage: 'Complete',
                etaSeconds: undefined,
                videoUrl: url ? (getAssetUrl(url) || url) : undefined,
                thumbnailUrl: typeof event.thumbnail_url === 'string'
                    ? (getAssetUrl(event.thumbnail_url) || event.thumbnail_url)
                    : undefined,
                numFrames: typeof event.actual_frames === 'number' ? event.actual_frames : undefined,
                lastJobId: event.job_id,
            });
        } else if (event.event === 'job.failed') {
            shotStore.updateShot(shotId, {
                isGenerating: false,
                status: 'failed',
                progress: 0,
                stage: 'failed',
                statusMessage: event.error || event.message || 'Failed',
                etaSeconds: undefined,
            });
        } else {
            shotStore.updateShot(shotId, {
                isGenerating: false,
                status: 'pending',
                progress: 0,
                stage: 'cancelled',
                statusMessage: 'Cancelled',
                etaSeconds: undefined,
            });
        }
    }

    const tone = event.event === 'job.failed' ? 'error' : event.event === 'job.cancelled' ? 'info' : 'success';
    const message = terminalMessage(event);
    const muted = isProjectMuted(event.project_id || identity.projectId);
    if (!muted) {
        toast(message, tone);
        if (tone !== 'info') {
            playCue(tone);
            notifyOs(tone === 'success' ? 'Milimo Video — done' : 'Milimo Video — failed', message, event.job_id);
        }
    }
    notifyWindow('milimo:job-terminal', { ...event, terminal: true });
}

export function handleStudioSystemEvent(type: string, data: Record<string, unknown>): void {
    if (type.startsWith('render_')) {
        notifyWindow('milimo:render-event', { type, data });
        if (type === 'render_completed') toast('Render ready', 'success');
        if (type === 'render_failed') toast(`Render failed: ${String(data.error || data.message || '')}`, 'error');
    } else if (type.startsWith('download_')) {
        notifyWindow('milimo:model-event', { type, data });
        if (type === 'download_complete') toast('Model download complete', 'success');
        if (type === 'download_error') toast(`Download failed: ${String(data.error || data.message || '')}`, 'error');
    } else if (type === 'download_queued' || type === 'queue_updated') {
        notifyWindow('milimo:model-event', { type, data });
    } else if (type === 'model_status_change' || type === 'lora_updated') {
        notifyWindow('milimo:model-event', { type, data });
    } else if (type === 'memory.pressure') {
        notifyWindow('milimo:memory-event', { type, data });
        if (data.pressure === 'critical') toast('System memory pressure is critical', 'error');
    } else if (type === 'memory.evicted') {
        notifyWindow('milimo:memory-event', { type, data });
    } else if (type === 'queue.paused') {
        toast('GPU queue paused', 'info');
    } else if (type === 'queue.resumed') {
        toast('GPU queue resumed', 'info');
    } else if (type === 'ai_parse_complete') {
        notifyWindow('milimo:job-event', { type, ...data });
        toast('Script parse complete', 'success');
    } else if (type === 'ai_parse_failed') {
        toast(`Script parse failed: ${String(data.error || '')}`, 'error');
    } else if (type === 'depth_complete') {
        toast('Depth overlay ready', 'success');
    } else if (type === 'depth_failed') {
        toast(`Depth failed: ${String(data.error || '')}`, 'error');
    }
}

/** Rebuild the index from `/active_jobs` after a reconnect or reload. */
export function registerActiveJobs(
    jobs: Array<{
        job_id: string;
        shot_id?: string | null;
        project_id?: string | null;
        type?: string;
        kind?: string;
        status?: string | null;
        stage?: string | null;
        progress?: number | null;
        status_message?: string | null;
        eta_seconds?: number | null;
    }>,
): void {
    for (const job of jobs) {
        registerStudioJob(job.job_id, {
            shotId: job.shot_id,
            projectId: job.project_id,
            kind: job.kind || job.type,
        });
        if (!job.shot_id || !['running', 'queued'].includes(job.status || '')) continue;
        const shotStore = useStudioDocumentStore.getState();
        if (!shotStore.project.shots.some((shot) => shot.id === job.shot_id)) continue;
        shotStore.updateShot(job.shot_id, {
            isGenerating: true,
            status: 'generating',
            progress: job.progress ?? 0,
            stage: job.stage || 'queued',
            statusMessage: job.status_message || (job.status === 'queued' ? 'Queued…' : 'Working…'),
            etaSeconds: job.eta_seconds ?? undefined,
            lastJobId: job.job_id,
        });
    }
}
