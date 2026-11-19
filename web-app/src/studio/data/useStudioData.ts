import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useShallow } from 'zustand/react/shallow';
import { useStudioDocumentStore, type Project, type Shot } from '../state/documentStore';
import { apiUrl, getAssetUrl } from '../../config';
import { HealthSchema, ReviewPayloadSchema } from '../../api/contracts';
import { useStudioJobs } from '../state/jobStore';
import { registerActiveJobs, registerStudioJob } from '../events/jobBridge';
import {
    useStudioEditor,
    type BoardArea,
    type BoardLink,
    type BoardPosition,
    type Camera,
} from '../state/editorStore';

export interface StudioHealth {
    status: string;
    device: string;
    gpu: { busy: boolean; queued: number };
    queue_paused?: boolean;
    slots: { active_slots: string[] };
    disk_free_bytes: number | null;
    auth_required: boolean;
}

export interface StudioModel {
    id: string;
    name: string;
    family: string;
    role: string;
    pipeline: string;
    type: string;
    status: 'downloaded' | 'active' | 'not_downloaded' | 'downloading' | 'incompatible' | 'error';
    size_bytes: number;
    vram_estimate_gb?: number;
    required?: boolean;
    description?: string;
    requires_base?: boolean;
    device_compatibility?: string[];
    recommended?: boolean;
    recommendation?: string;
}

export interface PipelineReadiness {
    ready: boolean;
    missing: string[];
    total: number;
    downloaded: number;
}

export function useStudioModels() {
    const addToast = useStudioDocumentStore((state) => state.addToast);
    const [models, setModels] = useState<StudioModel[]>([]);
    const [pipelines, setPipelines] = useState<Record<string, PipelineReadiness>>({});
    const [loading, setLoading] = useState(true);

    const refresh = useCallback(async () => {
        setLoading(true);
        try {
            const [modelsResponse, pipelineResponse] = await Promise.all([
                fetch(apiUrl('/models')),
                fetch(apiUrl('/models/pipelines')),
            ]);
            const modelData = modelsResponse.ok ? await modelsResponse.json() : { models: [] };
            const pipelineData = pipelineResponse.ok ? await pipelineResponse.json() : {};
            setModels(Array.isArray(modelData.models) ? modelData.models : []);
            setPipelines(pipelineData && typeof pipelineData === 'object' ? pipelineData : {});
        } catch {
            setModels([]);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => { void refresh(); }, [refresh]);

    useEffect(() => {
        const handler = () => { void refresh(); };
        window.addEventListener('milimo:model-event', handler);
        return () => window.removeEventListener('milimo:model-event', handler);
    }, [refresh]);

    useEffect(() => {
        if (!models.some((model) => model.status === 'downloading')) return undefined;
        const interval = window.setInterval(() => { void refresh(); }, 3000);
        return () => window.clearInterval(interval);
    }, [models, refresh]);

    const setActive = useCallback(async (model: StudioModel, active: boolean) => {
        const response = await fetch(apiUrl(`/models/${model.id}/${active ? 'activate' : 'deactivate'}`), {
            method: 'POST',
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
            addToast(data.detail || `Could not ${active ? 'activate' : 'deactivate'} ${model.name}`, 'error');
            return false;
        }
        addToast(data.message || `${model.name} ${active ? 'activated' : 'deactivated'}`, 'success');
        await refresh();
        return true;
    }, [addToast, refresh]);

    return { models, pipelines, loading, refresh, setActive };
}

export interface StudioElement {
    id: string;
    name: string;
    type: string;
    description: string;
    trigger_word: string;
    image_path?: string;
}

export interface StudioAsset {
    id: string;
    url: string;
    filename: string;
    type: string;
    thumbnail?: string | null;
    missing?: boolean;
    created_at: number;
}

export interface StudioTake {
    job_id: string;
    url: string;
    thumbnail_url?: string | null;
    type: string;
    prompt?: string | null;
    enhanced_prompt?: string | null;
    seed?: number | null;
    model?: string | null;
    num_frames?: number | null;
    llm?: {
        role?: string;
        provider?: string | null;
        model?: string | null;
        latency_ms?: number | null;
        tokens?: Record<string, number> | null;
        failed?: boolean;
    } | null;
    created_at?: string | null;
    selected: boolean;
    favorite?: boolean;
    missing?: boolean;
}

export interface ReviewComment {
    id: string;
    project_id: string;
    shot_id: string | null;
    kind: 'comment' | 'approval';
    frame: number;
    body: string;
    author: string;
    status: 'open' | 'resolved' | 'approved' | 'changes_requested';
    created_at: string | null;
    updated_at: string | null;
}

export interface ReviewPayload {
    comments: ReviewComment[];
    approvals: ReviewComment[];
    summary: {
        comments: number;
        open: number;
        approved: number;
        changes_requested: number;
    };
}

export interface StudioRender {
    id: string;
    status: string;
    progress: number;
    output_path: string | null;
    thumbnail_path?: string | null;
    error_message: string | null;
    created_at: string;
}

export function useStudioProject(): Project {
    return useStudioDocumentStore((state) => state.project);
}

/**
 * If the persisted project no longer exists on this machine, open the most
 * recent server project instead of silently editing a phantom local document.
 * Never creates projects implicitly.
 */
export function useStudioProjectBootstrap() {
    const projectId = useStudioDocumentStore((state) => state.project.id);
    const loadProject = useStudioDocumentStore((state) => state.loadProject);

    useEffect(() => {
        let alive = true;
        fetch(apiUrl('/projects'))
            .then((response) => response.ok ? response.json() : [])
            .then((projects) => {
                if (!alive || !Array.isArray(projects) || projects.length === 0) return;
                // `default` is the local placeholder sentinel, not a server
                // choice: open the newest real project on a fresh load. The
                // placeholder is only used when it is the only project.
                if (projectId !== 'default' && projects.some((item) => item.id === projectId)) return;
                const real = projects.filter((item) => item.id !== 'default');
                const pool = real.length ? real : projects;
                const newest = [...pool].sort(
                    (a, b) => Number(b.updated_at || 0) - Number(a.updated_at || 0),
                )[0];
                if (!newest?.id || newest.id === projectId) return;
                void loadProject(newest.id);
            })
            .catch(() => undefined);
        return () => { alive = false; };
    }, [loadProject, projectId]);
}

export function useStudioData() {
    const project = useStudioProject();
    const { health, error } = useStudioHealth();
    return { project, health, error };
}

export function useStudioShots(): Shot[] {
    return useStudioDocumentStore(useShallow((state) => state.project.shots));
}

export function useStudioSelection() {
    return useStudioDocumentStore(useShallow((state) => ({
        selectedShotId: state.selectedShotId,
        selectShot: state.selectShot,
        updateShot: state.updateShot,
        patchShot: state.patchShot,
        isPlaying: state.isPlaying,
        setIsPlaying: state.setIsPlaying,
        setCurrentTime: state.setCurrentTime,
        saveProject: state.saveProject,
        saveStatus: state.saveStatus,
        saveError: state.saveError,
        addToast: state.addToast,
    })));
}

export function useStudioHealth() {
    const query = useQuery({
        queryKey: ['studio', 'health'],
        queryFn: async (): Promise<StudioHealth> => {
            const response = await fetch(apiUrl('/health'));
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const parsed = HealthSchema.safeParse(await response.json());
            if (!parsed.success) throw new Error('Health response failed contract validation');
            return parsed.data;
        },
        refetchInterval: 10000,
    });

    return {
        health: query.data ?? null,
        error: query.error ? (query.error instanceof Error ? query.error.message : String(query.error)) : null,
    };
}

export function useStudioElements(projectId: string) {
    const [elements, setElements] = useState<StudioElement[]>([]);
    const [loading, setLoading] = useState(true);
    const refresh = useCallback(async () => {
        try {
            const response = await fetch(apiUrl(`/projects/${projectId}/elements`));
            const data = response.ok ? await response.json() : [];
            setElements(Array.isArray(data) ? data : []);
        } catch {
            setElements([]);
        } finally {
            setLoading(false);
        }
    }, [projectId]);
    useEffect(() => { void refresh(); }, [refresh]);
    return { elements, loading, refresh };
}

export interface ElementDraft {
    name: string;
    type: string;
    description: string;
    trigger_word: string;
    image_path?: string | null;
}

export function useStudioElementMutations(projectId: string) {
    const addToast = useStudioDocumentStore((state) => state.addToast);
    const create = useCallback(async (draft: ElementDraft) => {
        const response = await fetch(apiUrl(`/projects/${projectId}/elements`), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(draft),
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
            addToast(data.detail || 'Could not create reference', 'error');
            return null;
        }
        addToast(`Created ${draft.name}`, 'success');
        return data as StudioElement;
    }, [addToast, projectId]);

    const update = useCallback(async (elementId: string, draft: ElementDraft) => {
        const response = await fetch(apiUrl(`/elements/${elementId}`), {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(draft),
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
            addToast(data.detail || 'Could not update reference', 'error');
            return null;
        }
        addToast(`Updated ${draft.name}`, 'success');
        return data as StudioElement;
    }, [addToast]);

    const remove = useCallback(async (element: StudioElement) => {
        const response = await fetch(apiUrl(`/elements/${element.id}`), { method: 'DELETE' });
        const data = await response.json().catch(() => ({}));
        if (!response.ok || data.success === false) {
            addToast(data.detail || 'Could not delete reference', 'error');
            return false;
        }
        addToast(`Deleted ${element.name}`, 'success');
        return true;
    }, [addToast]);

    return { create, update, remove };
}

export function useStudioUploadAsset(projectId: string) {
    const addToast = useStudioDocumentStore((state) => state.addToast);
    return useCallback(async (file: File) => {
        const form = new FormData();
        form.append('file', file);
        form.append('project_id', projectId);
        form.append('type', 'general');
        const response = await fetch(apiUrl('/upload'), { method: 'POST', body: form });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
            addToast(data.detail || 'Upload failed', 'error');
            return null;
        }
        addToast(`Uploaded ${file.name}`, 'success');
        return data as { url: string; type: string; filename: string; asset_id: string };
    }, [addToast, projectId]);
}

export function useStudioAssets(projectId?: string) {
    const [assets, setAssets] = useState<StudioAsset[]>([]);
    const [loading, setLoading] = useState(true);
    const refresh = useCallback(async () => {
        setLoading(true);
        try {
            const query = projectId ? `?project_id=${encodeURIComponent(projectId)}` : '';
            const response = await fetch(apiUrl(`/uploads${query}`));
            const data = response.ok ? await response.json() : [];
            setAssets(Array.isArray(data) ? data : []);
        } catch {
            setAssets([]);
        } finally {
            setLoading(false);
        }
    }, [projectId]);
    useEffect(() => { void refresh(); }, [refresh]);
    useEffect(() => {
        const handler = () => { void refresh(); };
        window.addEventListener('milimo:job-terminal', handler);
        window.addEventListener('milimo:resync', handler);
        return () => {
            window.removeEventListener('milimo:job-terminal', handler);
            window.removeEventListener('milimo:resync', handler);
        };
    }, [refresh]);
    return { assets, loading, refresh };
}

export function useStudioTakes(shotId: string | undefined) {
    const [takes, setTakes] = useState<StudioTake[]>([]);
    const [loading, setLoading] = useState(false);
    const refresh = useCallback(async () => {
        if (!shotId) {
            setTakes([]);
            return;
        }
        setLoading(true);
        try {
            const response = await fetch(apiUrl(`/shots/${shotId}/takes`));
            const data = response.ok ? await response.json() : { takes: [] };
            setTakes(Array.isArray(data.takes) ? data.takes : []);
        } catch {
            setTakes([]);
        } finally {
            setLoading(false);
        }
    }, [shotId]);
    useEffect(() => { void refresh(); }, [refresh]);
    useEffect(() => {
        const handler = (event: Event) => {
            const detail = (event as CustomEvent).detail || {};
            if (!shotId) return;
            if (!detail.shot_id || detail.shot_id === shotId) void refresh();
        };
        window.addEventListener('milimo:job-terminal', handler);
        window.addEventListener('milimo:resync', handler);
        return () => {
            window.removeEventListener('milimo:job-terminal', handler);
            window.removeEventListener('milimo:resync', handler);
        };
    }, [refresh, shotId]);
    return { takes, loading, refresh };
}

export function useStudioSelectTake() {
    const { addToast, updateShot } = useStudioSelection();
    return useCallback(async (shot: Shot, take: StudioTake) => {
        const response = await fetch(apiUrl(`/shots/${shot.id}/select-take`), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ job_id: take.job_id }),
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
            addToast(data.detail || 'Could not select take', 'error');
            return false;
        }
        updateShot(shot.id, {
            videoUrl: getAssetUrl(data.url) || take.url,
            thumbnailUrl: getAssetUrl(take.thumbnail_url) || shot.thumbnailUrl,
            lastJobId: take.job_id,
            selectedTakeId: take.job_id,
            status: 'completed',
            isGenerating: false,
        });
        addToast('Take selected', 'success');
        return true;
    }, [addToast, updateShot]);
}

const EMPTY_REVIEW: ReviewPayload = {
    comments: [],
    approvals: [],
    summary: { comments: 0, open: 0, approved: 0, changes_requested: 0 },
};

export interface IntegrityIssue {
    kind: string;
    severity: 'error' | 'warning' | 'info';
    message: string;
    shot_id?: string | null;
    path?: string | null;
}

export interface IntegrityReport {
    project_id: string;
    ok: boolean;
    counts: {
        shots: number;
        scenes: number;
        issues: number;
        errors: number;
        warnings: number;
    };
    issues: IntegrityIssue[];
}

export function useStudioIntegrity(projectId: string) {
    const [report, setReport] = useState<IntegrityReport | null>(null);
    const [loading, setLoading] = useState(false);
    const scan = useCallback(async () => {
        setLoading(true);
        try {
            const response = await fetch(apiUrl(`/projects/${projectId}/integrity`));
            const data = response.ok ? await response.json() : null;
            setReport(data);
        } catch {
            setReport(null);
        } finally {
            setLoading(false);
        }
    }, [projectId]);
    return { report, loading, scan };
}

export interface MemorySlot {
    slot: string;
    model_id: string;
    cost_gb: number;
    leases: number;
    keep_warm: boolean;
    last_used: number;
    acquired_at: number;
}

export interface ExternalEngine {
    provider_id: string;
    provider_type: string;
    name: string;
    base_url: string;
    resident_gb: number;
    models: string[];
    can_unload: boolean;
    source: string;
    error?: string | null;
    unknown: boolean;
}

export interface MemoryStatus {
    total_gb: number;
    reserve_gb: number;
    ceiling_gb: number;
    in_process_gb: number;
    resident_gb: number;
    external_gb: number;
    sam_gb: number;
    used_gb: number;
    available_gb: number;
    swap_percent: number;
    pressure: 'normal' | 'warning' | 'critical' | string;
    enforce: boolean;
    external_engines: ExternalEngine[];
    slots: Record<string, MemorySlot>;
    active_jobs: Array<{ job_id: string; kind: string; started_at: number }>;
    active_slots?: string[];
    notes: string[];
    counters?: { loads: number; unloads: number; evictions: number; refusals: number };
}

export interface MemoryDecision {
    ok: boolean;
    kind: string;
    reason: string | null;
    estimate: {
        total_gb: number;
        base_gb: number;
        workspace_gb: number;
        frames?: number | null;
        width?: number | null;
        height?: number | null;
        notes: string[];
    } | null;
    reclaimable_gb: number;
    will_unload: string[];
    suggestions: string[];
    warnings: string[];
}

export function useStudioMemory(intervalMs = 5000) {
    const [status, setStatus] = useState<MemoryStatus | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [busy, setBusy] = useState(false);

    const refresh = useCallback(async () => {
        try {
            const response = await fetch(apiUrl('/system/memory'));
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const data = await response.json();
            if (!data || typeof data !== 'object' || Array.isArray(data) || typeof data.total_gb !== 'number') {
                throw new Error('Memory status response was not a snapshot');
            }
            setStatus({
                ...data,
                external_engines: Array.isArray(data.external_engines) ? data.external_engines : [],
                slots: data.slots && typeof data.slots === 'object' ? data.slots : {},
                notes: Array.isArray(data.notes) ? data.notes : [],
                active_jobs: Array.isArray(data.active_jobs) ? data.active_jobs : [],
            });
            setError(null);
        } catch (exc) {
            setError(exc instanceof Error ? exc.message : 'Memory status unavailable');
        }
    }, []);

    useEffect(() => {
        void refresh();
        const interval = window.setInterval(() => { void refresh(); }, intervalMs);
        const handler = () => { void refresh(); };
        window.addEventListener('milimo:memory-event', handler);
        window.addEventListener('milimo:resync', handler);
        return () => {
            window.clearInterval(interval);
            window.removeEventListener('milimo:memory-event', handler);
            window.removeEventListener('milimo:resync', handler);
        };
    }, [intervalMs, refresh]);

    const unload = useCallback(async (target: { slot?: string; providerId?: string; all?: boolean }) => {
        setBusy(true);
        try {
            const response = await fetch(apiUrl('/system/memory/unload'), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    all: target.all ?? false,
                    slot: target.slot ?? null,
                    provider_id: target.providerId ?? null,
                }),
            });
            const data = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(data.detail || 'Unload failed');
            setStatus(data.memory || null);
            return data;
        } finally {
            setBusy(false);
        }
    }, []);

    const keepWarm = useCallback(async (slot: string, keepWarmValue: boolean) => {
        setBusy(true);
        try {
            const response = await fetch(apiUrl('/system/memory/keep-warm'), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ slot, keep_warm: keepWarmValue }),
            });
            const data = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(data.detail || 'Keep-warm update failed');
            setStatus(data);
        } finally {
            setBusy(false);
        }
    }, []);

    return { status, error, busy, refresh, unload, keepWarm };
}

export interface JobMetrics {
    window_days: number;
    total: number;
    by_type: Record<string, number>;
    by_status: Record<string, number>;
    recent_failures: Array<{ job_id: string; type: string; error: string | null }>;
}

export function useStudioMetrics(days = 7) {
    const metrics = useStudioJobs((state) => state.metrics);
    const setMetrics = useStudioJobs((state) => state.setMetrics);
    useEffect(() => {
        const load = async () => {
            try {
                const response = await fetch(apiUrl(`/metrics/jobs?days=${days}`));
                const data = response.ok ? await response.json() : null;
                setMetrics(
                    data && !Array.isArray(data) && data.by_status && data.by_type
                        ? data
                        : null,
                );
            } catch {
                setMetrics(null);
            }
        };
        void load();
        const interval = window.setInterval(load, 15000);
        return () => window.clearInterval(interval);
    }, [days, setMetrics]);
    return metrics;
}

export interface ActiveJob {
    job_id: string;
    type: string;
    kind?: string;
    shot_id?: string | null;
    status: string;
    stage?: string | null;
    step?: number | null;
    total?: number | null;
    progress: number;
    status_message: string;
    position?: number;
    estimate_seconds?: number;
    eta_seconds?: number | null;
    started_at?: string | null;
}

export interface ActiveLora {
    model_id: string;
    name: string;
    strength: number;
    role: string;
}

export function useStudioLoras(slot: 'video' | 'image' = 'video') {
    const addToast = useStudioDocumentStore((state) => state.addToast);
    const [loras, setLoras] = useState<ActiveLora[]>([]);
    const refresh = useCallback(async () => {
        try {
            const response = await fetch(apiUrl(`/models/loras/${slot}`));
            const data = response.ok ? await response.json() : { loras: [] };
            setLoras(Array.isArray(data.loras) ? data.loras : []);
        } catch {
            setLoras([]);
        }
    }, [slot]);
    useEffect(() => { void refresh(); }, [refresh]);

    const setStrength = useCallback(async (modelId: string, strength: number) => {
        const response = await fetch(apiUrl(`/models/${modelId}/lora`), {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ strength }),
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
            addToast(data.detail || 'Could not update LoRA strength', 'error');
            return false;
        }
        await refresh();
        return true;
    }, [addToast, refresh]);

    return { loras, refresh, setStrength };
}

export function useStudioActiveJobs(projectId: string) {
    const jobs = useStudioJobs((state) => state.activeJobs);
    const setJobs = useStudioJobs((state) => state.setActiveJobs);
    const refresh = useCallback(async () => {
        try {
            const response = await fetch(apiUrl(`/projects/${projectId}/active_jobs`));
            const data = response.ok ? await response.json() : [];
            const rows = Array.isArray(data) ? data : [];
            setJobs(rows);
            registerActiveJobs(rows);
        } catch {
            setJobs([]);
        }
    }, [projectId, setJobs]);

    const reconcile = useCallback(async () => {
        const shots = useStudioDocumentStore.getState().project.shots.filter(
            (shot) => shot.isGenerating && shot.lastJobId,
        );
        const activeIds = new Set(useStudioJobs.getState().activeJobs.map((job) => job.job_id));
        for (const shot of shots) {
            if (!shot.lastJobId || activeIds.has(shot.lastJobId)) continue;
            try {
                const status = await fetch(apiUrl(`/status/${shot.lastJobId}`)).then((response) => response.json());
                if (status.status === 'completed') {
                    useStudioDocumentStore.getState().updateShot(shot.id, {
                        isGenerating: false, status: 'completed', progress: 100,
                        statusMessage: 'Complete', stage: 'completed',
                        videoUrl: status.video_url ? (getAssetUrl(status.video_url) || status.video_url) : undefined,
                        thumbnailUrl: status.thumbnail_url
                            ? (getAssetUrl(status.thumbnail_url) || status.thumbnail_url)
                            : undefined,
                    });
                } else if (status.status === 'failed' || status.status === 'cancelled') {
                    useStudioDocumentStore.getState().updateShot(shot.id, {
                        isGenerating: false,
                        status: status.status === 'failed' ? 'failed' : 'pending',
                        progress: 0,
                        statusMessage: status.error || (status.status === 'failed' ? 'Failed' : 'Cancelled'),
                        stage: status.status,
                    });
                }
            } catch {
                useStudioDocumentStore.getState().updateShot(shot.id, {
                    isGenerating: false, statusMessage: 'Unknown', stage: 'failed',
                });
            }
        }
    }, []);

    useEffect(() => {
        void refresh();
        void reconcile();
        const interval = window.setInterval(() => { void refresh(); }, 2500);
        const onResync = () => { void refresh(); void reconcile(); };
        window.addEventListener('milimo:resync', onResync);
        window.addEventListener('milimo:job-terminal', onResync);
        return () => {
            window.clearInterval(interval);
            window.removeEventListener('milimo:resync', onResync);
            window.removeEventListener('milimo:job-terminal', onResync);
        };
    }, [refresh, reconcile]);
    return { jobs, refresh };
}

export function useStudioReview(projectId: string) {
    const addToast = useStudioDocumentStore((state) => state.addToast);
    const [review, setReview] = useState<ReviewPayload>(EMPTY_REVIEW);
    const [loading, setLoading] = useState(true);

    const refresh = useCallback(async () => {
        try {
            const response = await fetch(apiUrl(`/projects/${projectId}/review`));
            if (!response.ok) {
                setReview(EMPTY_REVIEW);
                return;
            }
            const parsed = ReviewPayloadSchema.safeParse(await response.json());
            if (!parsed.success) {
                addToast('Review response failed contract validation', 'error');
                setReview(EMPTY_REVIEW);
                return;
            }
            setReview(parsed.data);
        } catch {
            setReview(EMPTY_REVIEW);
        } finally {
            setLoading(false);
        }
    }, [addToast, projectId]);

    useEffect(() => {
        void refresh();
        const interval = window.setInterval(refresh, 15000);
        return () => window.clearInterval(interval);
    }, [refresh]);

    const addComment = useCallback(async (shotId: string | null, frame: number, body: string) => {
        const response = await fetch(apiUrl(`/projects/${projectId}/review/comments`), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ shot_id: shotId, frame, body }),
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
            addToast(data.detail || 'Could not add comment', 'error');
            return false;
        }
        await refresh();
        addToast(`Comment added at frame ${frame}`, 'success');
        return true;
    }, [addToast, projectId, refresh]);

    const updateComment = useCallback(async (commentId: string, patch: { status?: 'open' | 'resolved'; body?: string }) => {
        const response = await fetch(apiUrl(`/projects/${projectId}/review/comments/${commentId}`), {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(patch),
        });
        if (!response.ok) {
            addToast('Could not update comment', 'error');
            return false;
        }
        await refresh();
        return true;
    }, [addToast, projectId, refresh]);

    const deleteComment = useCallback(async (commentId: string) => {
        const response = await fetch(apiUrl(`/projects/${projectId}/review/comments/${commentId}`), {
            method: 'DELETE',
        });
        if (!response.ok) {
            addToast('Could not delete comment', 'error');
            return false;
        }
        await refresh();
        return true;
    }, [addToast, projectId, refresh]);

    const setApproval = useCallback(async (shotId: string, status: 'approved' | 'changes_requested', note = '') => {
        const response = await fetch(apiUrl(`/projects/${projectId}/review/approvals`), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ shot_id: shotId, status, note }),
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
            addToast(data.detail || 'Could not save approval', 'error');
            return false;
        }
        await refresh();
        addToast(status === 'approved' ? 'Shot approved' : 'Changes requested', 'success');
        return true;
    }, [addToast, projectId, refresh]);

    return { review, loading, refresh, addComment, updateComment, deleteComment, setApproval };
}

export interface StudioProjectTake extends StudioTake {
    shot_id?: string | null;
    shot_label?: string | null;
    sequence_index?: number | null;
}

export function useStudioProjectTakes(projectId: string) {
    const [takes, setTakes] = useState<StudioProjectTake[]>([]);
    const [loading, setLoading] = useState(true);
    const refresh = useCallback(async () => {
        setLoading(true);
        try {
            const response = await fetch(apiUrl(`/projects/${projectId}/takes`));
            const data = response.ok ? await response.json() : { takes: [] };
            setTakes(Array.isArray(data.takes) ? data.takes : []);
        } catch {
            setTakes([]);
        } finally {
            setLoading(false);
        }
    }, [projectId]);
    useEffect(() => { void refresh(); }, [refresh]);
    useEffect(() => {
        const handler = () => { void refresh(); };
        window.addEventListener('milimo:job-terminal', handler);
        window.addEventListener('milimo:resync', handler);
        return () => {
            window.removeEventListener('milimo:job-terminal', handler);
            window.removeEventListener('milimo:resync', handler);
        };
    }, [refresh]);
    return { takes, loading, refresh };
}

export function useStudioFavoriteTake() {
    const { addToast, updateShot } = useStudioSelection();
    return useCallback(async (shot: Shot, take: StudioTake | null) => {
        const response = await fetch(apiUrl(`/shots/${shot.id}/favorite-take`), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ job_id: take?.job_id ?? null }),
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
            addToast(data.detail || 'Could not update favorite take', 'error');
            return false;
        }
        updateShot(shot.id, { favoriteTakeId: take?.job_id });
        addToast(take ? 'Take marked as favorite' : 'Favorite cleared', 'success');
        return true;
    }, [addToast, updateShot]);
}

export function useStudioRenders(projectId: string) {
    const [renders, setRenders] = useState<StudioRender[]>([]);
    const refresh = useCallback(async () => {
        try {
            const response = await fetch(apiUrl(`/projects/${projectId}/renders`));
            const data = response.ok ? await response.json() : [];
            setRenders(Array.isArray(data) ? data : []);
        } catch {
            setRenders([]);
        }
    }, [projectId]);
    useEffect(() => {
        void refresh();
        const interval = window.setInterval(refresh, 4000);
        const handler = () => { void refresh(); };
        window.addEventListener('milimo:render-event', handler);
        window.addEventListener('milimo:resync', handler);
        return () => {
            window.clearInterval(interval);
            window.removeEventListener('milimo:render-event', handler);
            window.removeEventListener('milimo:resync', handler);
        };
    }, [refresh]);
    return { renders, refresh };
}

export interface MemoryDowngrade {
    params: { num_frames?: number; width?: number; height?: number; upscale?: boolean };
    label: string;
    estimate?: { total_gb?: number } | null;
}

export function useStudioGenerate(options?: {
    onDowngrade?: (shot: Shot, downgrade: MemoryDowngrade, reason: string) => void;
}) {
    const { addToast, updateShot } = useStudioSelection();
    const onDowngrade = options?.onDowngrade;

    return useCallback(async (shot: Shot, randomizeSeed = false) => {
        if (randomizeSeed && shot.seedLocked !== false) {
            addToast('Seed is locked. Unlock it in Advanced parameters to make a variation.', 'info');
            return null;
        }

        // Budget preflight: refuse before the server starts a load.
        const kind = shot.numFrames === 1 ? 'image' : 'video';
        try {
            const preflight = await fetch(apiUrl('/system/memory/preflight'), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    kind,
                    params: {
                        width: shot.width,
                        height: shot.height,
                        num_frames: shot.numFrames,
                        upscale: shot.upscale,
                        reference_count: shot.timeline.filter((item) => item.type === 'image').length,
                    },
                }),
            }).then((response) => response.ok ? response.json() : null);
            if (preflight && preflight.ok === false) {
                if (preflight.retryable) {
                    // A running job holds the bytes; the scheduler will wait and
                    // re-admit instead of failing this job immediately.
                    addToast('Another job is running. This shot will wait for memory.', 'info');
                } else if (preflight.downgrade && onDowngrade) {
                    onDowngrade(shot, preflight.downgrade as MemoryDowngrade, String(preflight.reason || ''));
                    updateShot(shot.id, { isGenerating: false, statusMessage: 'Needs a smaller preset' });
                    return null;
                } else {
                    const suggestions = Array.isArray(preflight.suggestions) && preflight.suggestions.length
                        ? ` ${preflight.suggestions.slice(0, 2).join('; ')}.`
                        : '';
                    addToast(`Generation blocked: ${preflight.reason}${suggestions}`, 'error');
                    updateShot(shot.id, { isGenerating: false, statusMessage: 'Blocked: not enough memory' });
                    return null;
                }
            }
            if (preflight?.will_unload?.length) {
                addToast(`Will release ${preflight.will_unload.slice(0, 2).join(', ')} to make room`, 'info');
            }
        } catch {
            // Older backend or transient failure: server admission still guards the load.
        }

        updateShot(shot.id, {
            isGenerating: true, progress: 0, stage: 'queued', statusMessage: 'Queued…',
        });
        const projectId = useStudioDocumentStore.getState().project.id;
        try {
            if (shot.sceneId) {
                const response = await fetch(
                    apiUrl(`/shots/${shot.id}/generate${randomizeSeed ? '?randomize_seed=true' : ''}`),
                    { method: 'POST' },
                );
                const data = await response.json().catch(() => ({}));
                if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
                const jobId = data.job_id as string;
                registerStudioJob(jobId, { shotId: shot.id, projectId, kind });
                useStudioJobs.getState().upsertActive({
                    job_id: jobId, type: kind, kind, shot_id: shot.id, status: 'queued',
                    stage: 'queued', progress: 0, status_message: 'Queued…',
                });
                updateShot(shot.id, { lastJobId: jobId, statusMessage: 'Queued…' });
                addToast('Shot queued', 'success');
                return jobId;
            }

            const response = await fetch(apiUrl('/generate/advanced'), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    project_id: projectId,
                    shot_config: {
                        id: shot.id,
                        prompt: shot.prompt,
                        negative_prompt: shot.negativePrompt,
                        seed: shot.seed,
                        width: shot.width,
                        height: shot.height,
                        fps: shot.fps || 25,
                        num_frames: shot.numFrames,
                        num_inference_steps: shot.numInferenceSteps ?? 40,
                        priority: shot.priority ?? 50,
                        cfg_scale: shot.cfgScale,
                        enhance_prompt: shot.enhancePrompt,
                        upscale: shot.upscale,
                        pipeline_override: shot.pipelineOverride,
                        auto_continue: !!shot.autoContinue,
                        timeline: shot.timeline.map((item) => ({
                            type: item.type,
                            path: item.path,
                            frame_index: item.frameIndex,
                            strength: item.strength,
                        })),
                    },
                }),
            });
            const data = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
            const jobId = data.job_id as string;
            registerStudioJob(jobId, { shotId: shot.id, projectId, kind });
            useStudioJobs.getState().upsertActive({
                job_id: jobId, type: kind, kind, shot_id: shot.id, status: 'queued',
                stage: 'queued', progress: 0, status_message: 'Queued…',
            });
            updateShot(shot.id, { lastJobId: jobId, statusMessage: 'Queued…' });
            addToast('Shot queued', 'success');
            return jobId;
        } catch (exc) {
            updateShot(shot.id, { isGenerating: false, statusMessage: 'Failed' });
            addToast(exc instanceof Error ? exc.message : 'Generation failed', 'error');
            throw exc;
        }
    }, [addToast, onDowngrade, updateShot]);
}

export function useStudioCancel() {
    const { addToast, updateShot } = useStudioSelection();
    return useCallback(async (shot: Shot) => {
        if (!shot.lastJobId) {
            updateShot(shot.id, { isGenerating: false, statusMessage: 'Cancelled', stage: 'cancelled' });
            return;
        }
        const jobId = shot.lastJobId;
        updateShot(shot.id, { statusMessage: 'Cancelling…', stage: 'cancelling' });
        await fetch(apiUrl(`/jobs/${jobId}/cancel`), { method: 'POST' });
        addToast('Cancellation requested', 'info');

        // Watchdog: if the SSE terminal event is missed, reconcile from /status.
        window.setTimeout(async () => {
            try {
                const status = await fetch(apiUrl(`/status/${jobId}`)).then((response) => response.json());
                if (status.status === 'cancelled') {
                    updateShot(shot.id, { isGenerating: false, statusMessage: 'Cancelled', stage: 'cancelled', progress: 0 });
                } else if (status.status === 'failed') {
                    updateShot(shot.id, { isGenerating: false, statusMessage: status.error || 'Failed', stage: 'failed', progress: 0 });
                } else if (status.status === 'completed') {
                    updateShot(shot.id, { isGenerating: false, statusMessage: 'Complete', stage: 'completed', progress: 100 });
                }
            } catch {
                updateShot(shot.id, { isGenerating: false, statusMessage: 'Cancelled', stage: 'cancelled' });
            }
        }, 5000);
    }, [addToast, updateShot]);
}

export function useStudioCancelJob() {
    const addToast = useStudioDocumentStore((state) => state.addToast);
    return useCallback(async (jobId: string, label = 'Job') => {
        try {
            const response = await fetch(apiUrl(`/jobs/${jobId}/cancel`), { method: 'POST' });
            const data = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(data.detail || 'Cancel failed');
            addToast(`${label} ${data.status || 'cancelled'}`, 'info');
            return true;
        } catch (error) {
            addToast(error instanceof Error ? error.message : 'Cancel failed', 'error');
            return false;
        }
    }, [addToast]);
}

export function useStudioCreateRender(projectId: string) {
    const { addToast } = useStudioSelection();
    const trackStates = useStudioDocumentStore((state) => state.trackStates);
    return useCallback(async (settings: {
        width: number;
        height: number;
        fps: number;
        start_frame?: number;
        end_frame?: number;
        fit?: 'pad' | 'crop';
        source_path?: string;
        crf?: number;
        preset?: string;
        audio_bitrate?: string;
        audio_normalize?: boolean;
        lut?: string;
    }) => {
        const { source_path, ...renderSettings } = settings;
        const response = await fetch(apiUrl(`/projects/${projectId}/render`), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                settings: renderSettings,
                track_states: trackStates,
                source_path,
            }),
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
            addToast(data.detail || 'Export failed to start', 'error');
            throw new Error(data.detail || `HTTP ${response.status}`);
        }
        addToast('Export queued', 'success');
        return data.job_id as string;
    }, [addToast, projectId, trackStates]);
}

export function useStudioBoardSync(projectId: string) {
    const {
        boardPositions, boardPinned, boardAreas, boardLinks, boardCamera,
        hydrateBoard, setCamera,
    } = useStudioEditor();
    const addToast = useStudioDocumentStore((state) => state.addToast);
    const loaded = useRef(false);
    const revision = useRef(0);
    const saveTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
    const viewTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

    useEffect(() => {
        let alive = true;
        loaded.current = false;
        Promise.all([
            fetch(apiUrl(`/projects/${projectId}/board`)).then((response) => response.ok ? response.json() : null),
            fetch(apiUrl(`/projects/${projectId}/view`)).then((response) => response.ok ? response.json() : null),
        ]).then(([board, view]) => {
            if (!alive || !board) return;
            revision.current = board.revision || 0;
            const nodes = Array.isArray(board.layout?.nodes) ? board.layout.nodes : [];
            const positions: Record<string, BoardPosition> = {};
            const pinned: Record<string, boolean> = {};
            for (const node of nodes) {
                if (!node?.id) continue;
                positions[node.id] = {
                    x: Number(node.x) || 0,
                    y: Number(node.y) || 0,
                    width: Number(node.width) || 300,
                    height: Number(node.height) || 230,
                };
                if (node.pinned) pinned[node.id] = true;
            }
            const areas: Record<string, BoardArea> = {};
            for (const area of Array.isArray(board.layout?.areas) ? board.layout.areas : []) {
                if (!area?.id) continue;
                areas[area.id] = {
                    id: area.id,
                    title: area.title || 'Area',
                    x: Number(area.x) || 0,
                    y: Number(area.y) || 0,
                    width: Number(area.width) || 480,
                    height: Number(area.height) || 320,
                    color: area.color || undefined,
                    collapsed: !!area.collapsed,
                };
            }
            const links: BoardLink[] = (Array.isArray(board.layout?.links) ? board.layout.links : [])
                .filter((link: BoardLink) => link?.source && link?.target)
                .map((link: BoardLink) => ({
                    id: link.id || `${link.source}-${link.target}`,
                    source: link.source,
                    target: link.target,
                    kind: link.kind || 'thread',
                }));
            const camera = view?.view?.camera as Camera | undefined;
            if (Object.keys(positions).length > 0 || Object.keys(areas).length > 0 || links.length > 0) {
                hydrateBoard(positions, pinned, camera, areas, links);
            } else if (camera) {
                setCamera(camera);
            }
            loaded.current = true;
        }).catch(() => {
            loaded.current = true;
        });
        return () => { alive = false; };
    }, [hydrateBoard, projectId, setCamera]);

    useEffect(() => {
        if (!loaded.current) return;
        if (saveTimer.current) clearTimeout(saveTimer.current);
        saveTimer.current = setTimeout(async () => {
            const nodes = Object.entries(boardPositions).map(([id, position]) => ({
                id,
                ...position,
                pinned: !!boardPinned[id],
            }));
            const areas = Object.values(boardAreas).map(({ id, title, x, y, width, height, color, collapsed }) => ({
                id, title, x, y, width, height, color: color || null, collapsed: !!collapsed,
            }));
            const links = boardLinks.map(({ id, source, target, kind }) => ({ id, source, target, kind }));
            try {
                const response = await fetch(apiUrl(`/projects/${projectId}/board`), {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ nodes, areas, links, revision: revision.current }),
                });
                if (response.status === 409) {
                    addToast('Board layout changed elsewhere; reload to merge.', 'error');
                    return;
                }
                const data = await response.json().catch(() => ({}));
                if (response.ok && typeof data.revision === 'number') revision.current = data.revision;
            } catch {
                addToast('Board layout save failed', 'error');
            }
        }, 700);
        return () => { if (saveTimer.current) clearTimeout(saveTimer.current); };
    }, [addToast, boardAreas, boardLinks, boardPinned, boardPositions, projectId]);

    useEffect(() => {
        if (!loaded.current) return;
        if (viewTimer.current) clearTimeout(viewTimer.current);
        viewTimer.current = setTimeout(() => {
            void fetch(apiUrl(`/projects/${projectId}/view`), {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ camera: boardCamera, selection: [], surface: 'story' }),
            }).catch(() => undefined);
        }, 1200);
        return () => { if (viewTimer.current) clearTimeout(viewTimer.current); };
    }, [boardCamera, projectId]);

}

export function useAssetUrl(path?: string | null) {
    return useMemo(() => getAssetUrl(path || undefined) || '', [path]);
}

export function formatDuration(frames: number, fps: number): string {
    const seconds = Math.max(0, frames) / Math.max(1, fps);
    const minutes = Math.floor(seconds / 60);
    const rest = seconds - minutes * 60;
    return `${minutes}:${rest.toFixed(1).padStart(4, '0')}`;
}
