import { useCallback, useEffect, useMemo, useState } from 'react';
import { useShallow } from 'zustand/react/shallow';
import { useTimelineStore, type Project, type Shot } from '../../stores/timelineStore';
import { apiUrl, getAssetUrl } from '../../config';

export interface StudioHealth {
    status: string;
    device: string;
    gpu: { busy: boolean; queued: number };
    slots: { active_slots: string[] };
    disk_free_bytes: number | null;
    auth_required: boolean;
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
    created_at: number;
}

export interface StudioRender {
    id: string;
    status: string;
    progress: number;
    output_path: string | null;
    error_message: string | null;
    created_at: string;
}

export function useStudioProject(): Project {
    return useTimelineStore((state) => state.project);
}

export function useStudioData() {
    const project = useStudioProject();
    const { health, error } = useStudioHealth();
    return { project, health, error };
}

export function useStudioShots(): Shot[] {
    return useTimelineStore(useShallow((state) => state.project.shots));
}

export function useStudioSelection() {
    return useTimelineStore(useShallow((state) => ({
        selectedShotId: state.selectedShotId,
        selectShot: state.selectShot,
        updateShot: state.updateShot,
        patchShot: state.patchShot,
        currentTime: state.currentTime,
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
    const [health, setHealth] = useState<StudioHealth | null>(null);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        let alive = true;
        const load = async () => {
            try {
                const response = await fetch(apiUrl('/health'));
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                const data = await response.json();
                if (alive) {
                    setHealth(data);
                    setError(null);
                }
            } catch (exc) {
                if (alive) setError(exc instanceof Error ? exc.message : String(exc));
            }
        };
        void load();
        const interval = window.setInterval(load, 10000);
        return () => {
            alive = false;
            window.clearInterval(interval);
        };
    }, []);

    return { health, error };
}

export function useStudioElements(projectId: string) {
    const [elements, setElements] = useState<StudioElement[]>([]);
    useEffect(() => {
        let alive = true;
        fetch(apiUrl(`/projects/${projectId}/elements`))
            .then((response) => response.ok ? response.json() : [])
            .then((data) => { if (alive) setElements(Array.isArray(data) ? data : []); })
            .catch(() => { if (alive) setElements([]); });
        return () => { alive = false; };
    }, [projectId]);
    return elements;
}

export function useStudioAssets() {
    const [assets, setAssets] = useState<StudioAsset[]>([]);
    const [loading, setLoading] = useState(true);
    const refresh = useCallback(async () => {
        setLoading(true);
        try {
            const response = await fetch(apiUrl('/uploads'));
            const data = response.ok ? await response.json() : [];
            setAssets(Array.isArray(data) ? data : []);
        } catch {
            setAssets([]);
        } finally {
            setLoading(false);
        }
    }, []);
    useEffect(() => { void refresh(); }, [refresh]);
    return { assets, loading, refresh };
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
        return () => window.clearInterval(interval);
    }, [refresh]);
    return { renders, refresh };
}

export function useStudioGenerate() {
    const { addToast, updateShot } = useStudioSelection();

    return useCallback(async (shot: Shot, randomizeSeed = false) => {
        updateShot(shot.id, { isGenerating: true, progress: 0, statusMessage: 'Queued…' });
        try {
            if (shot.sceneId) {
                const response = await fetch(
                    apiUrl(`/shots/${shot.id}/generate${randomizeSeed ? '?randomize_seed=true' : ''}`),
                    { method: 'POST' },
                );
                const data = await response.json().catch(() => ({}));
                if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
                updateShot(shot.id, { lastJobId: data.job_id, statusMessage: 'Queued…' });
                addToast('Shot queued', 'success');
                return data.job_id as string;
            }

            const projectId = useTimelineStore.getState().project.id;
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
                        num_inference_steps: 40,
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
            updateShot(shot.id, { lastJobId: data.job_id, statusMessage: 'Queued…' });
            addToast('Shot queued', 'success');
            return data.job_id as string;
        } catch (exc) {
            updateShot(shot.id, { isGenerating: false, statusMessage: 'Failed' });
            addToast(exc instanceof Error ? exc.message : 'Generation failed', 'error');
            throw exc;
        }
    }, [addToast, updateShot]);
}

export function useStudioCancel() {
    const { addToast, updateShot } = useStudioSelection();
    return useCallback(async (shot: Shot) => {
        if (!shot.lastJobId) {
            updateShot(shot.id, { isGenerating: false, statusMessage: 'Cancelled' });
            return;
        }
        updateShot(shot.id, { statusMessage: 'Cancelling…' });
        await fetch(apiUrl(`/jobs/${shot.lastJobId}/cancel`), { method: 'POST' });
        addToast('Cancellation requested', 'info');
    }, [addToast, updateShot]);
}

export function useStudioCreateRender(projectId: string) {
    const { addToast } = useStudioSelection();
    const trackStates = useTimelineStore((state) => state.trackStates);
    return useCallback(async (settings: { width: number; height: number; fps: number; crf?: number; preset?: string }) => {
        const response = await fetch(apiUrl(`/projects/${projectId}/render`), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ settings, track_states: trackStates }),
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

export function useAssetUrl(path?: string | null) {
    return useMemo(() => getAssetUrl(path || undefined) || '', [path]);
}

export function formatDuration(frames: number, fps: number): string {
    const seconds = Math.max(0, frames) / Math.max(1, fps);
    const minutes = Math.floor(seconds / 60);
    const rest = seconds - minutes * 60;
    return `${minutes}:${rest.toFixed(1).padStart(4, '0')}`;
}
