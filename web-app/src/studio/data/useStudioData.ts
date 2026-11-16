import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useShallow } from 'zustand/react/shallow';
import { useTimelineStore, type Project, type Shot } from '../../stores/timelineStore';
import { apiUrl, getAssetUrl } from '../../config';
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
    created_at?: string | null;
    selected: boolean;
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

export function useStudioReview(projectId: string) {
    const addToast = useTimelineStore((state) => state.addToast);
    const [review, setReview] = useState<ReviewPayload>(EMPTY_REVIEW);
    const [loading, setLoading] = useState(true);

    const refresh = useCallback(async () => {
        try {
            const response = await fetch(apiUrl(`/projects/${projectId}/review`));
            const data = response.ok ? await response.json() : EMPTY_REVIEW;
            setReview({
                comments: Array.isArray(data.comments) ? data.comments : [],
                approvals: Array.isArray(data.approvals) ? data.approvals : [],
                summary: data.summary || EMPTY_REVIEW.summary,
            });
        } catch {
            setReview(EMPTY_REVIEW);
        } finally {
            setLoading(false);
        }
    }, [projectId]);

    useEffect(() => { void refresh(); }, [refresh]);

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

export function useStudioBoardSync(projectId: string) {
    const {
        boardPositions, boardPinned, boardAreas, boardLinks, boardCamera,
        hydrateBoard, setCamera,
    } = useStudioEditor();
    const addToast = useTimelineStore((state) => state.addToast);
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
            const areas = Object.values(boardAreas).map(({ id, title, x, y, width, height, color }) => ({
                id, title, x, y, width, height, color: color || null,
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
