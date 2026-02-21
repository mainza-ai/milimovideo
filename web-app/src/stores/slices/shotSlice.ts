import type { StateCreator } from 'zustand';
import type { TimelineState, ShotSlice, Shot } from '../types';
import { v4 as uuidv4 } from 'uuid';
import { getAssetUrl } from '../../config';

const DEFAULT_NEGATIVE_PROMPT = "low quality, worst quality, inconsistent motion, blurry, jittery, distorted, watermark, text, static, freeze, loop, pause, still image, motionless";

export const createShotSlice: StateCreator<TimelineState, [], [], ShotSlice> = (set, get) => ({
    getShotStartTime: (shotId: string) => {
        const { project } = get();
        let time = 0;
        for (const shot of project.shots) {
            if (shot.id === shotId) return time;
            time += shot.numFrames / (project.fps || 25);
        }
        return 0; // Fallback or if not found
    },

    addShot: (config: Partial<Shot> = {}) => set((state) => {
        const newShot: Shot = {
            id: uuidv4(),
            prompt: "A cinematic shot...",
            negativePrompt: DEFAULT_NEGATIVE_PROMPT,
            seed: state.project.seed,
            width: state.project.resolutionW,
            height: state.project.resolutionH,
            numFrames: 121,
            // fps inherited from project
            timeline: [],

            // Defaults
            cfgScale: 3.0,
            enhancePrompt: true,
            upscale: false,
            pipelineOverride: 'auto',
            trackIndex: 0,
            trimIn: 0,
            trimOut: 0,

            // Overrides
            ...config
        };
        return {
            project: {
                ...state.project,
                shots: [...state.project.shots, newShot]
            },
            selectedShotId: newShot.id
        };
    }),

    updateShot: (id, updates) => set((state) => ({
        project: {
            ...state.project,
            shots: state.project.shots.map(s => s.id === id ? { ...s, ...updates } : s)
        }
    })),

    patchShot: async (id, updates) => {
        const { updateShot, addToast } = get();
        // 1. Optimistic Update
        updateShot(id, updates);

        try {
            // 2. Map camelCase to snake_case for Backend
            const snakeUpdates: any = {};
            for (const [k, v] of Object.entries(updates)) {
                if (k === 'negativePrompt') snakeUpdates['negative_prompt'] = v;
                else if (k === 'numFrames') snakeUpdates['num_frames'] = v;
                else if (k === 'cfgScale') snakeUpdates['cfg_scale'] = v;
                else if (k === 'enhancePrompt') snakeUpdates['enhance_prompt'] = v;
                else if (k === 'lastJobId') snakeUpdates['last_job_id'] = v;
                else if (k === 'videoUrl') snakeUpdates['video_url'] = v;
                else if (k === 'thumbnailUrl') snakeUpdates['thumbnail_url'] = v;
                else if (k === 'pipelineOverride') snakeUpdates['pipeline_override'] = v;
                else if (k === 'trackIndex') snakeUpdates['track_index'] = v;
                else if (k === 'startFrame') snakeUpdates['start_frame'] = v;
                else if (k === 'trimIn') snakeUpdates['trim_in'] = v;
                else if (k === 'trimOut') snakeUpdates['trim_out'] = v;
                else if (k === 'autoContinue') snakeUpdates['auto_continue'] = v;
                else if (k === 'sceneId') snakeUpdates['scene_id'] = v;
                // Simple fields that match but better be explicit or just fall through if key is same (action, dialogue, character are same)
                else snakeUpdates[k] = v;
            }

            // 3. Send Network Request
            await fetch(`http://localhost:8000/shots/${id}`, {
                method: 'PATCH',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(snakeUpdates)
            });

        } catch (e) {
            console.error("Patch failed", e);
            addToast("Failed to save changes", "error");
        }
    },

    splitShot: async (id, splitFrame) => {
        const { addToast, project } = get();
        const shot = project.shots.find(s => s.id === id);
        if (!shot) return;

        try {
            const res = await fetch(`http://localhost:8000/shots/${id}/split`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ split_frame: splitFrame })
            });

            if (!res.ok) {
                throw new Error("Split failed");
            }

            const data = await res.json();
            const { original_shot, new_shot } = data;

            // Map backend to frontend
            const mapShot = (s: any): Shot => ({
                id: s.id,
                prompt: s.prompt,
                negativePrompt: s.negative_prompt,
                seed: s.seed,
                width: s.width,
                height: s.height,
                numFrames: s.num_frames,
                cfgScale: s.cfg_scale,
                enhancePrompt: s.enhance_prompt ?? true,
                upscale: s.upscale,
                pipelineOverride: s.pipeline_override,
                trackIndex: s.track_index || 0,
                startFrame: s.start_frame || 0,
                trimIn: s.trim_in || 0,
                trimOut: s.trim_out || 0,
                timeline: (s.timeline || []).map((t: any) => ({
                    id: uuidv4(),
                    type: t.type,
                    path: t.path,
                    frameIndex: t.frame_index,
                    strength: t.strength
                })),
                lastJobId: s.last_job_id,
                videoUrl: s.video_url ? getAssetUrl(s.video_url) : undefined,
                thumbnailUrl: getAssetUrl(s.thumbnail_url),
                enhancedPromptResult: s.enhanced_prompt_result
            });

            const originalUpdated = mapShot(original_shot);
            const newSplitShot = mapShot(new_shot);

            set(state => {
                const shots = [...state.project.shots];
                const index = shots.findIndex(s => s.id === id);
                if (index !== -1) {
                    shots[index] = originalUpdated;
                    shots.splice(index + 1, 0, newSplitShot);
                }
                return {
                    project: { ...state.project, shots }
                };
            });

            addToast("Shot split successfully", "success");

        } catch (e) {
            console.error("Split error", e);
            addToast("Failed to split shot", "error");
        }
    },

    reorderShots: (fromIndex, toIndex) => set((state) => {
        const shots = [...state.project.shots];
        const [removed] = shots.splice(fromIndex, 1);
        shots.splice(toIndex, 0, removed);
        return {
            project: {
                ...state.project,
                shots
            }
        };
    }),

    deleteShot: (id) => set((state) => ({
        project: {
            ...state.project,
            shots: state.project.shots.filter(s => s.id !== id)
        },
        selectedShotId: state.selectedShotId === id ? null : state.selectedShotId
    })),

    moveShotToValues: async (id, trackIndex, startFrame) => {
        const { updateShot } = get();
        // Optimistic
        updateShot(id, { trackIndex, startFrame });

        try {
            await get().patchShot(id, { trackIndex, startFrame });
        } catch (e) {
            console.error("Failed to move shot", e);
            get().addToast("Failed to move shot", "error");
        }
    },

    addConditioningToShot: (shotId, item) => set((state) => ({
        project: {
            ...state.project,
            shots: state.project.shots.map(s => {
                if (s.id !== shotId) return s;
                return {
                    ...s,
                    timeline: [...s.timeline, { ...item, id: uuidv4() }]
                };
            })
        }
    })),

    updateConditioning: (shotId, itemId, updates) => set((state) => ({
        project: {
            ...state.project,
            shots: state.project.shots.map(s => {
                if (s.id !== shotId) return s;
                return {
                    ...s,
                    timeline: s.timeline.map(ti => ti.id === itemId ? { ...ti, ...updates } : ti)
                };
            })
        }
    })),

    removeConditioning: (shotId, itemId) => set((state) => ({
        project: {
            ...state.project,
            shots: state.project.shots.map(s => {
                if (s.id !== shotId) return s;
                return {
                    ...s,
                    timeline: s.timeline.filter(ti => ti.id !== itemId)
                };
            })
        }
    })),

    generateShot: async (shotId: string) => {
        const { addToast, updateShot } = get();
        try {
            // Optimistic update
            updateShot(shotId, { isGenerating: true, statusMessage: "Queued..." });

            const res = await fetch(`http://localhost:8000/shots/${shotId}/generate`, {
                method: 'POST'
            });

            if (!res.ok) throw new Error("Gen failed");
            const data = await res.json();
            addToast(`Shot generation started (Job ${data.job_id})`, "success");

            updateShot(shotId, { statusMessage: "Generating...", lastJobId: data.job_id });

        } catch (e) {
            console.error("Generate failed", e);
            updateShot(shotId, { isGenerating: false, statusMessage: "Failed" });
            addToast("Failed to start generation", "error");
        }
    },

    inpaintShot: async (shotId, frameDataUrl, maskDataUrl, prompt) => {
        const { project, addToast, updateShot } = get();
        const shot = project.shots.find(s => s.id === shotId);
        if (!shot) return;

        try {
            addToast("Uploading assets for in-painting...", "info");

            // 1. Upload Frame
            const frameBlob = await (await fetch(frameDataUrl)).blob();
            const frameFile = new File([frameBlob], "edit_frame.jpg", { type: "image/jpeg" });
            const frameForm = new FormData();
            frameForm.append('file', frameFile);
            frameForm.append('project_id', project.id);

            const frameRes = await fetch('http://localhost:8000/upload', {
                method: 'POST',
                body: frameForm
            });
            const frameData = await frameRes.json();
            const imagePath = frameData.access_path; // Absolute path on server

            // 2. Upload Mask
            const maskBlob = await (await fetch(maskDataUrl)).blob();
            const maskFile = new File([maskBlob], "edit_mask.png", { type: "image/png" });
            const maskForm = new FormData();
            maskForm.append('file', maskFile);
            maskForm.append('project_id', project.id);

            const maskRes = await fetch('http://localhost:8000/upload', {
                method: 'POST',
                body: maskForm
            });
            const maskData = await maskRes.json();
            const maskPath = maskData.access_path;

            // 3. Trigger In-Painting
            addToast("Starting generation...", "info");
            // Generate a temporary Job ID for tracking
            const jobId = `job_${uuidv4().substring(0, 8)}`;

            // Optimistic update
            updateShot(shotId, {
                isGenerating: true,
                statusMessage: "In-Painting...",
                lastJobId: jobId
            });

            const inpaintRes = await fetch(`http://localhost:8000/edit/inpaint?job_id=${jobId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image_path: imagePath,
                    mask_path: maskPath,
                    prompt: prompt
                })
            });

            if (!inpaintRes.ok) throw new Error("Inpaint failed");

            // Fire-and-forget: SSE handles progress + completion via handleServerEvent
            addToast("In-painting started", "success");

        } catch (e) {
            console.error("Inpaint error", e);
            addToast("In-painting failed", "error");
            updateShot(shotId, { isGenerating: false, statusMessage: "Failed" });
        }
    },

    cancelShotGeneration: async (shotId: string) => {
        const { project, updateShot, addToast } = get();
        const shot = project.shots.find(s => s.id === shotId);

        if (!shot) return;

        if (shot.lastJobId) {
            try {
                // Optimistic UI state just turns to "Cancelling...", it leaves isGenerating=true 
                // until the backend `cancelled` SSE broadcast actually unlocks it.
                updateShot(shotId, { statusMessage: "Cancelling..." });
                await fetch(`http://localhost:8000/jobs/${shot.lastJobId}/cancel`, { method: 'POST' });
                addToast("Cancellation requested.", "info");
            } catch (e) {
                console.error("Failed to cancel job", e);
            }
        } else {
            // If there's no job ID yet, just force unlock
            updateShot(shotId, {
                isGenerating: false,
                statusMessage: "Cancelled"
            });
        }
    },

    // ── Storyboard Operations ──────────────────────────────────────

    batchGenerateShots: async (shotIds: string[]) => {
        const { project, addToast, updateShot } = get();

        // Optimistic: mark all as generating
        for (const id of shotIds) {
            updateShot(id, { isGenerating: true, statusMessage: "Queued..." });
        }

        try {
            const res = await fetch(`http://localhost:8000/projects/${project.id}/storyboard/batch-generate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ shot_ids: shotIds })
            });

            if (!res.ok) throw new Error("Batch generate failed");

            const data = await res.json();
            addToast(`Batch generation started for ${data.results?.length || shotIds.length} shots`, "success");

            // Update each with its job ID
            for (const result of (data.results || [])) {
                if (result.status === 'queued') {
                    updateShot(result.shot_id, { statusMessage: "Generating...", lastJobId: result.job_id });
                } else if (result.status === 'error') {
                    updateShot(result.shot_id, { isGenerating: false, statusMessage: "Failed" });
                }
            }
        } catch (e) {
            console.error("Batch generate failed", e);
            addToast("Batch generation failed", "error");
            for (const id of shotIds) {
                updateShot(id, { isGenerating: false, statusMessage: "Failed" });
            }
        }
    },

    reorderShotsInScene: async (sceneId: string, shotIds: string[]) => {
        const { project, addToast } = get();

        // Optimistic reorder
        set(state => {
            const reordered = shotIds.map((id, idx) => {
                const shot = state.project.shots.find(s => s.id === id);
                return shot ? { ...shot, index: idx } : null;
            }).filter(Boolean) as Shot[];

            const otherShots = state.project.shots.filter(s => !shotIds.includes(s.id));

            return {
                project: {
                    ...state.project,
                    shots: [...otherShots, ...reordered]
                }
            };
        });

        try {
            await fetch(`http://localhost:8000/projects/${project.id}/storyboard/shots/reorder`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ scene_id: sceneId, shot_ids: shotIds })
            });
        } catch (e) {
            console.error("Reorder failed", e);
            addToast("Failed to reorder shots", "error");
        }
    },

    addShotToScene: async (sceneId: string, data?: { action?: string; dialogue?: string; character?: string; shotType?: string }) => {
        const { project, addToast, loadProject } = get();

        try {
            const res = await fetch(`http://localhost:8000/projects/${project.id}/storyboard/shots/add`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    scene_id: sceneId,
                    action: data?.action || "A cinematic shot...",
                    dialogue: data?.dialogue || null,
                    character: data?.character || null,
                    shot_type: data?.shotType || "medium"
                })
            });

            if (!res.ok) throw new Error("Add shot failed");
            addToast("Shot added", "success");
            await loadProject(project.id);
        } catch (e) {
            console.error("Add shot failed", e);
            addToast("Failed to add shot", "error");
        }
    },

    deleteShotFromStoryboard: async (shotId: string) => {
        const { project, addToast } = get();

        // Optimistic removal
        set(state => ({
            project: {
                ...state.project,
                shots: state.project.shots.filter(s => s.id !== shotId)
            },
            selectedShotId: state.selectedShotId === shotId ? null : state.selectedShotId
        }));

        try {
            const res = await fetch(`http://localhost:8000/projects/${project.id}/storyboard/shots/${shotId}`, {
                method: 'DELETE'
            });

            if (!res.ok) throw new Error("Delete failed");
            addToast("Shot deleted", "info");
        } catch (e) {
            console.error("Delete shot failed", e);
            addToast("Failed to delete shot", "error");
        }
    },

    generateThumbnail: async (shotId: string, force?: boolean) => {
        const { project, addToast, updateShot } = get();
        try {
            const res = await fetch(`http://localhost:8000/projects/${project.id}/storyboard/generate-thumbnails`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ shot_ids: [shotId], force: !!force })
            });
            if (!res.ok) throw new Error("Thumbnail generation failed");

            const data = await res.json();
            const result = data.results.find((r: any) => r.shot_id === shotId);

            if (result && result.status === 'queued') {
                updateShot(shotId, {
                    isGenerating: true,
                    statusMessage: "Preparing...",
                    lastJobId: result.job_id,
                    videoUrl: force ? undefined : project.shots.find(s => s.id === shotId)?.videoUrl
                });
                addToast("Generating concept art...", "info");
            }
        } catch (e) {
            console.error("Thumbnail generation failed", e);
            addToast("Failed to generate thumbnail", "error");
        }
    },

    batchGenerateThumbnails: async (shotIds: string[]) => {
        const { project, addToast, updateShot } = get();
        try {
            const res = await fetch(`http://localhost:8000/projects/${project.id}/storyboard/generate-thumbnails`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ shot_ids: shotIds })
            });
            if (!res.ok) throw new Error("Batch thumbnail generation failed");
            const data = await res.json();

            let queuedCount = 0;
            (data.results || []).forEach((r: any) => {
                if (r.status === 'queued') {
                    queuedCount++;
                    updateShot(r.shot_id, {
                        isGenerating: true,
                        statusMessage: "Queued",
                        lastJobId: r.job_id
                    });
                }
            });

            if (queuedCount > 0) {
                addToast(`Generating ${queuedCount} concept art thumbnails...`, "info");
            }
        } catch (e) {
            console.error("Batch thumbnail generation failed", e);
            addToast("Failed to generate thumbnails", "error");
        }
    },

    pushStoryboardToTimeline: () => {
        const { project, addToast, updateShot } = get();

        // Collect all completed shots (with video) in scene/index order
        const scenes = project.scenes || [];
        const orderedShots: Shot[] = [];

        for (const scene of scenes) {
            const sceneShots = project.shots
                .filter(s => s.sceneId === scene.id && s.videoUrl)
                .sort((a, b) => (a.index ?? 0) - (b.index ?? 0));
            orderedShots.push(...sceneShots);
        }

        // Also include unassigned completed shots
        const assignedIds = new Set(orderedShots.map(s => s.id));
        const unassigned = project.shots
            .filter(s => !s.sceneId && s.videoUrl && !assignedIds.has(s.id));
        orderedShots.push(...unassigned);

        if (orderedShots.length === 0) {
            addToast("No completed shots to push to timeline", "info");
            return;
        }

        // Place sequentially on V1 (trackIndex 0)
        let currentFrame = 0;
        for (const shot of orderedShots) {
            updateShot(shot.id, {
                trackIndex: 0,
                startFrame: currentFrame,
                trimIn: 0,
                trimOut: shot.numFrames,
            });
            currentFrame += shot.numFrames;
        }

        addToast(`Placed ${orderedShots.length} shots on timeline`, "success");
    },
});

