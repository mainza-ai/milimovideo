import type { StateCreator } from 'zustand';
import type { TimelineState, ProjectSlice, Project } from '../types';
import { v4 as uuidv4 } from 'uuid';
import { getAssetUrl } from '../../config';

const DEFAULT_NEGATIVE_PROMPT = "low quality, worst quality, inconsistent motion, blurry, jittery, distorted, watermark, text, static, freeze, loop, pause, still image, motionless";

export const DEFAULT_PROJECT: Project = {
    id: 'default',
    name: 'Untitled Project',
    shots: [
        {
            id: 'shot-init', // Static ID for init
            prompt: "A cinematic shot...",
            negativePrompt: DEFAULT_NEGATIVE_PROMPT,
            seed: 42,
            width: 768,
            height: 512,
            numFrames: 121,
            // fps inherited from project
            timeline: [],
            cfgScale: 3.0,
            enhancePrompt: true,
            upscale: true,
            pipelineOverride: 'auto',
            trackIndex: 0,
            startFrame: 0,
            trimIn: 0,
            trimOut: 0
        }
    ],
    fps: 25,
    resolutionW: 768,
    resolutionH: 512,
    seed: 42
};

const LAST_PROJECT_KEY = 'milimo_last_project_id';
const saveLastProjectId = (projectId: string) => {
    try {
        localStorage.setItem(LAST_PROJECT_KEY, projectId);
    } catch (e) {
        console.warn('Failed to save last project ID:', e);
    }
};

export const createProjectSlice: StateCreator<TimelineState, [], [], ProjectSlice> = (set, get) => ({
    project: {
        ...DEFAULT_PROJECT,
        shots: [
            {
                ...DEFAULT_PROJECT.shots[0],
                negativePrompt: DEFAULT_NEGATIVE_PROMPT
            }
        ]
    },
    assetRefreshVersion: 0,

    setProject: (p) => set({ project: p }),

    triggerAssetRefresh: () => set(state => ({ assetRefreshVersion: state.assetRefreshVersion + 1 })),

    saveProject: async () => {
        const { project, addToast } = get();
        try {
            // Convert to snake_case for backend
            const payload = {
                id: project.id,
                name: project.name,
                fps: project.fps,
                seed: project.seed,
                resolution_w: project.resolutionW,
                resolution_h: project.resolutionH,
                shots: project.shots.map(s => ({
                    id: s.id,
                    prompt: s.prompt,
                    negative_prompt: s.negativePrompt,
                    seed: s.seed,
                    width: s.width,
                    num_frames: s.numFrames,
                    cfg_scale: s.cfgScale,
                    enhance_prompt: s.enhancePrompt,
                    upscale: s.upscale,
                    pipeline_override: s.pipelineOverride,
                    track_index: s.trackIndex,
                    start_frame: s.startFrame,
                    trim_in: s.trimIn,
                    trim_out: s.trimOut,
                    auto_continue: s.autoContinue,
                    scene_id: s.sceneId,
                    action: s.action,
                    dialogue: s.dialogue,
                    character: s.character,
                    timeline: s.timeline.map(t => ({
                        type: t.type,
                        path: t.path,
                        frameIndex: t.frameIndex,
                        strength: t.strength
                    })),
                    last_job_id: s.lastJobId,
                    thumbnail_url: s.thumbnailUrl,
                    video_url: s.videoUrl
                })),
                scenes: (project.scenes || []).map(sc => ({
                    id: sc.id,
                    index: sc.index,
                    name: sc.name,
                    script_content: sc.scriptContent
                }))
            };

            const res = await fetch(`http://localhost:8000/projects/${project.id}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (res.ok) addToast("Project saved successfully", "success");
            else {
                const err = await res.json();
                console.error("Save failed", err);
                addToast("Failed to save project", "error");
            }
        } catch (e) {
            console.error("Save error", e);
            addToast("Failed to save project", "error");
        }
    },

    createNewProject: async (name: string, settings) => {
        const { addToast } = get();
        const defaults = { resolutionW: 768, resolutionH: 512, fps: 25, seed: 42 };
        const finalSettings = { ...defaults, ...settings };

        try {
            const res = await fetch('http://localhost:8000/projects', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    name,
                    resolution_w: finalSettings.resolutionW,
                    resolution_h: finalSettings.resolutionH,
                    fps: finalSettings.fps,
                    seed: finalSettings.seed
                })
            });
            const data = await res.json();

            // Convert back to store format
            const newProject: Project = {
                id: data.id,
                name: data.name,
                shots: [], // New project has no shots
                fps: data.fps || finalSettings.fps,
                resolutionW: data.resolution_w || finalSettings.resolutionW,
                resolutionH: data.resolution_h || finalSettings.resolutionH,
                seed: data.seed || finalSettings.seed
            };

            set({ project: newProject, selectedShotId: null });
            addToast("New project created", "success");
        } catch (e) {
            console.error(e);
            addToast("Failed to create project", "error");
        }
    },

    loadProject: async (id: string) => {
        const { addToast } = get();
        try {
            const res = await fetch(`http://localhost:8000/projects/${id}`);
            if (!res.ok) throw new Error("Load failed");

            const data = await res.json();
            // Map Backend snake_case to Frontend Store
            const loadedProject: Project = {
                id: data.id,
                name: data.name,
                fps: data.fps || 25,
                resolutionW: data.resolution_w || 768,
                resolutionH: data.resolution_h || 512,
                seed: data.seed || 42,
                shots: (data.shots || []).map((s: any) => ({
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
                    autoContinue: s.auto_continue,
                    sceneId: s.scene_id,
                    action: s.action,
                    dialogue: s.dialogue,
                    character: s.character,
                    timeline: (s.timeline || []).map((t: any) => ({
                        id: uuidv4(), // Generate temporary UI IDs
                        type: t.type,
                        path: t.path,
                        frameIndex: t.frame_index,
                        strength: t.strength
                    })),
                    lastJobId: s.last_job_id,
                    videoUrl: s.video_url ? getAssetUrl(s.video_url) : undefined,  // Use DB value
                    thumbnailUrl: getAssetUrl(s.thumbnail_url),
                    enhancedPromptResult: s.enhanced_prompt_result, // Map from backend
                    matchedElements: s.matched_elements ? (typeof s.matched_elements === 'string' ? JSON.parse(s.matched_elements) : s.matched_elements) : undefined,
                })),
                scriptContent: data.script_content, // Load persisted script
                scenes: (data.scenes || []).map((sc: any) => ({
                    id: sc.id,
                    index: sc.index,
                    name: sc.name,
                    scriptContent: sc.script_content,
                    shots: [] // Will be populated by store if needed, or we rely on ID link
                }))
            };

            set({ project: loadedProject, selectedShotId: loadedProject.shots[0]?.id || null });
            saveLastProjectId(id);  // Remember last opened project
            addToast("Project loaded", "success");
        } catch (e) {
            console.error("Load failed for project:", id, e);
            addToast("Failed to load project", "error");

            // If load fails (e.g. 404), likely project was deleted or invalid.
            saveLastProjectId("");
            set({ project: DEFAULT_PROJECT });
        }
    },

    deleteProject: async (id: string) => {
        const { addToast, project } = get();
        try {
            await fetch(`http://localhost:8000/projects/${id}`, { method: 'DELETE' });
            addToast("Project deleted", "success");

            if (project.id === id) {
                // Reset to default project instead of reloading
                const defaultProj = JSON.parse(JSON.stringify(DEFAULT_PROJECT));
                defaultProj.id = uuidv4(); // New ID for new project
                defaultProj.shots[0].id = uuidv4();

                set({
                    project: defaultProj,
                    selectedShotId: defaultProj.shots[0].id
                });
            }
        } catch (e) {
            console.error(e);
            addToast("Failed to delete", "error");
        }
    },
});
