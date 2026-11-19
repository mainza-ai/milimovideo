import type { StateCreator } from 'zustand';
import { API_BASE_URL } from '../../config';
import type { TimelineState, ProjectSlice, Project } from '../types';
import { v4 as uuidv4 } from 'uuid';
import { getAssetUrl } from '../../config';
import { documentSignature } from '../saveStatus';

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
            seedLocked: true,
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

let saveInFlight: Promise<void> | null = null;
let saveQueued = false;

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
    saveStatus: 'idle',
    saveError: null,
    savedBaseline: null,

    setProject: (p) => set({ project: p }),

    triggerAssetRefresh: () => set(state => ({ assetRefreshVersion: state.assetRefreshVersion + 1 })),

    saveProject: async () => {
        if (get().saveStatus === 'conflict') {
            get().addToast('Resolve the save conflict: reload or overwrite first.', 'error');
            return;
        }
        if (saveInFlight) {
            saveQueued = true;
            return saveInFlight;
        }
        const run = async () => {
        const { project, addToast } = get();
        set({ saveStatus: 'saving', saveError: null });
        try {
            // Convert to snake_case for backend
            const payload = {
                id: project.id,
                name: project.name,
                fps: project.fps,
                seed: project.seed,
                resolution_w: project.resolutionW,
                resolution_h: project.resolutionH,
                revision: project.revision,
                shots: project.shots.map((s, order) => ({
                    id: s.id,
                    prompt: s.prompt,
                    negative_prompt: s.negativePrompt,
                    seed: s.seed,
                    seed_locked: s.seedLocked ?? true,
                    width: s.width,
                    height: s.height,
                    fps: s.fps || project.fps,
                    num_frames: s.numFrames,
                    cfg_scale: s.cfgScale,
                    enhance_prompt: s.enhancePrompt,
                    upscale: s.upscale,
                    pipeline_override: s.pipelineOverride,
                    track_index: s.trackIndex,
                    start_frame: s.startFrame,
                    trim_in: s.trimIn,
                    trim_out: s.trimOut,
                    transition_in: s.transitionIn ?? 0,
                    audio_gain: s.audioGain ?? 1,
                    audio_muted: s.audioMuted ?? false,
                    keyframes: s.keyframes ?? [],
                    blocking: s.blocking ?? [],
                    speed: s.speed ?? 1,
                    audio_fade_in: s.audioFadeIn ?? 0,
                    audio_fade_out: s.audioFadeOut ?? 0,
                    auto_continue: s.autoContinue,
                    scene_id: s.sceneId,
                    index: s.index,
                    sequence_index: order,
                    shot_type: s.shotType,
                    action: s.action,
                    dialogue: s.dialogue,
                    character: s.character,
                    timeline: s.timeline.map(t => ({
                        type: t.type,
                        path: t.path,
                        frame_index: t.frameIndex,
                        strength: t.strength
                    })),
                    last_job_id: s.lastJobId,
                    selected_take_id: s.selectedTakeId,
                    favorite_take_id: s.favoriteTakeId,
                    thumbnail_url: s.thumbnailUrl,
                    video_url: s.videoUrl
                })),
                scenes: (project.scenes || []).map(sc => ({
                    id: sc.id,
                    index: sc.index,
                    name: sc.name,
                    script_content: sc.scriptContent
                })),
                script_content: project.scriptContent,
                captions_burn: project.captionsBurn ?? true,
                render_preset: project.renderPreset ?? null,
                render_crf: project.renderCrf ?? 18,
                render_speed: project.renderSpeed ?? 'fast',
                render_audio_bitrate: project.renderAudioBitrate ?? '192k',
                render_audio_normalize: project.renderAudioNormalize ?? false,
                render_lut: project.renderLut ?? null,
                titles: (project.titles || []).map(cue => ({
                    id: cue.id,
                    text: cue.text,
                    start_frame: cue.startFrame,
                    end_frame: cue.endFrame,
                    x: cue.x ?? 0.5,
                    y: cue.y ?? 0.5,
                    size: cue.size ?? 48,
                    color: cue.color ?? '#ffffff',
                    box: cue.box ?? true,
                })),
                captions: (project.captions || []).map(cue => ({
                    id: cue.id,
                    start_frame: cue.startFrame,
                    end_frame: cue.endFrame,
                    text: cue.text
                }))
            };

            const res = await fetch(`${API_BASE_URL}/projects/${project.id}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (res.ok) {
                const data = await res.json().catch(() => ({}));
                set(state => {
                    const project = {
                        ...state.project,
                        revision: typeof data.revision === "number" ? data.revision : state.project.revision,
                    };
                    return { project, saveStatus: 'saved' as const, savedBaseline: documentSignature(project) };
                });
                addToast("Project saved successfully", "success");
            } else {
                const err = await res.json().catch(() => ({}));
                console.error("Save failed", err);
                if (res.status === 409) {
                    set({ saveStatus: 'conflict', saveError: err.detail || 'Revision conflict' });
                    addToast("Save conflict: reload the project before saving again.", "error");
                } else {
                    set({ saveStatus: 'error', saveError: err.detail || `HTTP ${res.status}` });
                    addToast("Failed to save project", "error");
                }
            }
        } catch (e) {
            console.error("Save error", e);
            set({ saveStatus: 'error', saveError: String(e) });
            addToast("Failed to save project", "error");
        }
        };

        saveInFlight = run().finally(() => {
            saveInFlight = null;
            if (saveQueued) {
                saveQueued = false;
                void get().saveProject();
            }
        });
        return saveInFlight;
    },

    createNewProject: async (name: string, settings) => {
        const { addToast } = get();
        const defaults = { resolutionW: 768, resolutionH: 512, fps: 25, seed: 42 };
        const finalSettings = { ...defaults, ...settings };

        try {
            const res = await fetch(API_BASE_URL + '/projects', {
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
                seed: data.seed || finalSettings.seed,
                revision: typeof data.revision === "number" ? data.revision : 0,
            };

            set({
                project: newProject,
                selectedShotId: null,
                saveStatus: 'saved',
                savedBaseline: documentSignature(newProject),
                saveError: null,
            });
            addToast("New project created", "success");
        } catch (e) {
            console.error(e);
            addToast("Failed to create project", "error");
        }
    },

    loadProject: async (id: string) => {
        const { addToast } = get();
        try {
            const res = await fetch(`${API_BASE_URL}/projects/${id}`);
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
                revision: typeof data.revision === "number" ? data.revision : 0,
                shots: (data.shots || []).map((s: any) => ({
                    id: s.id,
                    prompt: s.prompt,
                    negativePrompt: s.negative_prompt,
                    seed: s.seed,
                    seedLocked: s.seed_locked ?? true,
                    width: s.width,
                    height: s.height,
                    fps: s.fps,
                    numFrames: s.num_frames,
                    numInferenceSteps: s.num_inference_steps ?? 40,
                    priority: s.priority ?? 50,
                    cfgScale: s.cfg_scale,
                    enhancePrompt: s.enhance_prompt ?? true,
                    upscale: s.upscale,
                    pipelineOverride: s.pipeline_override,
                    trackIndex: s.track_index || 0,
                    startFrame: s.start_frame || 0,
                    trimIn: s.trim_in || 0,
                    trimOut: s.trim_out || 0,
                    transitionIn: s.transition_in || 0,
                    audioGain: s.audio_gain ?? 1,
                    audioMuted: s.audio_muted ?? false,
                    keyframes: Array.isArray(s.keyframes) ? s.keyframes : [],
                    blocking: Array.isArray(s.blocking) ? s.blocking : [],
                    speed: s.speed ?? 1,
                    audioFadeIn: s.audio_fade_in ?? 0,
                    audioFadeOut: s.audio_fade_out ?? 0,
                    autoContinue: s.auto_continue,
                    sceneId: s.scene_id,
                    index: s.index,
                    shotType: s.shot_type,
                    action: s.action,
                    dialogue: s.dialogue,
                    character: s.character,
                    status: s.status === 'generating' ? 'pending' : (s.status || (s.video_url ? 'completed' : 'pending')),
                    isGenerating: false,
                    progress: undefined,
                    stage: undefined,
                    statusMessage: undefined,
                    etaSeconds: undefined,
                    timeline: (typeof s.timeline === "string"
                        ? (() => { try { return JSON.parse(s.timeline || "[]"); } catch { return []; } })()
                        : (s.timeline || [])
                    ).map((t: any) => ({
                        id: uuidv4(), // Generate temporary UI IDs
                        type: t.type,
                        path: t.path,
                        frameIndex: t.frame_index ?? t.frameIndex ?? 0,
                        strength: t.strength ?? 1.0
                    })),
                    lastJobId: s.last_job_id,
                    selectedTakeId: s.selected_take_id,
                    favoriteTakeId: s.favorite_take_id,
                    videoUrl: s.video_url ? getAssetUrl(s.video_url) : undefined,  // Use DB value
                    thumbnailUrl: getAssetUrl(s.thumbnail_url),
                    enhancedPromptResult: s.enhanced_prompt_result, // Map from backend
                    matchedElements: s.matched_elements ? (typeof s.matched_elements === 'string' ? JSON.parse(s.matched_elements) : s.matched_elements) : undefined,
                })),
                scriptContent: data.script_content, // Load persisted script
                captionsBurn: data.captions_burn ?? true,
                renderPreset: data.render_preset || undefined,
                renderCrf: data.render_crf ?? 18,
                renderSpeed: data.render_speed || 'fast',
                renderAudioBitrate: data.render_audio_bitrate || '192k',
                renderAudioNormalize: data.render_audio_normalize ?? false,
                renderLut: data.render_lut || undefined,
                titles: (data.titles || []).map((cue: any) => ({
                    id: cue.id || uuidv4(),
                    text: cue.text ?? '',
                    startFrame: cue.start_frame ?? 0,
                    endFrame: cue.end_frame ?? 0,
                    x: cue.x ?? 0.5,
                    y: cue.y ?? 0.5,
                    size: cue.size ?? 48,
                    color: cue.color ?? '#ffffff',
                    box: cue.box ?? true,
                })),
                captions: (data.captions || []).map((cue: any) => ({
                    id: cue.id || uuidv4(),
                    startFrame: cue.start_frame ?? 0,
                    endFrame: cue.end_frame ?? 0,
                    text: cue.text ?? '',
                })),
                scenes: (data.scenes || []).map((sc: any) => ({
                    id: sc.id,
                    index: sc.index,
                    name: sc.name,
                    scriptContent: sc.script_content,
                    shots: [] // Will be populated by store if needed, or we rely on ID link
                }))
            };

            set({
                project: loadedProject,
                selectedShotId: loadedProject.shots[0]?.id || null,
                saveStatus: 'saved',
                savedBaseline: documentSignature(loadedProject),
                saveError: null,
            });
            saveLastProjectId(id);  // Remember last opened project

            // Hydrate a job that was already running before this reload. Without
            // this the shot shows Pending until the next SSE stage event.
            try {
                const active = await fetch(`${API_BASE_URL}/projects/${id}/active_jobs`)
                    .then((response) => response.ok ? response.json() : [])
                    .catch(() => []);
                if (Array.isArray(active)) {
                    for (const job of active) {
                        if (!job?.shot_id || !['running', 'queued'].includes(job.status)) continue;
                        get().updateShot(job.shot_id, {
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
            } catch {
                // A missing active-jobs endpoint must not block project load.
            }
            addToast("Project loaded", "success");
        } catch (e) {
            console.error("Load failed for project:", id, e);
            addToast("Failed to load project", "error");

            // If load fails (e.g. 404), likely project was deleted or invalid.
            saveLastProjectId("");
            set({ project: DEFAULT_PROJECT, saveStatus: 'idle', savedBaseline: null });
        }
    },

    overwriteProject: async () => {
        const { project, addToast } = get();
        try {
            const response = await fetch(`${API_BASE_URL}/projects/${project.id}`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const server = await response.json();
            set(state => ({
                project: {
                    ...state.project,
                    revision: typeof server.revision === 'number' ? server.revision : state.project.revision,
                },
                saveStatus: 'dirty',
                saveError: null,
            }));
            await get().saveProject();
        } catch (error) {
            addToast(error instanceof Error ? error.message : 'Could not overwrite the server copy', 'error');
        }
    },

    deleteProject: async (id: string) => {
        const { addToast, project } = get();
        try {
            await fetch(`${API_BASE_URL}/projects/${id}`, { method: 'DELETE' });
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
