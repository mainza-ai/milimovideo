import { create } from 'zustand';
import { v4 as uuidv4 } from 'uuid';
import { persist } from 'zustand/middleware';
import { temporal } from 'zundo';

const LAST_PROJECT_KEY = 'milimo_last_project_id';

// Helper to save last project ID to localStorage
const saveLastProjectId = (projectId: string) => {
    try {
        localStorage.setItem(LAST_PROJECT_KEY, projectId);
    } catch (e) {
        console.warn('Failed to save last project ID:', e);
    }
};

// Helper to get last project ID from localStorage
export const getLastProjectId = (): string | null => {
    try {
        return localStorage.getItem(LAST_PROJECT_KEY);
    } catch (e) {
        console.warn('Failed to get last project ID:', e);
        return null;
    }
};

export type ConditioningType = 'image' | 'video';

export interface ConditioningItem {
    id: string; // Internal ID
    type: ConditioningType;
    path: string; // URL /uploads/...
    frameIndex: number;
    strength: number;
}

export interface StoryElement {
    id: string;
    project_id: string;
    name: string;
    triggerWord: string; // @Hero
    type: 'character' | 'location' | 'object';
    description: string;
    image_path?: string;
}

export interface Shot {
    id: string;
    prompt: string;
    negativePrompt: string;
    seed: number;
    width: number;
    height: number;
    numFrames: number;
    fps?: number; // Optional override, defaults to project FPS
    timeline: ConditioningItem[];

    // Advanced Params
    cfgScale: number;
    enhancePrompt: boolean;
    upscale: boolean;
    pipelineOverride: 'auto' | 'ti2vid' | 'ic_lora' | 'keyframe';
    autoContinue?: boolean;
    progress?: number;

    // Result
    lastJobId?: string;
    videoUrl?: string; // Derived from jobID
    thumbnailUrl?: string; // Static Preview
    enhancedPromptResult?: string; // Result from backend
    statusMessage?: string; // Real-time status text
    currentPrompt?: string; // Live evolving prompt during generation
    etaSeconds?: number; // Estimated time remaining

    // UI State
    isGenerating?: boolean;

    // Storyboard metadata
    sceneId?: string;
    index?: number;
    action?: string;
    dialogue?: string;
    character?: string;
}

export interface ShotConfig {
    id: string;
    prompt: string;
    negativePrompt: string;
    seed: number;
    width: number;
    height: number;
    numFrames: number;
    fps?: number; // Optional override, defaults to project FPS
    timeline: ConditioningItem[];

    // Advanced Params
    cfgScale: number;
    enhancePrompt: boolean;
    upscale: boolean;
    pipelineOverride: 'auto' | 'ti2vid' | 'ic_lora' | 'keyframe';
    autoContinue?: boolean; // New: Smart Continue Toggle
    progress?: number;

    // Result
    lastJobId?: string;
    videoUrl?: string; // Derived from jobID
    thumbnailUrl?: string; // Static Preview
    enhancedPromptResult?: string; // Result from backend
    statusMessage?: string; // Real-time status text
    currentPrompt?: string; // Live evolving prompt during generation
    etaSeconds?: number; // Estimated time remaining

    // UI State
    isGenerating?: boolean;

    // Storyboard metadata
    sceneId?: string;
    index?: number;
    action?: string;
    dialogue?: string;
    character?: string;
}

export interface Scene {
    id: string;
    index: number;
    name: string;
    scriptContent?: string;
    shots: Shot[]; // Frontend convenience: Nested shots
}

export interface ParsedShot {
    action: string;
    dialogue?: string;
    character?: string;
}

export interface ParsedScene {
    id?: string; // Optional (not in DB yet)
    header: string;
    content: string;
    shots: ParsedShot[];
}

export interface Project {
    id: string;
    name: string;
    shots: Shot[]; // Legacy flat list (still useful for timeline view)
    scenes?: Scene[]; // New Hierarchy
    fps: number;
    resolutionW: number;
    resolutionH: number;
    seed: number; // Global seed
}

interface TimelineState {
    project: Project;
    selectedShotId: string | null;
    currentTime: number; // In seconds
    isPlaying: boolean;

    // Actions
    setProject: (p: Project) => void;
    addShot: () => void;
    updateShot: (id: string, updates: Partial<Shot>) => void;
    reorderShots: (fromIndex: number, toIndex: number) => void;
    deleteShot: (id: string) => void;
    selectShot: (id: string | null) => void;

    addConditioningToShot: (shotId: string, item: Omit<ConditioningItem, 'id'>) => void;
    updateConditioning: (shotId: string, itemId: string, updates: Partial<ConditioningItem>) => void;
    removeConditioning: (shotId: string, itemId: string) => void;

    setCurrentTime: (t: number) => void;
    setIsPlaying: (p: boolean) => void;

    // Async Actions (placeholders for now, will connect to API)
    saveProject: () => Promise<void>;

    // Notifications
    toasts: { id: string; message: string; type: 'success' | 'error' | 'info' }[];
    addToast: (message: string, type?: 'success' | 'error' | 'info') => void;
    removeToast: (id: string) => void;

    // Asset Management
    assetRefreshVersion: number;
    triggerAssetRefresh: () => void;

    // Project Actions
    createNewProject: (name: string, settings?: { resolutionW: number; resolutionH: number; fps: number; seed: number }) => Promise<void>;
    loadProject: (id: string) => Promise<void>;
    deleteProject: (id: string) => Promise<void>;

    // Storyboard Actions
    parseScript: (text: string) => Promise<ParsedScene[]>;
    commitStoryboard: (scenes: ParsedScene[]) => Promise<void>;
    generateShot: (shotId: string) => Promise<void>;

    // Selectors
    getShotStartTime: (shotId: string) => number;

    // In-Painting
    isEditing: boolean;
    setEditing: (e: boolean) => void;
    inpaintShot: (shotId: string, frameDataUrl: string, maskDataUrl: string, prompt: string) => Promise<void>;

    // Elements
    elements: StoryElement[];
    generatingElementIds: Record<string, string>; // Maps elementId -> jobId (or "true" for legacy/optimistic)
    fetchElements: (projectId: string) => Promise<void>;
    createElement: (projectId: string, data: Partial<StoryElement>) => Promise<void>;
    deleteElement: (elementId: string) => Promise<void>;
    generateVisual: (elementId: string, promptOverride?: string, guidanceOverride?: number, enableAeOverride?: boolean) => Promise<void>;
    cancelElementGeneration: (elementId: string) => Promise<void>;

    // View Mode
    viewMode: 'timeline' | 'elements' | 'storyboard' | 'images';
    setViewMode: (mode: 'timeline' | 'elements' | 'storyboard' | 'images') => void;

    // SSE Handling
    handleServerEvent: (type: string, data: any) => void;
}

const DEFAULT_PROJECT: Project = {
    id: 'default',
    name: 'Untitled Project',
    shots: [
        {
            id: 'shot-init', // Static ID for init
            prompt: "A cinematic shot...",
            negativePrompt: "low quality, worst quality, inconsistent motion, blurry, jittery, distorted, watermark, text, static, freeze, loop, pause, still image, motionless",
            seed: 42,
            width: 768,
            height: 512,
            numFrames: 121,
            // fps inherited from project
            timeline: [],
            cfgScale: 3.0,
            enhancePrompt: true,
            upscale: true,
            pipelineOverride: 'auto'
        }
    ],
    fps: 25,
    resolutionW: 768,
    resolutionH: 512,
    seed: 42
};

const DEFAULT_NEGATIVE_PROMPT = "low quality, worst quality, inconsistent motion, blurry, jittery, distorted, watermark, text, static, freeze, loop, pause, still image, motionless";

import { getAssetUrl } from '../config';

export const useTimelineStore = create<TimelineState>()(
    temporal(
        persist(
            (set, get) => ({
                // ... (store implementation remains the same, just wrapped)
                project: {
                    ...DEFAULT_PROJECT,
                    shots: [
                        {
                            ...DEFAULT_PROJECT.shots[0],
                            negativePrompt: DEFAULT_NEGATIVE_PROMPT
                        }
                    ]
                },
                selectedShotId: 'shot-init', // Select by default
                currentTime: 0,
                isPlaying: false,
                isEditing: false,
                toasts: [],
                assetRefreshVersion: 0,
                viewMode: 'timeline',

                // Elements
                elements: [],
                generatingElementIds: {}, // Maps elementId -> jobId

                setProject: (p) => set({ project: p }),

                addToast: (message, type = 'info') => {
                    const id = uuidv4();
                    set(state => ({ toasts: [...state.toasts, { id, message, type }] }));
                    setTimeout(() => get().removeToast(id), 3000);
                },

                removeToast: (id) => set(state => ({ toasts: state.toasts.filter(t => t.id !== id) })),

                addShot: () => set((state) => {
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
                        upscale: true,
                        pipelineOverride: 'auto'
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

                selectShot: (id) => set({ selectedShotId: id }),

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

                setCurrentTime: (t) => set({ currentTime: t }),
                setIsPlaying: (p) => set({ isPlaying: p }),
                setEditing: (e) => set({ isEditing: e, isPlaying: !e ? get().isPlaying : false }), // Pause if editing

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

                        await inpaintRes.json();
                        // Backend returns { status: "queued", job_id: ... }

                        addToast("In-painting queued successfully", "success");

                    } catch (e) {
                        console.error("Inpaint error", e);
                        addToast("In-painting failed", "error");
                        updateShot(shotId, { isGenerating: false, statusMessage: "Failed" });
                    }
                },

                setViewMode: (mode) => set({ viewMode: mode }),

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
                                height: s.height,
                                num_frames: s.numFrames,
                                cfg_scale: s.cfgScale,
                                enhance_prompt: s.enhancePrompt,
                                upscale: s.upscale,
                                pipeline_override: s.pipelineOverride,
                                timeline: s.timeline.map(t => ({
                                    type: t.type,
                                    path: t.path,
                                    frameIndex: t.frameIndex,
                                    strength: t.strength
                                })),
                                last_job_id: s.lastJobId,
                                thumbnail_url: s.thumbnailUrl,
                                video_url: s.videoUrl
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

                // Project Manager (New)

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
                                enhancePrompt: s.enhance_prompt ?? true, // Default to true if not set
                                upscale: s.upscale,
                                pipelineOverride: s.pipeline_override,
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
                                enhancedPromptResult: s.enhanced_prompt_result // Map from backend
                            }))
                        };

                        set({ project: loadedProject, selectedShotId: loadedProject.shots[0]?.id || null });
                        saveLastProjectId(id);  // Remember last opened project
                        // useTimelineStore.getState().setEditing(false); // This line was commented out in the original, but the instruction implies it should be there. I'll add it.
                        addToast("Project loaded", "success");
                    } catch (e) {
                        console.error("Load failed for project:", id, e);
                        addToast("Failed to load project", "error");

                        // If load fails (e.g. 404), likely project was deleted or invalid.
                        // Reset defaults to avoid infinite reload loop on startup.
                        saveLastProjectId("");
                        set({ project: DEFAULT_PROJECT });
                    }
                },

                // --- Elements ---
                fetchElements: async (projectId) => {
                    try {
                        // The backend returns snake_case, frontend uses camelCase for triggerWord
                        const res = await fetch(`http://localhost:8000/projects/${projectId}/elements`);
                        const data = await res.json();
                        const elements = data.map((e: any) => ({
                            ...e,
                            triggerWord: e.trigger_word // map backend to frontend
                        }));
                        set({ elements });
                    } catch (e) {
                        console.error("Failed to fetch elements", e);
                    }
                },
                createElement: async (projectId, elementData) => {
                    try {
                        const payload = {
                            name: elementData.name,
                            type: elementData.type,
                            description: elementData.description,
                            trigger_word: elementData.triggerWord // backend expects snake_case
                        };
                        const res = await fetch(`http://localhost:8000/projects/${projectId}/elements`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(payload)
                        });
                        if (res.ok) {
                            // Refresh list
                            get().fetchElements(projectId);
                            get().addToast("Element created", "success");
                        }
                    } catch (e) {
                        get().addToast("Failed to create element", "error");
                    }
                },
                deleteElement: async (elementId) => {
                    try {
                        const res = await fetch(`http://localhost:8000/elements/${elementId}`, { method: 'DELETE' });
                        if (res.ok) {
                            const { elements } = get();
                            set({ elements: elements.filter(e => e.id !== elementId) });
                            get().addToast("Element deleted", "info");
                        }
                    } catch (e) { console.error(e); }
                },

                generateVisual: async (elementId, promptOverride, guidanceOverride = 2.0, enableAeOverride = false) => {
                    const { addToast, fetchElements, project } = get();

                    addToast("Queuing Visual Generation...", "info");
                    try {
                        const res = await fetch(`http://localhost:8000/elements/${elementId}/visualize`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                prompt_override: promptOverride,
                                guidance_scale: guidanceOverride,
                                enable_ae: enableAeOverride
                            })
                        });

                        if (!res.ok) throw new Error("Generation failed");

                        const data = await res.json();
                        const jobId = data.job_id;

                        // Set loading state with Job ID
                        set(state => ({
                            generatingElementIds: { ...state.generatingElementIds, [elementId]: jobId }
                        }));
                        addToast("Generation Queued", "success");

                        // Start Polling 
                        const poll = async () => {
                            try {
                                const statusRes = await fetch(`http://localhost:8000/status/${jobId}`);
                                if (!statusRes.ok) return; // Retry?
                                const status = await statusRes.json();

                                if (status.status === 'completed') {
                                    addToast("Visual Generated!", "success");
                                    await fetchElements(project.id);
                                    // Clear loading
                                    set(state => {
                                        const newMap = { ...state.generatingElementIds };
                                        delete newMap[elementId];
                                        return { generatingElementIds: newMap };
                                    });
                                } else if (status.status === 'failed' || status.status === 'cancelled') {
                                    addToast(`Generation ${status.status}: ${status.status_message || ''}`, "error");
                                    set(state => {
                                        const newMap = { ...state.generatingElementIds };
                                        delete newMap[elementId];
                                        return { generatingElementIds: newMap };
                                    });
                                } else {
                                    // Still running
                                    setTimeout(poll, 1000);
                                }
                            } catch (e) {
                                console.error("Poll error", e);
                                // Don't clear immediately on poll error, retry mostly?
                                setTimeout(poll, 2000);
                            }
                        };
                        poll();

                    } catch (e) {
                        console.error(e);
                        addToast("Failed to queue visual", "error");
                        set(state => {
                            const newMap = { ...state.generatingElementIds };
                            delete newMap[elementId];
                            return { generatingElementIds: newMap };
                        });
                    }
                },

                cancelElementGeneration: async (elementId) => {
                    const { generatingElementIds, addToast } = get();
                    const jobId = generatingElementIds[elementId];
                    if (!jobId || jobId === "true") return; // Can't cancel if no ID

                    try {
                        await fetch(`http://localhost:8000/jobs/${jobId}/cancel`, { method: 'POST' });
                        addToast("Cancelling generation...", "info");
                        // The poller will catch the 'cancelled' status and clean up
                    } catch (e) {
                        console.error(e);
                        addToast("Failed to cancel", "error");
                    }
                },

                deleteProject: async (id: string) => {
                    const { addToast, project } = get();
                    try {
                        await fetch(`http://localhost:8000/projects/${id}`, { method: 'DELETE' });
                        addToast("Project deleted", "success");

                        // If current project deleted, what do? Reload window or create default?
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

                triggerAssetRefresh: () => set(state => ({ assetRefreshVersion: state.assetRefreshVersion + 1 })),

                getShotStartTime: (shotId: string) => {
                    const { project } = get();
                    let time = 0;
                    for (const shot of project.shots) {
                        if (shot.id === shotId) return time;
                        time += shot.numFrames / (project.fps || 25);
                    }
                    return 0; // Fallback or if not found
                },

                // Storyboard Implementation
                parseScript: async (text: string) => {
                    try {
                        const project_id = get().project.id;
                        const res = await fetch(`http://localhost:8000/projects/${project_id}/script/parse`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ script_text: text })
                        });
                        const data = await res.json();
                        return data.scenes as ParsedScene[];
                    } catch (e) {
                        console.error("Parse failed", e);
                        get().addToast("Failed to parse script", "error");
                        return [];
                    }
                },

                commitStoryboard: async (scenes: ParsedScene[]) => {
                    const { project, addToast, loadProject } = get();
                    try {
                        const res = await fetch(`http://localhost:8000/projects/${project.id}/storyboard/commit`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ scenes: scenes })
                        });
                        if (!res.ok) throw new Error("Commit failed");

                        addToast("Storyboard saved!", "success");
                        // Reload project to get the new shots
                        await loadProject(project.id);
                    } catch (e) {
                        console.error("Commit failed", e);
                        addToast("Failed to commit storyboard", "error");
                    }
                },

                handleServerEvent: (type: string, data: any) => {
                    const { updateShot, triggerAssetRefresh, project, addToast } = get();

                    if (type === 'progress') {
                        const shot = project.shots.find(s => s.lastJobId === data.job_id);
                        if (shot) {
                            const updates: any = {
                                progress: data.progress,
                                isGenerating: true
                            };
                            if (data.message) updates.statusMessage = data.message;
                            if (data.status) updates.statusMessage = data.status === 'processing' ? (data.message || 'Processing...') : data.status;
                            updateShot(shot.id, updates);
                        }
                    } else if (type === 'complete') {
                        const shot = project.shots.find(s => s.lastJobId === data.job_id);
                        if (shot) {
                            updateShot(shot.id, {
                                isGenerating: false,
                                progress: 100,
                                videoUrl: getAssetUrl(data.url),
                                thumbnailUrl: getAssetUrl(data.thumbnail_url),
                                numFrames: data.actual_frames || shot.numFrames,
                                statusMessage: "Complete"
                            });
                            triggerAssetRefresh();
                            addToast("Generation Complete", "success");
                        }
                    } else if (type === 'error') {
                        const shot = project.shots.find(s => s.lastJobId === data.job_id);
                        if (shot) {
                            updateShot(shot.id, {
                                isGenerating: false,
                                statusMessage: "Failed",
                                progress: 0
                            });
                            addToast(`Generation Failed: ${data.message}`, "error");
                        }
                    }
                },
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
                }
            }),
            {
                name: 'milimo-timeline-storage',
                partialize: (state: TimelineState) => ({ project: state.project }),
                merge: (persistedState: any, currentState: TimelineState) => ({
                    ...currentState,
                    ...(persistedState as Partial<TimelineState>),
                    toasts: [] as { id: string; message: string; type: 'success' | 'error' | 'info' }[]
                }),
            }
        ),
        {
            limit: 20, // Limit history to 20 steps
            partialize: (state: TimelineState) => ({ project: state.project }),
            equality: (a: any, b: any) => JSON.stringify(a) === JSON.stringify(b)
        }
    ));
