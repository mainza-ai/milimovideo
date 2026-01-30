import { create } from 'zustand';
import { v4 as uuidv4 } from 'uuid';
import { persist } from 'zustand/middleware';
import { temporal } from 'zundo';

export type ConditioningType = 'image' | 'video';

export interface ConditioningItem {
    id: string; // Internal ID
    type: ConditioningType;
    path: string; // URL /uploads/...
    frameIndex: number;
    strength: number;
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
}

export interface Project {
    id: string;
    name: string;
    shots: Shot[];
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

    // Selectors
    getShotStartTime: (shotId: string) => number;
}

const DEFAULT_PROJECT: Project = {
    id: 'default',
    name: 'Untitled Project',
    shots: [
        {
            id: 'shot-init', // Static ID for init
            prompt: "A cinematic shot...",
            negativePrompt: "worst quality, inconsistent motion, blurry, jittery, distorted, watermark, text, static, freeze, loop, pause, still image, motionless",
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

const DEFAULT_NEGATIVE_PROMPT = "worst quality, inconsistent motion, blurry, jittery, distorted, watermark, text, static, freeze, loop, pause, still image, motionless";

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
                toasts: [],
                assetRefreshVersion: 0,

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
                                    frame_index: t.frameIndex,
                                    strength: t.strength
                                })),
                                last_job_id: s.lastJobId
                            }))
                        };

                        const res = await fetch(`http://localhost:8000/project/${project.id}`, {
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
                        const res = await fetch('http://localhost:8000/project', {
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
                        const res = await fetch(`http://localhost:8000/project/${id}`);
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
                                enhancePrompt: s.enhance_prompt,
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
                                videoUrl: s.last_job_id ? `http://localhost:8000/generated/${s.last_job_id}.mp4` : undefined
                            }))
                        };

                        set({ project: loadedProject, selectedShotId: loadedProject.shots[0]?.id || null });
                        addToast("Project loaded", "success");
                    } catch (e) {
                        console.error(e);
                        addToast("Failed to load project", "error");
                    }
                },

                deleteProject: async (id: string) => {
                    const { addToast, project } = get();
                    try {
                        await fetch(`http://localhost:8000/project/${id}`, { method: 'DELETE' });
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
