export type ConditioningType = 'image' | 'video';
export type ShotType = 'close_up' | 'medium' | 'wide' | 'establishing' | 'insert' | 'tracking';

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

    // Multi-Track
    trackIndex: number; // 0=V1, 1=V2, etc.
    startFrame?: number; // Absolute start for Free tracks
    trimIn: number;
    trimOut: number;

    // Result
    lastJobId?: string;
    videoUrl?: string | null; // Derived from jobID
    thumbnailUrl?: string; // Static Preview
    enhancedPromptResult?: string; // Result from backend
    statusMessage?: string; // Real-time status text
    currentPrompt?: string; // Live evolving prompt during generation
    etaSeconds?: number; // Estimated time remaining

    // UI State
    isGenerating?: boolean;
    status?: 'pending' | 'generating' | 'completed' | 'failed';

    // Storyboard metadata
    sceneId?: string;
    index?: number;
    action?: string;
    dialogue?: string;
    character?: string;
    shotType?: ShotType;
    matchedElements?: MatchedElement[];
}

export interface ShotConfig extends Partial<Shot> {
    id?: string;
}

export interface Scene {
    id: string;
    index: number;
    name: string;
    scriptContent?: string;
    shots: Shot[]; // Frontend convenience: Nested shots
}

export interface MatchedElement {
    element_id: string;
    element_name: string;
    element_type: 'character' | 'location' | 'object';
    image_url?: string;
    trigger_word: string;
    confidence: number;
    match_source?: string;
}

export interface ParsedShot {
    action: string;
    dialogue?: string;
    character?: string;
    shot_type?: string;
    matched_elements?: MatchedElement[];
}

export interface ParsedScene {
    id?: string; // Optional (not in DB yet)
    name: string;
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
    scriptContent?: string;
}

export interface ProjectSlice {
    project: Project;
    setProject: (p: Project) => void;

    // Asset Management
    assetRefreshVersion: number;
    triggerAssetRefresh: () => void;

    // Async
    saveProject: () => Promise<void>;
    createNewProject: (name: string, settings?: { resolutionW: number; resolutionH: number; fps: number; seed: number }) => Promise<void>;
    loadProject: (id: string) => Promise<void>;
    deleteProject: (id: string) => Promise<void>;
}

export interface ShotSlice {
    addShot: (config?: Partial<Shot>) => void;
    updateShot: (id: string, updates: Partial<Shot>) => void;
    patchShot: (id: string, updates: Partial<Shot>) => Promise<void>;
    splitShot: (id: string, splitFrame: number) => Promise<void>;
    reorderShots: (fromIndex: number, toIndex: number) => void;
    deleteShot: (id: string) => void;

    moveShotToValues: (id: string, trackIndex: number, startFrame: number) => Promise<void>;

    addConditioningToShot: (shotId: string, item: Omit<ConditioningItem, 'id'>) => void;
    updateConditioning: (shotId: string, itemId: string, updates: Partial<ConditioningItem>) => void;
    removeConditioning: (shotId: string, itemId: string) => void;

    getShotStartTime: (shotId: string) => number;
    generateShot: (shotId: string) => Promise<void>;
    inpaintShot: (shotId: string, frameDataUrl: string, maskDataUrl: string, prompt: string) => Promise<void>;
    cancelShotGeneration: (shotId: string) => Promise<void>;

    // Storyboard operations
    batchGenerateShots: (shotIds: string[]) => Promise<void>;
    reorderShotsInScene: (sceneId: string, shotIds: string[]) => Promise<void>;
    addShotToScene: (sceneId: string, data?: { action?: string; dialogue?: string; character?: string; shotType?: string }) => Promise<void>;
    deleteShotFromStoryboard: (shotId: string) => Promise<void>;

    // Phase 2: Thumbnails
    generateThumbnail: (shotId: string, force?: boolean) => Promise<void>;
    batchGenerateThumbnails: (shotIds: string[]) => Promise<void>;

    // Phase 3: Timeline integration
    pushStoryboardToTimeline: () => void;
}

export interface PlaybackSlice {
    currentTime: number;
    isPlaying: boolean;
    setCurrentTime: (t: number) => void;
    setIsPlaying: (p: boolean) => void;
}

export interface UISlice {
    selectedShotId: string | null;
    toasts: { id: string; message: string; type: 'success' | 'error' | 'info' }[];
    addToast: (message: string, type?: 'success' | 'error' | 'info') => void;
    removeToast: (id: string) => void;

    transientDuration: number | null;
    setTransientDuration: (d: number | null) => void;

    viewMode: 'timeline' | 'elements' | 'storyboard' | 'images';
    setViewMode: (mode: 'timeline' | 'elements' | 'storyboard' | 'images') => void;

    selectShot: (id: string | null) => void;

    isEditing: boolean;
    setEditing: (e: boolean) => void;
}

export interface TrackSlice {
    trackStates: Record<number, { muted: boolean; locked: boolean; hidden: boolean }>;
    toggleTrackMute: (trackIndex: number) => void;
    toggleTrackLock: (trackIndex: number) => void;
    toggleTrackHidden: (trackIndex: number) => void;
}

export interface ElementSlice {
    elements: StoryElement[];
    generatingElementIds: Record<string, string>;
    fetchElements: (projectId: string) => Promise<void>;
    createElement: (projectId: string, data: Partial<StoryElement>) => Promise<void>;
    updateElement: (elementId: string, data: Partial<StoryElement>) => Promise<void>;
    deleteElement: (elementId: string) => Promise<void>;
    generateVisual: (elementId: string, promptOverride?: string, guidanceOverride?: number, enableAeOverride?: boolean) => Promise<void>;
    cancelElementGeneration: (elementId: string) => Promise<void>;

    // Storyboard
    parseScript: (text: string) => Promise<ParsedScene[]>;
    aiParseScript: (text: string) => Promise<ParsedScene[]>;
    commitStoryboard: (scenes: ParsedScene[], scriptText?: string) => Promise<void>;
    updateSceneName: (sceneId: string, name: string) => Promise<void>;
}

export interface ServerSlice {
    handleServerEvent: (type: string, data: any) => void;
}

export type TimelineState = ProjectSlice & ShotSlice & PlaybackSlice & UISlice & TrackSlice & ElementSlice & ServerSlice;
