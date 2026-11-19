import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export type StudioSurface = 'story' | 'generate' | 'edit' | 'review' | 'library' | 'deliver' | 'shorts' | 'setup';

/** One primary right rail per surface. Generate defaults closed because its
 * shot-params rail already carries the same prompt/seed/frames controls. */
export const DEFAULT_INSPECTOR_BY_SURFACE: Record<StudioSurface, boolean> = {
    story: true,
    generate: false,
    edit: true,
    review: true,
    library: false,
    deliver: false,
    shorts: false,
    setup: false,
};
export type StoryView = 'board' | 'list' | 'script';

export interface Camera {
    x: number;
    y: number;
    zoom: number;
}

export interface BoardPosition {
    x: number;
    y: number;
    width: number;
    height: number;
}

export interface BoardArea extends BoardPosition {
    id: string;
    title: string;
    color?: string;
    collapsed?: boolean;
}

export type LinkKind = 'thread' | 'blocking' | 'reference';

export interface BoardLink {
    id: string;
    source: string;
    target: string;
    kind: LinkKind;
}

export type BoardMode = 'select' | 'area' | 'link';

export interface PlanRecipe {
    id: string;
    name: string;
    mode: 'ask' | 'direct' | 'generate' | 'find';
    text: string;
    created: number;
}

export interface BoardNote {
    id: string;
    text: string;
    x: number;
    y: number;
    width: number;
    height: number;
}

export interface BoardSnapshot {
    boardPositions: Record<string, BoardPosition>;
    boardPinned: Record<string, boolean>;
    boardAreas: Record<string, BoardArea>;
    boardLinks: BoardLink[];
    boardNotes: BoardNote[];
}

interface EditorState {
    activeSurface: StudioSurface;
    storyView: StoryView;
    selectedShotId: string | null;
    selectedShotIds: string[];
    selectedSceneId: string | null;
    inspectorTab: 'shot' | 'prompt' | 'motion' | 'takes' | 'audio' | 'effects' | 'review' | 'history';
    boardCamera: Camera;
    boardPast: BoardSnapshot[];
    boardFuture: BoardSnapshot[];
    boardPositions: Record<string, BoardPosition>;
    boardPinned: Record<string, boolean>;
    boardAreas: Record<string, BoardArea>;
    boardLinks: BoardLink[];
    boardNotes: BoardNote[];
    referencePositions: Record<string, BoardPosition>;
    showReferenceCards: boolean;
    boardQuery: string;
    collabEnabled: boolean;
    boardMode: BoardMode;
    linkKind: LinkKind;
    selectedAreaId: string | null;
    boardMinimapOpen: boolean;
    boardFollowSelection: boolean;
    timelineZoom: number;
    timelineHeight: number;
    inspectorOpen: boolean;
    inspectorWidth: number;
    railBySurface: Record<StudioSurface, boolean>;
    mobileSpineOpen: boolean;
    hiddenTracks: number[];
    commandDockCollapsed: boolean;
    mobileTimelineExpanded: boolean;
    commandOpen: boolean;
    commandMode: 'ask' | 'direct' | 'generate' | 'find';
    commandDraft: string;
    spineCollapsed: boolean;
    generateRailCollapsed: boolean;
    generateParamsCollapsed: boolean;
    density: 'comfortable' | 'compact';
    recipes: PlanRecipe[];
    focusMode: 'none' | 'canvas' | 'stage';
    setSurface: (surface: StudioSurface) => void;
    setStoryView: (view: StoryView) => void;
    selectShot: (shotId: string | null) => void;
    selectShots: (shotIds: string[]) => void;
    toggleShotSelection: (shotId: string) => void;
    selectScene: (sceneId: string | null) => void;
    setInspectorTab: (tab: EditorState['inspectorTab']) => void;
    setCamera: (camera: Camera) => void;
    hydrateBoard: (
        positions: Record<string, BoardPosition>,
        pinned: Record<string, boolean>,
        camera?: Camera,
        areas?: Record<string, BoardArea>,
        links?: BoardLink[],
    ) => void;
    setBoardPosition: (shotId: string, position: Partial<BoardPosition>) => void;
    togglePin: (shotId: string) => void;
    pushBoardHistory: () => void;
    undoBoard: () => void;
    redoBoard: () => void;
    setBoardMode: (mode: BoardMode) => void;
    setBoardQuery: (query: string) => void;
    setCollabEnabled: (enabled: boolean) => void;
    setLinkKind: (kind: LinkKind) => void;
    selectArea: (areaId: string | null) => void;
    addArea: (area: Omit<BoardArea, 'id'>) => string;
    updateArea: (areaId: string, patch: Partial<BoardArea>) => void;
    removeArea: (areaId: string) => void;
    addLink: (source: string, target: string, kind: LinkKind) => void;
    removeLink: (linkId: string) => void;
    addNote: (note: Omit<BoardNote, 'id'>) => string;
    updateNote: (noteId: string, patch: Partial<BoardNote>) => void;
    removeNote: (noteId: string) => void;
    setReferencePosition: (elementId: string, position: Partial<BoardPosition>) => void;
    setShowReferenceCards: (show: boolean) => void;
    setMinimapOpen: (open: boolean) => void;
    setFollowSelection: (follow: boolean) => void;
    setTimelineZoom: (zoom: number) => void;
    setTimelineHeight: (height: number) => void;
    setInspectorOpen: (open: boolean) => void;
    setInspectorWidth: (width: number) => void;
    setMobileSpineOpen: (open: boolean) => void;
    toggleTrackVisibility: (trackId: number) => void;
    setCommandDockCollapsed: (collapsed: boolean) => void;
    setMobileTimelineExpanded: (expanded: boolean) => void;
    setCommandOpen: (open: boolean) => void;
    setCommandMode: (mode: EditorState['commandMode']) => void;
    setCommandDraft: (draft: string) => void;
    setSpineCollapsed: (collapsed: boolean) => void;
    setGenerateRailCollapsed: (collapsed: boolean) => void;
    setGenerateParamsCollapsed: (collapsed: boolean) => void;
    setDensity: (density: 'comfortable' | 'compact') => void;
    saveRecipe: (recipe: Omit<PlanRecipe, 'id' | 'created'>) => void;
    deleteRecipe: (recipeId: string) => void;
    setFocusMode: (mode: EditorState['focusMode']) => void;
    resetBoard: () => void;
}

const DEFAULT_CAMERA: Camera = { x: 48, y: 48, zoom: 0.85 };

function snapshotBoard(state: BoardSnapshot): BoardSnapshot {
    return {
        boardPositions: { ...state.boardPositions },
        boardPinned: { ...state.boardPinned },
        boardAreas: { ...state.boardAreas },
        boardLinks: [...state.boardLinks],
        boardNotes: [...state.boardNotes],
    };
}

function makeId(prefix: string): string {
    const random = typeof crypto !== 'undefined' && 'randomUUID' in crypto
        ? crypto.randomUUID()
        : Math.random().toString(36).slice(2, 10);
    return `${prefix}_${random}`;
}

export const useStudioEditor = create<EditorState>()(
    persist(
        (set) => ({
            activeSurface: 'story',
            storyView: 'board',
            selectedShotId: null,
            selectedShotIds: [],
            selectedSceneId: null,
            inspectorTab: 'shot',
            boardCamera: DEFAULT_CAMERA,
            boardPast: [],
            boardFuture: [],
            boardPositions: {},
            boardPinned: {},
            boardAreas: {},
            boardLinks: [],
            boardNotes: [],
            referencePositions: {},
            showReferenceCards: false,
            boardQuery: '',
            collabEnabled: false,
            boardMode: 'select',
            linkKind: 'thread',
            selectedAreaId: null,
            boardMinimapOpen: true,
            boardFollowSelection: true,
            timelineZoom: 48,
            timelineHeight: 330,
            inspectorOpen: DEFAULT_INSPECTOR_BY_SURFACE.story,
            inspectorWidth: 360,
            railBySurface: { ...DEFAULT_INSPECTOR_BY_SURFACE },
            mobileSpineOpen: false,
            hiddenTracks: [],
            commandDockCollapsed: false,
            mobileTimelineExpanded: false,
            commandOpen: false,
            commandMode: 'direct',
            commandDraft: '',
            spineCollapsed: false,
            generateRailCollapsed: false,
            generateParamsCollapsed: false,
            density: 'comfortable',
            recipes: [],
            focusMode: 'none',
            setSurface: (activeSurface) => set((state) => ({
                activeSurface,
                inspectorOpen: state.railBySurface[activeSurface] ?? DEFAULT_INSPECTOR_BY_SURFACE[activeSurface],
            })),
            setStoryView: (storyView) => set({ storyView }),
            selectShot: (selectedShotId) => set({
                selectedShotId,
                selectedShotIds: selectedShotId ? [selectedShotId] : [],
            }),
            selectShots: (selectedShotIds) => set({
                selectedShotIds,
                selectedShotId: selectedShotIds.length ? selectedShotIds[selectedShotIds.length - 1] : null,
            }),
            toggleShotSelection: (shotId) => set((state) => {
                const exists = state.selectedShotIds.includes(shotId);
                const selectedShotIds = exists
                    ? state.selectedShotIds.filter((id) => id !== shotId)
                    : [...state.selectedShotIds, shotId];
                return {
                    selectedShotIds,
                    selectedShotId: selectedShotIds.length
                        ? selectedShotIds[selectedShotIds.length - 1]
                        : null,
                };
            }),
            selectScene: (selectedSceneId) => set({ selectedSceneId }),
            setInspectorTab: (inspectorTab) => set({ inspectorTab }),
            setCamera: (boardCamera) => set({ boardCamera }),
            pushBoardHistory: () => set((state) => ({
                boardPast: [...state.boardPast.slice(-19), snapshotBoard(state)],
                boardFuture: [],
            })),
            undoBoard: () => set((state) => {
                const previous = state.boardPast[state.boardPast.length - 1];
                if (!previous) return state;
                return {
                    ...snapshotBoard(previous),
                    boardPast: state.boardPast.slice(0, -1),
                    boardFuture: [snapshotBoard(state), ...state.boardFuture].slice(0, 20),
                };
            }),
            redoBoard: () => set((state) => {
                const next = state.boardFuture[0];
                if (!next) return state;
                return {
                    ...snapshotBoard(next),
                    boardFuture: state.boardFuture.slice(1),
                    boardPast: [...state.boardPast.slice(-19), snapshotBoard(state)],
                };
            }),
            hydrateBoard: (boardPositions, boardPinned, boardCamera, boardAreas, boardLinks) => set((state) => ({
                boardPositions,
                boardPinned,
                boardAreas: boardAreas || state.boardAreas,
                boardLinks: boardLinks || state.boardLinks,
                boardCamera: boardCamera || state.boardCamera,
            })),
            setBoardPosition: (shotId, position) => set((state) => ({
                boardPositions: {
                    ...state.boardPositions,
                    [shotId]: {
                        ...(state.boardPositions[shotId] || { x: 0, y: 0, width: 300, height: 230 }),
                        ...position,
                    },
                },
            })),
            togglePin: (shotId) => set((state) => ({
                boardPinned: { ...state.boardPinned, [shotId]: !state.boardPinned[shotId] },
            })),
            setBoardMode: (boardMode) => set({ boardMode }),
            setBoardQuery: (boardQuery) => set({ boardQuery }),
            setCollabEnabled: (collabEnabled) => set({ collabEnabled }),
            setLinkKind: (linkKind) => set({ linkKind }),
            selectArea: (selectedAreaId) => set({ selectedAreaId }),
            addArea: (area) => {
                const id = makeId('area');
                set((state) => ({
                    boardAreas: { ...state.boardAreas, [id]: { ...area, id } },
                    selectedAreaId: id,
                }));
                return id;
            },
            updateArea: (areaId, patch) => set((state) => {
                const current = state.boardAreas[areaId];
                if (!current) return state;
                return {
                    boardAreas: { ...state.boardAreas, [areaId]: { ...current, ...patch } },
                };
            }),
            removeArea: (areaId) => set((state) => {
                const next = { ...state.boardAreas };
                delete next[areaId];
                return {
                    boardAreas: next,
                    selectedAreaId: state.selectedAreaId === areaId ? null : state.selectedAreaId,
                };
            }),
            addLink: (source, target, kind) => set((state) => {
                if (!source || !target || source === target) return state;
                const exists = state.boardLinks.some((link) =>
                    link.kind === kind
                    && ((link.source === source && link.target === target)
                        || (link.source === target && link.target === source)),
                );
                if (exists) return state;
                return {
                    boardLinks: [...state.boardLinks, { id: makeId('link'), source, target, kind }],
                };
            }),
            removeLink: (linkId) => set((state) => ({
                boardLinks: state.boardLinks.filter((link) => link.id !== linkId),
            })),
            addNote: (note) => {
                const id = makeId('note');
                set((state) => ({ boardNotes: [...state.boardNotes, { ...note, id }] }));
                return id;
            },
            updateNote: (noteId, patch) => set((state) => ({
                boardNotes: state.boardNotes.map((note) => (
                    note.id === noteId ? { ...note, ...patch } : note
                )),
            })),
            removeNote: (noteId) => set((state) => ({
                boardNotes: state.boardNotes.filter((note) => note.id !== noteId),
            })),
            setReferencePosition: (elementId, position) => set((state) => ({
                referencePositions: {
                    ...state.referencePositions,
                    [elementId]: {
                        ...(state.referencePositions[elementId] || { x: 0, y: 0, width: 220, height: 170 }),
                        ...position,
                    },
                },
            })),
            setShowReferenceCards: (showReferenceCards) => set({ showReferenceCards }),
            setMinimapOpen: (boardMinimapOpen) => set({ boardMinimapOpen }),
            setFollowSelection: (boardFollowSelection) => set({ boardFollowSelection }),
            setTimelineZoom: (timelineZoom) => set({ timelineZoom: Math.max(8, Math.min(240, timelineZoom)) }),
            setTimelineHeight: (timelineHeight) => set({ timelineHeight: Math.max(180, Math.min(620, timelineHeight)) }),
            setInspectorOpen: (inspectorOpen) => set((state) => ({
                inspectorOpen,
                railBySurface: { ...state.railBySurface, [state.activeSurface]: inspectorOpen },
            })),
            setInspectorWidth: (inspectorWidth) => set({ inspectorWidth: Math.max(300, Math.min(560, inspectorWidth)) }),
            setMobileSpineOpen: (mobileSpineOpen) => set({ mobileSpineOpen }),
            toggleTrackVisibility: (trackId) => set((state) => ({
                hiddenTracks: state.hiddenTracks.includes(trackId)
                    ? state.hiddenTracks.filter((id) => id !== trackId)
                    : [...state.hiddenTracks, trackId],
            })),
            setCommandDockCollapsed: (commandDockCollapsed) => set({ commandDockCollapsed }),
            setMobileTimelineExpanded: (mobileTimelineExpanded) => set({ mobileTimelineExpanded }),
            setCommandOpen: (commandOpen) => set({ commandOpen }),
            setCommandMode: (commandMode) => set({ commandMode }),
            setCommandDraft: (commandDraft) => set({ commandDraft }),
            setSpineCollapsed: (spineCollapsed) => set({ spineCollapsed }),
            setGenerateRailCollapsed: (generateRailCollapsed) => set({ generateRailCollapsed }),
            setGenerateParamsCollapsed: (generateParamsCollapsed) => set({ generateParamsCollapsed }),
            setDensity: (density) => set({ density }),
            saveRecipe: (recipe) => set((state) => ({
                recipes: [
                    ...state.recipes.filter((item) => item.name !== recipe.name),
                    { ...recipe, id: makeId('recipe'), created: Date.now() },
                ].slice(-30),
            })),
            deleteRecipe: (recipeId) => set((state) => ({
                recipes: state.recipes.filter((item) => item.id !== recipeId),
            })),
            setFocusMode: (focusMode) => set({ focusMode }),
            resetBoard: () => set({
                boardPositions: {},
                boardPinned: {},
                boardAreas: {},
                boardLinks: [],
                boardNotes: [],
                referencePositions: {},
                boardCamera: DEFAULT_CAMERA,
                selectedAreaId: null,
            }),
        }),
        {
            name: 'milimo-studio-editor',
            version: 3,
            migrate: (persisted: unknown, version: number) => {
                const next = { ...(persisted as Record<string, unknown>) };
                delete next.boardPositions;
                delete next.boardPinned;
                delete next.boardAreas;
                delete next.boardLinks;
                delete next.boardCamera;
                if (version < 3) next.commandOpen = false;
                return next;
            },
            partialize: (state) => ({
                activeSurface: state.activeSurface,
                storyView: state.storyView,
                selectedShotId: state.selectedShotId,
                selectedShotIds: state.selectedShotIds,
                inspectorTab: state.inspectorTab,
                // Board geometry/camera are persisted by the server board/view
                // endpoints. Keeping them in localStorage too serialized 500
                // cards on every pan frame.
                boardNotes: state.boardNotes,
                referencePositions: state.referencePositions,
                showReferenceCards: state.showReferenceCards,
                boardQuery: state.boardQuery,
                collabEnabled: state.collabEnabled,
                boardMode: state.boardMode,
                linkKind: state.linkKind,
                boardMinimapOpen: state.boardMinimapOpen,
                boardFollowSelection: state.boardFollowSelection,
                timelineZoom: state.timelineZoom,
                timelineHeight: state.timelineHeight,
                inspectorOpen: state.inspectorOpen,
                inspectorWidth: state.inspectorWidth,
                railBySurface: state.railBySurface,
                mobileSpineOpen: state.mobileSpineOpen,
                hiddenTracks: state.hiddenTracks,
                commandDockCollapsed: state.commandDockCollapsed,
                mobileTimelineExpanded: state.mobileTimelineExpanded,
                commandOpen: state.commandOpen,
                commandMode: state.commandMode,
                spineCollapsed: state.spineCollapsed,
                generateRailCollapsed: state.generateRailCollapsed,
                generateParamsCollapsed: state.generateParamsCollapsed,
                density: state.density,
                recipes: state.recipes,
            }),
        },
    ),
);
