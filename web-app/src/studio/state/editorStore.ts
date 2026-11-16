import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export type StudioSurface = 'story' | 'generate' | 'edit' | 'review' | 'library' | 'deliver';
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

interface EditorState {
    activeSurface: StudioSurface;
    storyView: StoryView;
    selectedShotId: string | null;
    selectedSceneId: string | null;
    inspectorTab: 'shot' | 'prompt' | 'takes' | 'audio' | 'review' | 'history';
    boardCamera: Camera;
    boardPositions: Record<string, BoardPosition>;
    boardPinned: Record<string, boolean>;
    boardMinimapOpen: boolean;
    boardFollowSelection: boolean;
    timelineZoom: number;
    timelineHeight: number;
    inspectorOpen: boolean;
    inspectorWidth: number;
    commandOpen: boolean;
    commandMode: 'ask' | 'direct' | 'generate' | 'find';
    focusMode: 'none' | 'canvas' | 'stage';
    setSurface: (surface: StudioSurface) => void;
    setStoryView: (view: StoryView) => void;
    selectShot: (shotId: string | null) => void;
    selectScene: (sceneId: string | null) => void;
    setInspectorTab: (tab: EditorState['inspectorTab']) => void;
    setCamera: (camera: Camera) => void;
    hydrateBoard: (positions: Record<string, BoardPosition>, pinned: Record<string, boolean>, camera?: Camera) => void;
    setBoardPosition: (shotId: string, position: Partial<BoardPosition>) => void;
    togglePin: (shotId: string) => void;
    setMinimapOpen: (open: boolean) => void;
    setFollowSelection: (follow: boolean) => void;
    setTimelineZoom: (zoom: number) => void;
    setTimelineHeight: (height: number) => void;
    setInspectorOpen: (open: boolean) => void;
    setInspectorWidth: (width: number) => void;
    setCommandOpen: (open: boolean) => void;
    setCommandMode: (mode: EditorState['commandMode']) => void;
    setFocusMode: (mode: EditorState['focusMode']) => void;
    resetBoard: () => void;
}

const DEFAULT_CAMERA: Camera = { x: 48, y: 48, zoom: 0.85 };

export const useStudioEditor = create<EditorState>()(
    persist(
        (set) => ({
            activeSurface: 'story',
            storyView: 'board',
            selectedShotId: null,
            selectedSceneId: null,
            inspectorTab: 'shot',
            boardCamera: DEFAULT_CAMERA,
            boardPositions: {},
            boardPinned: {},
            boardMinimapOpen: true,
            boardFollowSelection: true,
            timelineZoom: 48,
            timelineHeight: 330,
            inspectorOpen: true,
            inspectorWidth: 360,
            commandOpen: true,
            commandMode: 'direct',
            focusMode: 'none',
            setSurface: (activeSurface) => set({ activeSurface }),
            setStoryView: (storyView) => set({ storyView }),
            selectShot: (selectedShotId) => set({ selectedShotId }),
            selectScene: (selectedSceneId) => set({ selectedSceneId }),
            setInspectorTab: (inspectorTab) => set({ inspectorTab }),
            setCamera: (boardCamera) => set({ boardCamera }),
            hydrateBoard: (boardPositions, boardPinned, boardCamera) => set((state) => ({
                boardPositions,
                boardPinned,
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
            setMinimapOpen: (boardMinimapOpen) => set({ boardMinimapOpen }),
            setFollowSelection: (boardFollowSelection) => set({ boardFollowSelection }),
            setTimelineZoom: (timelineZoom) => set({ timelineZoom: Math.max(8, Math.min(240, timelineZoom)) }),
            setTimelineHeight: (timelineHeight) => set({ timelineHeight: Math.max(180, Math.min(620, timelineHeight)) }),
            setInspectorOpen: (inspectorOpen) => set({ inspectorOpen }),
            setInspectorWidth: (inspectorWidth) => set({ inspectorWidth: Math.max(300, Math.min(560, inspectorWidth)) }),
            setCommandOpen: (commandOpen) => set({ commandOpen }),
            setCommandMode: (commandMode) => set({ commandMode }),
            setFocusMode: (focusMode) => set({ focusMode }),
            resetBoard: () => set({ boardPositions: {}, boardPinned: {}, boardCamera: DEFAULT_CAMERA }),
        }),
        {
            name: 'milimo-studio-editor',
            partialize: (state) => ({
                activeSurface: state.activeSurface,
                storyView: state.storyView,
                selectedShotId: state.selectedShotId,
                inspectorTab: state.inspectorTab,
                boardCamera: state.boardCamera,
                boardPositions: state.boardPositions,
                boardPinned: state.boardPinned,
                boardMinimapOpen: state.boardMinimapOpen,
                boardFollowSelection: state.boardFollowSelection,
                timelineZoom: state.timelineZoom,
                timelineHeight: state.timelineHeight,
                inspectorOpen: state.inspectorOpen,
                inspectorWidth: state.inspectorWidth,
                commandOpen: state.commandOpen,
                commandMode: state.commandMode,
            }),
        },
    ),
);
