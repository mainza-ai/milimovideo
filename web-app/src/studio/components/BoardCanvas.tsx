import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
    ChevronDown, ChevronRight, Grid3X3, Link2, List, Maximize2, Map, Minus, MousePointer2,
    Plus, Redo2, RotateCcw, Search, SquareDashed, Trash2, Pin, Undo2, X,
} from 'lucide-react';
import { useShallow } from 'zustand/react/shallow';
import { useTimelineStore } from '../../stores/timelineStore';
import {
    useStudioEditor, type BoardArea, type BoardPosition, type LinkKind,
} from '../state/editorStore';
import { ShotCard } from './ShotCard';
import { cullCards } from '../board/culling';

const CARD_W = 300;
const CARD_H = 238;
const GAP_X = 44;
const GAP_Y = 42;
const COLS = 4;
const AREA_COLORS = ['#06b2eb', '#9a7bff', '#3ddc97', '#f5b84b', '#ff5d6c'];

function fallbackPosition(index: number): BoardPosition {
    return {
        x: (index % COLS) * (CARD_W + GAP_X),
        y: Math.floor(index / COLS) * (CARD_H + GAP_Y),
        width: CARD_W,
        height: CARD_H,
    };
}

function linkGeometry(source: { x: number; y: number; width: number; height: number }, target: typeof source) {
    const forward = target.x >= source.x;
    const sx = forward ? source.x + source.width : source.x;
    const sy = source.y + source.height / 2;
    const tx = forward ? target.x : target.x + target.width;
    const ty = target.y + target.height / 2;
    const bend = Math.max(48, Math.abs(tx - sx) * 0.4);
    const c1 = forward ? sx + bend : sx - bend;
    const c2 = forward ? tx - bend : tx + bend;
    const minX = Math.min(sx, tx, c1, c2) - 12;
    const maxX = Math.max(sx, tx, c1, c2) + 12;
    const minY = Math.min(sy, ty) - 16;
    const maxY = Math.max(sy, ty) + 16;
    return {
        d: `M ${sx - minX} ${sy - minY} C ${c1 - minX} ${sy - minY}, ${c2 - minX} ${ty - minY}, ${tx - minX} ${ty - minY}`,
        minX,
        minY,
        width: Math.max(24, maxX - minX),
        height: Math.max(24, maxY - minY),
        mx: (sx + tx) / 2 - minX,
        my: (sy + ty) / 2 - minY,
    };
}

export function BoardCanvas() {
    const project = useTimelineStore((state) => state.project);
    const {
        selectedShotId, selectShot, setCurrentTime, addShot, updateShot, deleteShot,
    } = useTimelineStore(useShallow((state) => ({
        selectedShotId: state.selectedShotId,
        selectShot: state.selectShot,
        setCurrentTime: state.setCurrentTime,
        addShot: state.addShot,
        updateShot: state.updateShot,
        deleteShot: state.deleteShot,
    })));
    const {
        storyView, setStoryView, boardCamera, setCamera, boardPositions,
        setBoardPosition, boardPinned, boardMinimapOpen,
        setMinimapOpen, boardFollowSelection, setFollowSelection, resetBoard,
        boardAreas, boardLinks, boardMode, setBoardMode, linkKind, setLinkKind,
        selectedAreaId, selectArea, addArea, updateArea, removeArea, addLink, removeLink,
        boardQuery, setBoardQuery, selectedShotIds, selectShots, toggleShotSelection,
        boardPast, boardFuture, pushBoardHistory, undoBoard, redoBoard, togglePin,
    } = useStudioEditor();
    const hostRef = useRef<HTMLDivElement>(null);
    const [size, setSize] = useState({ width: 900, height: 600 });
    const [marquee, setMarquee] = useState<{ x0: number; y0: number; x1: number; y1: number } | null>(null);
    const drag = useRef<{ x: number; y: number; cameraX: number; cameraY: number } | null>(null);
    const cardDrag = useRef<{
        startX: number;
        startY: number;
        items: Array<{ id: string; origX: number; origY: number }>;
    } | null>(null);
    const cardResize = useRef<{
        id: string;
        startX: number;
        startY: number;
        origW: number;
        origH: number;
    } | null>(null);
    const areaDrag = useRef<{
        mode: 'create' | 'move' | 'resize';
        id?: string;
        startWorldX: number;
        startWorldY: number;
        orig: BoardArea;
        members: Array<{ id: string; x: number; y: number }>;
    } | null>(null);
    const [areaDraft, setAreaDraft] = useState<BoardArea | null>(null);
    const [linkSource, setLinkSource] = useState<string | null>(null);

    const shots = useMemo(() => {
        const q = boardQuery.trim().toLowerCase();
        if (!q) return project.shots;
        return project.shots.filter((shot) =>
            `${shot.action || ''} ${shot.prompt || ''} ${shot.character || ''} ${shot.dialogue || ''}`
                .toLowerCase()
                .includes(q),
        );
    }, [boardQuery, project.shots]);

    // Auto-place new shots in a readable scene-agnostic grid. Pinned cards and
    // cards already placed are never moved.
    useEffect(() => {
        const missing = shots.filter((shot) => !boardPositions[shot.id] && !boardPinned[shot.id]);
        if (missing.length === 0) return;
        const start = Object.keys(boardPositions).length;
        missing.forEach((shot, index) => {
            const slot = start + index;
            setBoardPosition(shot.id, {
                x: (slot % COLS) * (CARD_W + GAP_X),
                y: Math.floor(slot / COLS) * (CARD_H + GAP_Y),
                width: CARD_W,
                height: CARD_H,
            });
        });
    }, [shots, boardPositions, boardPinned, setBoardPosition]);

    useEffect(() => {
        const host = hostRef.current;
        if (!host) return;
        const observer = new ResizeObserver(() => {
            setSize({ width: host.clientWidth, height: host.clientHeight });
        });
        observer.observe(host);
        return () => observer.disconnect();
    }, []);

    // Follow selection centers the camera on the selected card, but only when
    // the selection or follow toggle changes — panning afterwards is stable.
    const hasSelectedPosition = Boolean(selectedShotId && boardPositions[selectedShotId]);
    useEffect(() => {
        if (!boardFollowSelection || !selectedShotId) return;
        const state = useStudioEditor.getState();
        const position = state.boardPositions[selectedShotId];
        if (!position) return;
        const zoom = state.boardCamera.zoom;
        state.setCamera({
            zoom,
            x: size.width / 2 - (position.x + position.width / 2) * zoom,
            y: size.height / 2 - (position.y + position.height / 2) * zoom,
        });
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [selectedShotId, boardFollowSelection, hasSelectedPosition, size.width, size.height]);

    const fit = useCallback(() => {
        const positions = Object.values(boardPositions);
        if (positions.length === 0) {
            setCamera({ x: 48, y: 48, zoom: 0.85 });
            return;
        }
        const minX = Math.min(...positions.map((p) => p.x));
        const minY = Math.min(...positions.map((p) => p.y));
        const maxX = Math.max(...positions.map((p) => p.x + p.width));
        const maxY = Math.max(...positions.map((p) => p.y + p.height));
        const zoom = Math.max(0.2, Math.min(1.4, Math.min(
            (size.width - 90) / Math.max(1, maxX - minX),
            (size.height - 90) / Math.max(1, maxY - minY),
        )));
        setCamera({
            x: (size.width - (maxX - minX) * zoom) / 2 - minX * zoom,
            y: (size.height - (maxY - minY) * zoom) / 2 - minY * zoom,
            zoom,
        });
    }, [boardPositions, setCamera, size.height, size.width]);

    const arrange = useCallback(() => {
        pushBoardHistory();
        project.shots.forEach((shot, index) => {
            setBoardPosition(shot.id, {
                x: (index % COLS) * (CARD_W + GAP_X),
                y: Math.floor(index / COLS) * (CARD_H + GAP_Y),
                width: CARD_W,
                height: CARD_H,
            });
        });
    }, [project.shots, pushBoardHistory, setBoardPosition]);

    const selectedPositions = useCallback(() => selectedShotIds
        .map((id, index) => ({ id, position: boardPositions[id] || fallbackPosition(index) }))
        .filter(({ position }) => !!position), [boardPositions, selectedShotIds]);

    const alignSelected = (mode: 'left' | 'top' | 'center-h' | 'center-v' | 'right' | 'bottom') => {
        const items = selectedPositions();
        if (items.length < 2) return;
        pushBoardHistory();
        const minX = Math.min(...items.map(({ position }) => position.x));
        const maxX = Math.max(...items.map(({ position }) => position.x + position.width));
        const minY = Math.min(...items.map(({ position }) => position.y));
        const maxY = Math.max(...items.map(({ position }) => position.y + position.height));
        for (const { id, position } of items) {
            if (mode === 'left') setBoardPosition(id, { x: minX });
            else if (mode === 'right') setBoardPosition(id, { x: maxX - position.width });
            else if (mode === 'top') setBoardPosition(id, { y: minY });
            else if (mode === 'bottom') setBoardPosition(id, { y: maxY - position.height });
            else if (mode === 'center-h') setBoardPosition(id, { x: (minX + maxX) / 2 - position.width / 2 });
            else setBoardPosition(id, { y: (minY + maxY) / 2 - position.height / 2 });
        }
    };

    const distributeSelected = (axis: 'h' | 'v') => {
        const items = selectedPositions();
        if (items.length < 3) return;
        pushBoardHistory();
        const sorted = [...items].sort((a, b) => (
            axis === 'h' ? a.position.x - b.position.x : a.position.y - b.position.y
        ));
        const first = sorted[0];
        const last = sorted[sorted.length - 1];
        const start = axis === 'h' ? first.position.x : first.position.y;
        const end = axis === 'h'
            ? last.position.x + last.position.width
            : last.position.y + last.position.height;
        const span = end - start;
        const totalSizes = sorted.reduce(
            (sum, item) => sum + (axis === 'h' ? item.position.width : item.position.height),
            0,
        );
        const gap = (span - totalSizes) / (sorted.length - 1);
        let cursor = start;
        for (const item of sorted) {
            if (axis === 'h') {
                setBoardPosition(item.id, { x: Math.round(cursor) });
                cursor += item.position.width + gap;
            } else {
                setBoardPosition(item.id, { y: Math.round(cursor) });
                cursor += item.position.height + gap;
            }
        }
    };

    const toWorld = useCallback((clientX: number, clientY: number) => {
        const rect = hostRef.current?.getBoundingClientRect();
        if (!rect) return { x: 0, y: 0 };
        return {
            x: (clientX - rect.left - boardCamera.x) / boardCamera.zoom,
            y: (clientY - rect.top - boardCamera.y) / boardCamera.zoom,
        };
    }, [boardCamera]);

    const startAreaMove = (event: React.PointerEvent, area: BoardArea) => {
        event.stopPropagation();
        selectArea(area.id);
        setBoardMode('area');
        pushBoardHistory();
        const world = toWorld(event.clientX, event.clientY);
        const members = project.shots
            .map((shot, index) => ({
                id: shot.id,
                position: boardPositions[shot.id] || fallbackPosition(index),
            }))
            .filter(({ position }) => {
                const cx = position.x + position.width / 2;
                const cy = position.y + position.height / 2;
                return cx >= area.x && cx <= area.x + area.width
                    && cy >= area.y && cy <= area.y + area.height;
            })
            .map(({ id, position }) => ({ id, x: position.x, y: position.y }));
        areaDrag.current = {
            mode: 'move', id: area.id, startWorldX: world.x, startWorldY: world.y, orig: area, members,
        };
        hostRef.current?.setPointerCapture(event.pointerId);
    };

    const startAreaResize = (event: React.PointerEvent, area: BoardArea) => {
        event.stopPropagation();
        selectArea(area.id);
        pushBoardHistory();
        const world = toWorld(event.clientX, event.clientY);
        areaDrag.current = {
            mode: 'resize', id: area.id, startWorldX: world.x, startWorldY: world.y, orig: area, members: [],
        };
        hostRef.current?.setPointerCapture(event.pointerId);
    };

    const onWheel = (event: React.WheelEvent) => {
        if (!event.ctrlKey && !event.metaKey && Math.abs(event.deltaY) < 2) return;
        event.preventDefault();
        const rect = hostRef.current?.getBoundingClientRect();
        if (!rect) return;
        const px = event.clientX - rect.left;
        const py = event.clientY - rect.top;
        const factor = event.deltaY > 0 ? 0.92 : 1.08;
        const zoom = Math.max(0.25, Math.min(2.2, boardCamera.zoom * factor));
        const ratio = zoom / boardCamera.zoom;
        setCamera({
            zoom,
            x: px - (px - boardCamera.x) * ratio,
            y: py - (py - boardCamera.y) * ratio,
        });
    };

    const startPan = (event: React.PointerEvent) => {
        if (event.button !== 0) return;
        const target = event.target as HTMLElement;
        if (target.closest('.st-shot-card') || target.closest('button')) return;
        if (target.closest('.st-board-area')) {
            selectArea(null);
            return;
        }
        if (boardMode === 'area') {
            const world = toWorld(event.clientX, event.clientY);
            const draft: BoardArea = {
                id: 'draft',
                title: `Area ${Object.keys(boardAreas).length + 1}`,
                x: world.x,
                y: world.y,
                width: 1,
                height: 1,
                color: AREA_COLORS[Object.keys(boardAreas).length % AREA_COLORS.length],
            };
            areaDrag.current = {
                mode: 'create', startWorldX: world.x, startWorldY: world.y, orig: draft, members: [],
            };
            setAreaDraft(draft);
            (event.currentTarget as HTMLElement).setPointerCapture(event.pointerId);
            return;
        }
        if (boardMode === 'link') return;
        selectArea(null);
        if (event.shiftKey) {
            const rect = hostRef.current?.getBoundingClientRect();
            if (!rect) return;
            const x = event.clientX - rect.left;
            const y = event.clientY - rect.top;
            setMarquee({ x0: x, y0: y, x1: x, y1: y });
            (event.currentTarget as HTMLElement).setPointerCapture(event.pointerId);
            return;
        }
        drag.current = {
            x: event.clientX,
            y: event.clientY,
            cameraX: boardCamera.x,
            cameraY: boardCamera.y,
        };
        (event.currentTarget as HTMLElement).setPointerCapture(event.pointerId);
    };

    const startCardDrag = (event: React.PointerEvent, shotId: string) => {
        event.stopPropagation();
        if (boardMode === 'link') {
            selectShot(shotId);
            if (!linkSource) {
                setLinkSource(shotId);
            } else if (linkSource !== shotId) {
                pushBoardHistory();
                addLink(linkSource, shotId, linkKind);
                setLinkSource(null);
            } else {
                setLinkSource(null);
            }
            return;
        }
        if (event.shiftKey) return;
        const ids = selectedShotIds.includes(shotId) ? selectedShotIds : [shotId];
        if (!selectedShotIds.includes(shotId)) selectShot(shotId);
        pushBoardHistory();
        cardDrag.current = {
            startX: event.clientX,
            startY: event.clientY,
            items: ids
                .map((id) => ({ id, position: boardPositions[id] }))
                .filter((item) => !!item.position)
                .map((item) => ({ id: item.id, origX: item.position!.x, origY: item.position!.y })),
        };
    };

    const startCardResize = (event: React.PointerEvent, shotId: string) => {
        event.stopPropagation();
        const position = boardPositions[shotId];
        if (!position) return;
        if (!selectedShotIds.includes(shotId)) selectShot(shotId);
        pushBoardHistory();
        cardResize.current = {
            id: shotId,
            startX: event.clientX,
            startY: event.clientY,
            origW: position.width,
            origH: position.height,
        };
        (event.currentTarget as HTMLElement).setPointerCapture(event.pointerId);
    };

    const onPointerMove = (event: React.PointerEvent) => {
        if (drag.current) {
            setCamera({
                ...boardCamera,
                x: drag.current.cameraX + (event.clientX - drag.current.x),
                y: drag.current.cameraY + (event.clientY - drag.current.y),
            });
        }
        if (marquee) {
            const rect = hostRef.current?.getBoundingClientRect();
            if (rect) {
                setMarquee({
                    ...marquee,
                    x1: event.clientX - rect.left,
                    y1: event.clientY - rect.top,
                });
            }
        }
        if (cardDrag.current) {
            const deltaX = (event.clientX - cardDrag.current.startX) / boardCamera.zoom;
            const deltaY = (event.clientY - cardDrag.current.startY) / boardCamera.zoom;
            for (const item of cardDrag.current.items) {
                setBoardPosition(item.id, {
                    x: Math.round((item.origX + deltaX) / 8) * 8,
                    y: Math.round((item.origY + deltaY) / 8) * 8,
                });
            }
        }
        if (cardResize.current) {
            const deltaX = (event.clientX - cardResize.current.startX) / boardCamera.zoom;
            const deltaY = (event.clientY - cardResize.current.startY) / boardCamera.zoom;
            setBoardPosition(cardResize.current.id, {
                width: Math.max(200, Math.round(cardResize.current.origW + deltaX)),
                height: Math.max(160, Math.round(cardResize.current.origH + deltaY)),
            });
        }
        if (areaDrag.current) {
            const world = toWorld(event.clientX, event.clientY);
            const dragState = areaDrag.current;
            if (dragState.mode === 'create') {
                setAreaDraft({
                    ...dragState.orig,
                    x: Math.min(dragState.startWorldX, world.x),
                    y: Math.min(dragState.startWorldY, world.y),
                    width: Math.max(1, Math.abs(world.x - dragState.startWorldX)),
                    height: Math.max(1, Math.abs(world.y - dragState.startWorldY)),
                });
            } else if (dragState.id) {
                if (dragState.mode === 'move') {
                    const dx = world.x - dragState.startWorldX;
                    const dy = world.y - dragState.startWorldY;
                    updateArea(dragState.id, {
                        x: Math.round(dragState.orig.x + dx),
                        y: Math.round(dragState.orig.y + dy),
                    });
                    for (const member of dragState.members) {
                        setBoardPosition(member.id, {
                            x: Math.round(member.x + dx),
                            y: Math.round(member.y + dy),
                        });
                    }
                } else {
                    updateArea(dragState.id, {
                        width: Math.max(160, Math.round(dragState.orig.width + world.x - dragState.startWorldX)),
                        height: Math.max(120, Math.round(dragState.orig.height + world.y - dragState.startWorldY)),
                    });
                }
            }
        }
    };

    const endDrag = () => {
        if (marquee) {
            const rect = hostRef.current?.getBoundingClientRect();
            if (rect) {
                const a = toWorld(rect.left + Math.min(marquee.x0, marquee.x1), rect.top + Math.min(marquee.y0, marquee.y1));
                const b = toWorld(rect.left + Math.max(marquee.x0, marquee.x1), rect.top + Math.max(marquee.y0, marquee.y1));
                const hits = project.shots
                    .map((shot, index) => ({ id: shot.id, position: boardPositions[shot.id] || fallbackPosition(index) }))
                    .filter(({ position }) => (
                        position.x + position.width >= a.x
                        && position.x <= b.x
                        && position.y + position.height >= a.y
                        && position.y <= b.y
                    ))
                    .map(({ id }) => id);
                if (hits.length) selectShots(hits);
                else selectShots([]);
            }
            setMarquee(null);
        }
        cardResize.current = null;
        if (areaDrag.current?.mode === 'create' && areaDraft) {
            if (areaDraft.width > 40 && areaDraft.height > 40) {
                pushBoardHistory();
                addArea({
                    title: areaDraft.title,
                    x: Math.round(areaDraft.x),
                    y: Math.round(areaDraft.y),
                    width: Math.round(areaDraft.width),
                    height: Math.round(areaDraft.height),
                    color: areaDraft.color,
                });
                setBoardMode('select');
            }
            setAreaDraft(null);
        }
        drag.current = null;
        cardDrag.current = null;
        areaDrag.current = null;
    };

    // Viewport culling: 500 cards stay cheap because only the visible slice is
    // mounted. The minimap still uses the full layout.
    const visibleShots = useMemo(() => {
        const cards = shots.map((shot, index) => ({
            shot,
            index,
            position: boardPositions[shot.id] || fallbackPosition(index),
        }));
        const culled = cullCards(cards, boardCamera, size);
        const collapsed = Object.values(boardAreas).filter((area) => area.collapsed);
        if (collapsed.length === 0) return culled;
        return culled.filter(({ position }) => {
            const cx = position.x + position.width / 2;
            const cy = position.y + position.height / 2;
            return !collapsed.some((area) => (
                cx >= area.x && cx <= area.x + area.width
                && cy >= area.y && cy <= area.y + area.height
            ));
        });
    }, [boardAreas, boardCamera, boardPositions, shots, size]);

    const hiddenShotIds = useMemo(() => {
        const collapsed = Object.values(boardAreas).filter((area) => area.collapsed);
        if (collapsed.length === 0) return new Set<string>();
        return new Set(project.shots
            .map((shot, index) => ({
                id: shot.id,
                position: boardPositions[shot.id] || fallbackPosition(index),
            }))
            .filter(({ position }) => {
                const cx = position.x + position.width / 2;
                const cy = position.y + position.height / 2;
                return collapsed.some((area) => (
                    cx >= area.x && cx <= area.x + area.width
                    && cy >= area.y && cy <= area.y + area.height
                ));
            })
            .map(({ id }) => id));
    }, [boardAreas, boardPositions, project.shots]);

    const lod = boardCamera.zoom < 0.55 ? 'summary' : boardCamera.zoom < 0.95 ? 'compact' : 'detail';
    const minimap = useMemo(() => {
        const positions = Object.values(boardPositions);
        if (positions.length === 0) return null;
        const minX = Math.min(...positions.map((p) => p.x));
        const minY = Math.min(...positions.map((p) => p.y));
        const maxX = Math.max(...positions.map((p) => p.x + p.width));
        const maxY = Math.max(...positions.map((p) => p.y + p.height));
        const spanX = Math.max(1, maxX - minX);
        const spanY = Math.max(1, maxY - minY);
        const scale = Math.min(160 / spanX, 98 / spanY);
        return {
            minX,
            minY,
            scale,
            nodes: positions.map((position) => ({
                x: (position.x - minX) * scale + 5,
                y: (position.y - minY) * scale + 5,
                w: Math.max(2, position.width * scale),
                h: Math.max(2, position.height * scale),
            })),
            view: {
                x: (-boardCamera.x / boardCamera.zoom - minX) * scale + 5,
                y: (-boardCamera.y / boardCamera.zoom - minY) * scale + 5,
                w: (size.width / boardCamera.zoom) * scale,
                h: (size.height / boardCamera.zoom) * scale,
            },
        };
    }, [boardCamera, boardPositions, size.height, size.width]);

    return (
        <div className="st-board-wrap">
            <div className="st-board-toolbar">
                <button type="button" className="st-button" data-size="sm" data-variant={storyView === 'board' ? 'primary' : 'ghost'} onClick={() => setStoryView('board')}>
                    <Grid3X3 size={13} /> Board
                </button>
                <button type="button" className="st-button" data-size="sm" data-variant={storyView === 'list' ? 'primary' : 'ghost'} onClick={() => setStoryView('list')}>
                    <List size={13} /> Shot list
                </button>
                <button type="button" className="st-button" data-size="sm" data-variant={storyView === 'script' ? 'primary' : 'ghost'} onClick={() => setStoryView('script')}>
                    <Search size={13} /> Script
                </button>
                <span style={{ width: 1, height: 18, background: 'var(--st-border-strong)', margin: '0 2px' }} />
                <button
                    type="button"
                    className="st-button"
                    data-size="sm"
                    data-variant={boardMode === 'select' ? 'primary' : 'ghost'}
                    onClick={() => { setBoardMode('select'); setLinkSource(null); }}
                    title="Select and pan"
                >
                    <MousePointer2 size={13} /> Select
                </button>
                <button
                    type="button"
                    className="st-button"
                    data-size="sm"
                    data-variant={boardMode === 'area' ? 'primary' : 'ghost'}
                    onClick={() => { setBoardMode('area'); setLinkSource(null); }}
                    title="Drag to create an area"
                >
                    <SquareDashed size={13} /> Area
                </button>
                <button
                    type="button"
                    className="st-button"
                    data-size="sm"
                    data-variant={boardMode === 'link' ? 'primary' : 'ghost'}
                    onClick={() => { setBoardMode('link'); setLinkSource(null); }}
                    title="Click two cards to link them"
                >
                    <Link2 size={13} /> Link
                </button>
                {boardMode === 'link' && (
                    <select
                        className="st-select"
                        style={{ width: 116, height: 27, padding: '2px 6px' }}
                        value={linkKind}
                        onChange={(event) => setLinkKind(event.target.value as LinkKind)}
                        aria-label="Link kind"
                    >
                        <option value="thread">Thread</option>
                        <option value="blocking">Blocking</option>
                        <option value="reference">Reference</option>
                    </select>
                )}
                <span className="st-spacer" />
                <input
                    className="st-input"
                    style={{ width: 180, height: 27, padding: '3px 8px' }}
                    placeholder="Filter shots…"
                    value={boardQuery}
                    onChange={(event) => setBoardQuery(event.target.value)}
                    aria-label="Filter shots"
                />
                <button
                    type="button"
                    className="st-button"
                    data-size="sm"
                    data-variant="primary"
                    onClick={() => addShot({ prompt: 'New shot' })}
                    title="Add a shot to the story"
                >
                    <Plus size={13} /> Add shot
                </button>
                <button type="button" className="st-button" data-size="sm" onClick={arrange} title="Arrange cards">
                    <Grid3X3 size={13} /> Arrange
                </button>
                {selectedShotIds.length >= 2 && (
                    <>
                        <select
                            className="st-select"
                            style={{ width: 104, height: 27, padding: '2px 6px' }}
                            value=""
                            aria-label="Align selected cards"
                            onChange={(event) => {
                                if (event.target.value) alignSelected(event.target.value as Parameters<typeof alignSelected>[0]);
                            }}
                        >
                            <option value="">Align…</option>
                            <option value="left">Left</option>
                            <option value="center-h">Center H</option>
                            <option value="right">Right</option>
                            <option value="top">Top</option>
                            <option value="center-v">Center V</option>
                            <option value="bottom">Bottom</option>
                        </select>
                        <button
                            type="button"
                            className="st-button"
                            data-size="sm"
                            data-variant="ghost"
                            disabled={selectedShotIds.length < 3}
                            onClick={() => distributeSelected('h')}
                            title="Distribute horizontally"
                        >
                            Dist H
                        </button>
                        <button
                            type="button"
                            className="st-button"
                            data-size="sm"
                            data-variant="ghost"
                            disabled={selectedShotIds.length < 3}
                            onClick={() => distributeSelected('v')}
                            title="Distribute vertically"
                        >
                            Dist V
                        </button>
                    </>
                )}
                <button type="button" className="st-button" data-size="sm" onClick={fit} title="Fit board">
                    <Maximize2 size={13} /> Fit
                </button>
                <button type="button" className="st-button" data-size="sm" data-variant={boardMinimapOpen ? 'primary' : 'ghost'} onClick={() => setMinimapOpen(!boardMinimapOpen)} title="Minimap">
                    <Map size={13} />
                </button>
                <button type="button" className="st-button" data-size="sm" data-variant={boardFollowSelection ? 'primary' : 'ghost'} onClick={() => setFollowSelection(!boardFollowSelection)} title="Follow selection">
                    <Pin size={13} />
                </button>
                {selectedShotIds.length === 1 && (
                    <button
                        type="button"
                        className="st-button"
                        data-size="sm"
                        data-variant={boardPinned[selectedShotIds[0]] ? 'primary' : 'ghost'}
                        title="Pin selected card"
                        onClick={() => {
                            pushBoardHistory();
                            togglePin(selectedShotIds[0]);
                        }}
                    >
                        <Pin size={13} /> {boardPinned[selectedShotIds[0]] ? 'Unpin' : 'Pin'}
                    </button>
                )}
                <button
                    type="button"
                    className="st-button"
                    data-size="sm"
                    data-variant="ghost"
                    disabled={boardPast.length === 0}
                    onClick={undoBoard}
                    title="Undo board change"
                >
                    <Undo2 size={13} />
                </button>
                <button
                    type="button"
                    className="st-button"
                    data-size="sm"
                    data-variant="ghost"
                    disabled={boardFuture.length === 0}
                    onClick={redoBoard}
                    title="Redo board change"
                >
                    <Redo2 size={13} />
                </button>
                <button
                    type="button"
                    className="st-button"
                    data-size="sm"
                    data-variant="ghost"
                    onClick={() => { pushBoardHistory(); resetBoard(); }}
                    title="Reset board layout"
                >
                    <RotateCcw size={13} />
                </button>
            </div>

            {storyView === 'board' ? (
                <div
                    ref={hostRef}
                    className={`st-board-viewport st-board-lod-${lod}`}
                    onPointerDown={startPan}
                    onPointerMove={onPointerMove}
                    onPointerUp={endDrag}
                    onPointerCancel={endDrag}
                    onWheel={onWheel}
                >
                    {shots.length === 0 ? (
                        <div className="st-empty">
                            <strong>No shots yet</strong>
                            <span>Import a script in Story → Script, or add a shot in the legacy editor while F1 lands.</span>
                        </div>
                    ) : (
                        <div
                            className="st-board-world"
                            style={{
                                transform: `translate3d(${boardCamera.x}px, ${boardCamera.y}px, 0) scale(${boardCamera.zoom})`,
                            }}
                        >
                            {Object.values(boardAreas).map((area) => (
                                <div
                                    key={area.id}
                                    className="st-board-area"
                                    data-selected={selectedAreaId === area.id}
                                    style={{
                                        left: area.x,
                                        top: area.y,
                                        width: area.width,
                                        height: area.collapsed ? 30 : area.height,
                                        borderColor: area.color || 'var(--st-border-strong)',
                                        background: `${area.color || '#06b2eb'}${area.collapsed ? '08' : '12'}`,
                                    }}
                                    onPointerDown={(event) => {
                                        if ((event.target as HTMLElement).closest('input, button')) return;
                                        event.stopPropagation();
                                        selectArea(area.id);
                                    }}
                                >
                                    <div
                                        className="st-board-area-head"
                                        style={{ color: area.color || 'var(--st-secondary)' }}
                                        onPointerDown={(event) => startAreaMove(event, area)}
                                    >
                                        <button
                                            type="button"
                                            className="st-icon-button"
                                            aria-label={area.collapsed ? `Expand ${area.title}` : `Collapse ${area.title}`}
                                            aria-expanded={!area.collapsed}
                                            onPointerDown={(event) => event.stopPropagation()}
                                            onClick={() => updateArea(area.id, { collapsed: !area.collapsed })}
                                        >
                                            {area.collapsed ? <ChevronRight size={12} /> : <ChevronDown size={12} />}
                                        </button>
                                        <input
                                            className="st-board-area-title"
                                            value={area.title}
                                            aria-label="Area title"
                                            onPointerDown={(event) => event.stopPropagation()}
                                            onChange={(event) => updateArea(area.id, { title: event.target.value })}
                                        />
                                        <button
                                            type="button"
                                            className="st-icon-button"
                                            aria-label={`Remove ${area.title}`}
                                            onPointerDown={(event) => event.stopPropagation()}
                                            onClick={() => removeArea(area.id)}
                                        >
                                            <Trash2 size={12} />
                                        </button>
                                    </div>
                                    {!area.collapsed && (
                                        <button
                                            type="button"
                                            className="st-board-area-resize"
                                            aria-label="Resize area"
                                            onPointerDown={(event) => startAreaResize(event, area)}
                                        />
                                    )}
                                </div>
                            ))}

                            {areaDraft && (
                                <div
                                    className="st-board-area st-board-area-draft"
                                    style={{
                                        left: areaDraft.x,
                                        top: areaDraft.y,
                                        width: areaDraft.width,
                                        height: areaDraft.height,
                                        borderColor: areaDraft.color,
                                        background: `${areaDraft.color || '#06b2eb'}12`,
                                    }}
                                />
                            )}

                            {boardLinks.map((link) => {
                                if (hiddenShotIds.has(link.source) || hiddenShotIds.has(link.target)) return null;
                                const source = boardPositions[link.source];
                                const target = boardPositions[link.target];
                                if (!source || !target) return null;
                                const geometry = linkGeometry(source, target);
                                return (
                                    <svg
                                        key={link.id}
                                        className={`st-board-link st-board-link-${link.kind}`}
                                        style={{
                                            left: geometry.minX,
                                            top: geometry.minY,
                                            width: geometry.width,
                                            height: geometry.height,
                                        }}
                                        aria-label={`${link.kind} link`}
                                    >
                                        <path d={geometry.d} vectorEffect="non-scaling-stroke" />
                                        <circle
                                            className="st-board-link-node"
                                            cx={geometry.mx}
                                            cy={geometry.my}
                                            r={6}
                                            onClick={(event) => {
                                                event.stopPropagation();
                                                removeLink(link.id);
                                            }}
                                        >
                                            <title>Remove {link.kind} link</title>
                                        </circle>
                                    </svg>
                                );
                            })}

                            {visibleShots.map(({ shot, index, position }) => {
                                return (
                                    <ShotCard
                                        key={shot.id}
                                        shot={shot}
                                        index={index}
                                        position={position}
                                        selected={selectedShotIds.includes(shot.id)}
                                        pinned={!!boardPinned[shot.id]}
                                        linkSource={linkSource === shot.id}
                                        onSelect={(event) => {
                                            if (event.shiftKey) {
                                                toggleShotSelection(shot.id);
                                                return;
                                            }
                                            if (!selectedShotIds.includes(shot.id)) selectShot(shot.id);
                                            setCurrentTime(0);
                                        }}
                                        onDragStart={startCardDrag}
                                        onResize={startCardResize}
                                    />
                                );
                            })}
                        </div>
                    )}

                    {marquee && (
                        <div
                            className="st-board-marquee"
                            style={{
                                left: Math.min(marquee.x0, marquee.x1),
                                top: Math.min(marquee.y0, marquee.y1),
                                width: Math.abs(marquee.x1 - marquee.x0),
                                height: Math.abs(marquee.y1 - marquee.y0),
                            }}
                            aria-hidden="true"
                        />
                    )}

                    <div className="st-board-zoom">
                        <button type="button" className="st-icon-button" onClick={() => setCamera({ ...boardCamera, zoom: Math.max(0.25, boardCamera.zoom / 1.15) })} aria-label="Zoom out">
                            <Minus size={12} />
                        </button>
                        <span className="st-mono">{Math.round(boardCamera.zoom * 100)}%</span>
                        {visibleShots.length !== shots.length && (
                            <span className="st-mono" title="Cards mounted in the viewport">
                                {visibleShots.length}/{shots.length}
                            </span>
                        )}
                        <button type="button" className="st-icon-button" onClick={() => setCamera({ ...boardCamera, zoom: Math.min(2.2, boardCamera.zoom * 1.15) })} aria-label="Zoom in">
                            <Plus size={12} />
                        </button>
                    </div>

                    {boardMinimapOpen && minimap && (
                        <div className="st-board-minimap" role="img" aria-label="Board minimap">
                            <div className="st-board-minimap-world">
                                {minimap.nodes.map((node, index) => (
                                    <div
                                        key={index}
                                        className="st-board-minimap-node"
                                        style={{ left: node.x, top: node.y, width: node.w, height: node.h }}
                                    />
                                ))}
                                <div
                                    className="st-board-minimap-view"
                                    style={{ left: minimap.view.x, top: minimap.view.y, width: minimap.view.w, height: minimap.view.h }}
                                />
                            </div>
                        </div>
                    )}
                </div>
            ) : storyView === 'list' ? (
                <div className="st-list-wrap">
                    <div className="st-button-row" style={{ marginBottom: 8 }}>
                        <span className="st-value" style={{ color: 'var(--st-muted)', fontSize: 10 }}>
                            {selectedShotIds.length} selected
                        </span>
                        <button
                            type="button"
                            className="st-button"
                            data-size="sm"
                            data-variant="ghost"
                            onClick={() => selectShots(shots.map((shot) => shot.id))}
                        >
                            Select all
                        </button>
                        <button
                            type="button"
                            className="st-button"
                            data-size="sm"
                            data-variant="danger"
                            disabled={selectedShotIds.length === 0}
                            onClick={() => {
                                if (!window.confirm(`Delete ${selectedShotIds.length} shot(s)?`)) return;
                                for (const id of selectedShotIds) deleteShot(id);
                                selectShots([]);
                            }}
                        >
                            <Trash2 size={12} /> Delete selected
                        </button>
                    </div>
                    <table className="st-table">
                        <caption>Shot list — text equivalent of the board</caption>
                        <thead>
                            <tr>
                                <th scope="col">
                                    <input
                                        type="checkbox"
                                        aria-label="Select all shots"
                                        checked={shots.length > 0 && selectedShotIds.length === shots.length}
                                        onChange={(event) => {
                                            selectShots(event.target.checked ? shots.map((shot) => shot.id) : []);
                                        }}
                                    />
                                </th>
                                <th scope="col">#</th>
                                <th scope="col">Shot</th>
                                <th scope="col">Type</th>
                                <th scope="col">Frames</th>
                                <th scope="col">Status</th>
                                <th scope="col">Take</th>
                            </tr>
                        </thead>
                        <tbody>
                            {shots.map((shot, index) => (
                                <tr
                                    key={shot.id}
                                    data-selected={selectedShotIds.includes(shot.id)}
                                    onClick={() => selectShot(shot.id)}
                                >
                                    <td onClick={(event) => event.stopPropagation()}>
                                        <input
                                            type="checkbox"
                                            aria-label={`Select shot ${index + 1}`}
                                            checked={selectedShotIds.includes(shot.id)}
                                            onChange={() => toggleShotSelection(shot.id)}
                                        />
                                    </td>
                                    <td className="st-mono">{index + 1}</td>
                                    <td>
                                        <input
                                            className="st-input"
                                            style={{ height: 28, padding: '3px 7px' }}
                                            value={shot.action || shot.prompt || ''}
                                            aria-label={`Shot ${index + 1} prompt`}
                                            onClick={(event) => event.stopPropagation()}
                                            onChange={(event) => updateShot(shot.id, { prompt: event.target.value })}
                                        />
                                    </td>
                                    <td onClick={(event) => event.stopPropagation()}>
                                        <select
                                            className="st-select"
                                            style={{ height: 28, padding: '2px 5px' }}
                                            value={shot.shotType || 'medium'}
                                            aria-label={`Shot ${index + 1} type`}
                                            onChange={(event) => updateShot(shot.id, {
                                                shotType: event.target.value as typeof shot.shotType,
                                            })}
                                        >
                                            {['close_up', 'medium', 'wide', 'establishing', 'insert', 'tracking'].map((type) => (
                                                <option key={type} value={type}>{type}</option>
                                            ))}
                                        </select>
                                    </td>
                                    <td className="st-mono">{shot.numFrames}</td>
                                    <td>{shot.status || 'pending'}</td>
                                    <td>{shot.videoUrl ? 'selected' : '—'}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>

                    <div className="st-panel" style={{ marginTop: 14 }}>
                        <h3 className="st-panel-title">Typed links · {boardLinks.length}</h3>
                        {boardLinks.length === 0 ? (
                            <span className="st-value" style={{ color: 'var(--st-muted)' }}>
                                No links yet. Use Link mode on the board.
                            </span>
                        ) : (
                            <div className="st-link-list">
                                {boardLinks.map((link) => {
                                    const source = project.shots.find((shot) => shot.id === link.source);
                                    const target = project.shots.find((shot) => shot.id === link.target);
                                    return (
                                        <div className="st-link-row" key={link.id}>
                                            <span className="st-chip">{link.kind}</span>
                                            <span>{source?.action || source?.prompt || link.source}</span>
                                            <span aria-hidden="true">→</span>
                                            <span>{target?.action || target?.prompt || link.target}</span>
                                            <button
                                                type="button"
                                                className="st-icon-button"
                                                aria-label={`Remove ${link.kind} link`}
                                                onClick={() => removeLink(link.id)}
                                            >
                                                <X size={12} />
                                            </button>
                                        </div>
                                    );
                                })}
                            </div>
                        )}
                    </div>
                </div>
            ) : (
                <div className="st-script-wrap">
                    <div className="st-script">
                        {project.scriptContent?.trim()
                            ? project.scriptContent
                            : project.shots
                                .map((shot, index) => `${index + 1}. ${shot.action || shot.prompt || ''}${shot.dialogue ? `\n   ${shot.character || 'CHARACTER'}: ${shot.dialogue}` : ''}`)
                                .join('\n\n') || 'No script text yet. Paste a script in the legacy editor or import it in F1.'}
                    </div>
                </div>
            )}
        </div>
    );
}
