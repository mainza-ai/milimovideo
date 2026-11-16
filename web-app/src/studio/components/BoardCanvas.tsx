import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
    Grid3X3, Link2, List, Maximize2, Map, Minus, MousePointer2, Plus,
    RotateCcw, Search, SquareDashed, Trash2, Pin,
} from 'lucide-react';
import { useShallow } from 'zustand/react/shallow';
import { useTimelineStore } from '../../stores/timelineStore';
import {
    useStudioEditor, type BoardArea, type BoardPosition, type LinkKind,
} from '../state/editorStore';
import { ShotCard } from './ShotCard';

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
    const { selectedShotId, selectShot, setCurrentTime } = useTimelineStore(useShallow((state) => ({
        selectedShotId: state.selectedShotId,
        selectShot: state.selectShot,
        setCurrentTime: state.setCurrentTime,
    })));
    const {
        storyView, setStoryView, boardCamera, setCamera, boardPositions,
        setBoardPosition, boardPinned, boardMinimapOpen,
        setMinimapOpen, boardFollowSelection, setFollowSelection, resetBoard,
        boardAreas, boardLinks, boardMode, setBoardMode, linkKind, setLinkKind,
        selectedAreaId, selectArea, addArea, updateArea, removeArea, addLink, removeLink,
        boardQuery, setBoardQuery,
    } = useStudioEditor();
    const hostRef = useRef<HTMLDivElement>(null);
    const [size, setSize] = useState({ width: 900, height: 600 });
    const drag = useRef<{ x: number; y: number; cameraX: number; cameraY: number } | null>(null);
    const cardDrag = useRef<{ id: string; x: number; y: number; origX: number; origY: number } | null>(null);
    const areaDrag = useRef<{
        mode: 'create' | 'move' | 'resize';
        id?: string;
        startWorldX: number;
        startWorldY: number;
        orig: BoardArea;
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
        project.shots.forEach((shot, index) => {
            setBoardPosition(shot.id, {
                x: (index % COLS) * (CARD_W + GAP_X),
                y: Math.floor(index / COLS) * (CARD_H + GAP_Y),
                width: CARD_W,
                height: CARD_H,
            });
        });
    }, [project.shots, setBoardPosition]);

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
        const world = toWorld(event.clientX, event.clientY);
        areaDrag.current = { mode: 'move', id: area.id, startWorldX: world.x, startWorldY: world.y, orig: area };
        hostRef.current?.setPointerCapture(event.pointerId);
    };

    const startAreaResize = (event: React.PointerEvent, area: BoardArea) => {
        event.stopPropagation();
        selectArea(area.id);
        const world = toWorld(event.clientX, event.clientY);
        areaDrag.current = { mode: 'resize', id: area.id, startWorldX: world.x, startWorldY: world.y, orig: area };
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
            areaDrag.current = { mode: 'create', startWorldX: world.x, startWorldY: world.y, orig: draft };
            setAreaDraft(draft);
            (event.currentTarget as HTMLElement).setPointerCapture(event.pointerId);
            return;
        }
        if (boardMode === 'link') return;
        selectArea(null);
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
                addLink(linkSource, shotId, linkKind);
                setLinkSource(null);
            } else {
                setLinkSource(null);
            }
            return;
        }
        const position = boardPositions[shotId];
        if (!position) return;
        cardDrag.current = {
            id: shotId,
            x: event.clientX,
            y: event.clientY,
            origX: position.x,
            origY: position.y,
        };
        selectShot(shotId);
    };

    const onPointerMove = (event: React.PointerEvent) => {
        if (drag.current) {
            setCamera({
                ...boardCamera,
                x: drag.current.cameraX + (event.clientX - drag.current.x),
                y: drag.current.cameraY + (event.clientY - drag.current.y),
            });
        }
        if (cardDrag.current) {
            const deltaX = (event.clientX - cardDrag.current.x) / boardCamera.zoom;
            const deltaY = (event.clientY - cardDrag.current.y) / boardCamera.zoom;
            setBoardPosition(cardDrag.current.id, {
                x: Math.round((cardDrag.current.origX + deltaX) / 8) * 8,
                y: Math.round((cardDrag.current.origY + deltaY) / 8) * 8,
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
                    updateArea(dragState.id, {
                        x: Math.round(dragState.orig.x + world.x - dragState.startWorldX),
                        y: Math.round(dragState.orig.y + world.y - dragState.startWorldY),
                    });
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
        if (areaDrag.current?.mode === 'create' && areaDraft) {
            if (areaDraft.width > 40 && areaDraft.height > 40) {
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
        const pad = 260 / Math.max(0.2, boardCamera.zoom);
        const left = -boardCamera.x / boardCamera.zoom - pad;
        const top = -boardCamera.y / boardCamera.zoom - pad;
        const right = left + size.width / boardCamera.zoom + pad * 2;
        const bottom = top + size.height / boardCamera.zoom + pad * 2;
        return shots
            .map((shot, index) => ({
                shot,
                index,
                position: boardPositions[shot.id] || fallbackPosition(index),
            }))
            .filter(({ position }) => (
                position.x + position.width >= left
                && position.x <= right
                && position.y + position.height >= top
                && position.y <= bottom
            ));
    }, [boardCamera, boardPositions, shots, size.height, size.width]);

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
                <button type="button" className="st-button" data-size="sm" onClick={arrange} title="Arrange cards">
                    <Grid3X3 size={13} /> Arrange
                </button>
                <button type="button" className="st-button" data-size="sm" onClick={fit} title="Fit board">
                    <Maximize2 size={13} /> Fit
                </button>
                <button type="button" className="st-button" data-size="sm" data-variant={boardMinimapOpen ? 'primary' : 'ghost'} onClick={() => setMinimapOpen(!boardMinimapOpen)} title="Minimap">
                    <Map size={13} />
                </button>
                <button type="button" className="st-button" data-size="sm" data-variant={boardFollowSelection ? 'primary' : 'ghost'} onClick={() => setFollowSelection(!boardFollowSelection)} title="Follow selection">
                    <Pin size={13} />
                </button>
                <button type="button" className="st-button" data-size="sm" data-variant="ghost" onClick={resetBoard} title="Reset board layout">
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
                                        height: area.height,
                                        borderColor: area.color || 'var(--st-border-strong)',
                                        background: `${area.color || '#06b2eb'}12`,
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
                                    <button
                                        type="button"
                                        className="st-board-area-resize"
                                        aria-label="Resize area"
                                        onPointerDown={(event) => startAreaResize(event, area)}
                                    />
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
                                        selected={selectedShotId === shot.id}
                                        pinned={!!boardPinned[shot.id]}
                                        linkSource={linkSource === shot.id}
                                        onSelect={() => {
                                            selectShot(shot.id);
                                            setCurrentTime(0);
                                        }}
                                        onDragStart={startCardDrag}
                                    />
                                );
                            })}
                        </div>
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
                    <table className="st-table">
                        <caption>Shot list — text equivalent of the board</caption>
                        <thead>
                            <tr>
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
                                    data-selected={selectedShotId === shot.id}
                                    onClick={() => selectShot(shot.id)}
                                >
                                    <td className="st-mono">{index + 1}</td>
                                    <td><strong>{shot.action || shot.prompt || 'Untitled'}</strong></td>
                                    <td>{shot.shotType || 'medium'}</td>
                                    <td className="st-mono">{shot.numFrames}</td>
                                    <td>{shot.status || 'pending'}</td>
                                    <td>{shot.videoUrl ? 'selected' : '—'}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
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
