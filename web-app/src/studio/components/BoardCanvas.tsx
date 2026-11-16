import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Grid3X3, List, Maximize2, Map, Minus, Plus, RotateCcw, Search, Pin } from 'lucide-react';
import { useShallow } from 'zustand/react/shallow';
import { useTimelineStore } from '../../stores/timelineStore';
import { useStudioEditor } from '../state/editorStore';
import { ShotCard } from './ShotCard';

const CARD_W = 300;
const CARD_H = 238;
const GAP_X = 44;
const GAP_Y = 42;
const COLS = 4;

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
    } = useStudioEditor();
    const hostRef = useRef<HTMLDivElement>(null);
    const [size, setSize] = useState({ width: 900, height: 600 });
    const drag = useRef<{ x: number; y: number; cameraX: number; cameraY: number } | null>(null);
    const cardDrag = useRef<{ id: string; x: number; y: number; origX: number; origY: number } | null>(null);
    const [query, setQuery] = useState('');

    const shots = useMemo(() => {
        const q = query.trim().toLowerCase();
        if (!q) return project.shots;
        return project.shots.filter((shot) =>
            `${shot.action || ''} ${shot.prompt || ''} ${shot.character || ''} ${shot.dialogue || ''}`
                .toLowerCase()
                .includes(q),
        );
    }, [project.shots, query]);

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
    };

    const endDrag = () => {
        drag.current = null;
        cardDrag.current = null;
    };

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
                <span className="st-spacer" />
                <input
                    className="st-input"
                    style={{ width: 180, height: 27, padding: '3px 8px' }}
                    placeholder="Filter shots…"
                    value={query}
                    onChange={(event) => setQuery(event.target.value)}
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
                            {shots.map((shot, index) => {
                                const position = boardPositions[shot.id] || {
                                    x: (index % COLS) * (CARD_W + GAP_X),
                                    y: Math.floor(index / COLS) * (CARD_H + GAP_Y),
                                    width: CARD_W,
                                    height: CARD_H,
                                };
                                return (
                                    <ShotCard
                                        key={shot.id}
                                        shot={shot}
                                        index={index}
                                        position={position}
                                        selected={selectedShotId === shot.id}
                                        pinned={!!boardPinned[shot.id]}
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
