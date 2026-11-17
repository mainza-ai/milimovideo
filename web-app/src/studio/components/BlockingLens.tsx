import { useEffect, useRef, useState } from 'react';
import { Camera, Plus, RotateCcw, Trash2, User, X } from 'lucide-react';
import type { Project, Shot, BlockingNode } from '../../stores/types';

interface Props {
    shot: Shot;
    project: Project;
    updateShot: (id: string, updates: Partial<Shot>) => void;
    onClose: () => void;
}

function defaultNodes(shot: Shot): BlockingNode[] {
    return [
        { id: 'camera', type: 'camera', label: 'Camera', x: 50, y: 88, rotation: 0 },
        { id: 'talent', type: 'character', label: shot.character || 'Talent', x: 50, y: 45, rotation: 0 },
    ];
}

export function BlockingLens({ shot, project, updateShot, onClose }: Props) {
    const [nodes, setNodes] = useState<BlockingNode[]>(shot.blocking?.length ? shot.blocking : defaultNodes(shot));
    const [selectedId, setSelectedId] = useState<string | null>(nodes[0]?.id || null);
    const [drag, setDrag] = useState<{ id: string } | null>(null);
    const svgRef = useRef<SVGSVGElement>(null);

    useEffect(() => {
        setNodes(shot.blocking?.length ? shot.blocking : defaultNodes(shot));
    }, [shot.id, shot.blocking]);

    const selected = nodes.find((node) => node.id === selectedId) || null;
    const fps = project.fps || 25;

    const commit = (next: BlockingNode[]) => {
        setNodes(next);
        updateShot(shot.id, { blocking: next });
    };

    const pointFromEvent = (event: React.PointerEvent) => {
        const rect = svgRef.current?.getBoundingClientRect();
        if (!rect) return null;
        return {
            x: Math.max(2, Math.min(98, ((event.clientX - rect.left) / rect.width) * 100)),
            y: Math.max(2, Math.min(98, ((event.clientY - rect.top) / rect.height) * 100)),
        };
    };

    const addNode = (type: BlockingNode['type']) => {
        const node: BlockingNode = {
            id: `node_${Date.now()}`,
            type,
            label: type === 'character' ? 'New character' : type === 'object' ? 'Object' : 'Camera',
            x: 50,
            y: 50,
            rotation: 0,
        };
        commit([...nodes, node]);
        setSelectedId(node.id);
    };

    return (
        <div className="st-lens-backdrop" role="dialog" aria-modal="true" aria-label="Blocking lens">
            <div className="st-lens">
                <div className="st-lens-head">
                    <div>
                        <strong>Blocking lens · {shot.action || shot.prompt || shot.id.slice(0, 10)}</strong>
                        <div style={{ color: 'var(--st-faint)', fontSize: 9 }}>
                            Top-down camera and talent diagram for this shot. Positions persist with the shot.
                        </div>
                    </div>
                    <button type="button" className="st-icon-button" aria-label="Close blocking lens" onClick={onClose}>
                        <X size={14} />
                    </button>
                </div>

                <div className="st-blocking-body">
                    <svg
                        ref={svgRef}
                        className="st-blocking-stage"
                        viewBox="0 0 100 100"
                        role="img"
                        aria-label="Top-down blocking stage"
                        onPointerMove={(event) => {
                            if (!drag) return;
                            const point = pointFromEvent(event);
                            if (!point) return;
                            setNodes((current) => current.map((node) => (
                                node.id === drag.id ? { ...node, ...point } : node
                            )));
                        }}
                        onPointerUp={() => {
                            if (drag) {
                                setDrag(null);
                                updateShot(shot.id, { blocking: nodes });
                            }
                        }}
                        onPointerLeave={() => {
                            if (drag) {
                                setDrag(null);
                                updateShot(shot.id, { blocking: nodes });
                            }
                        }}
                    >
                        <defs>
                            <pattern id="blocking-grid" width="10" height="10" patternUnits="userSpaceOnUse">
                                <path d="M 10 0 L 0 0 0 10" fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="0.4" />
                            </pattern>
                        </defs>
                        <rect width="100" height="100" fill="#0a0b0c" />
                        <rect width="100" height="100" fill="url(#blocking-grid)" />
                        <line x1="0" y1="100" x2="100" y2="0" stroke="rgba(255,255,255,0.06)" strokeWidth="0.4" />
                        <line x1="0" y1="0" x2="100" y2="100" stroke="rgba(255,255,255,0.06)" strokeWidth="0.4" />
                        {nodes.map((node) => (
                            <g
                                key={node.id}
                                transform={`translate(${node.x} ${node.y})`}
                                onPointerDown={(event) => {
                                    event.stopPropagation();
                                    setSelectedId(node.id);
                                    setDrag({ id: node.id });
                                    (event.currentTarget as unknown as SVGGElement).setPointerCapture?.(event.pointerId);
                                }}
                                style={{ cursor: 'grab' }}
                            >
                                {node.type === 'camera' && (
                                    <g transform={`rotate(${node.rotation || 0})`}>
                                        <path d="M -3 3 L 0 -4 L 3 3 Z" fill="var(--st-accent)" />
                                        <path d="M 0 -4 L -7 -11 M 0 -4 L 0 -12 M 0 -4 L 7 -11" stroke="rgba(6,178,235,0.55)" strokeWidth="0.6" fill="none" />
                                    </g>
                                )}
                                {node.type === 'character' && (
                                    <circle r="4" fill="rgba(154,123,255,0.28)" stroke="var(--st-v2)" strokeWidth="0.7" />
                                )}
                                {node.type === 'object' && (
                                    <rect x="-3" y="-3" width="6" height="6" fill="rgba(245,184,75,0.22)" stroke="var(--st-warning)" strokeWidth="0.6" />
                                )}
                                <text textAnchor="middle" y="9" fill="var(--st-secondary)" fontSize="3">
                                    {node.label}
                                </text>
                            </g>
                        ))}
                    </svg>

                    <aside>
                        <div className="st-button-row" style={{ marginBottom: 8 }}>
                            <button type="button" className="st-button" data-size="sm" onClick={() => addNode('character')}>
                                <User size={11} /> Character
                            </button>
                            <button type="button" className="st-button" data-size="sm" onClick={() => addNode('camera')}>
                                <Camera size={11} /> Camera
                            </button>
                            <button type="button" className="st-button" data-size="sm" onClick={() => addNode('object')}>
                                <Plus size={11} /> Object
                            </button>
                            <button
                                type="button"
                                className="st-button"
                                data-size="sm"
                                data-variant="ghost"
                                onClick={() => commit(defaultNodes(shot))}
                            >
                                <RotateCcw size={11} /> Reset
                            </button>
                        </div>

                        {selected ? (
                            <>
                                <label className="st-label" htmlFor="blocking-label">Label</label>
                                <input
                                    id="blocking-label"
                                    className="st-input"
                                    value={selected.label}
                                    onChange={(event) => setNodes(nodes.map((node) => (
                                        node.id === selected.id ? { ...node, label: event.target.value } : node
                                    )))}
                                    onBlur={() => updateShot(shot.id, { blocking: nodes })}
                                />
                                {selected.type === 'camera' && (
                                    <label style={{ display: 'block', marginTop: 8 }}>
                                        <span className="st-label">Rotation · {Math.round(selected.rotation || 0)}°</span>
                                        <input
                                            type="range"
                                            min={-180}
                                            max={180}
                                            value={selected.rotation || 0}
                                            style={{ width: '100%', accentColor: 'var(--st-accent)' }}
                                            onChange={(event) => {
                                                const next = nodes.map((node) => (
                                                    node.id === selected.id ? { ...node, rotation: Number(event.target.value) } : node
                                                ));
                                                commit(next);
                                            }}
                                        />
                                    </label>
                                )}
                                <button
                                    type="button"
                                    className="st-button"
                                    data-size="sm"
                                    data-variant="danger"
                                    style={{ marginTop: 8 }}
                                    onClick={() => {
                                        commit(nodes.filter((node) => node.id !== selected.id));
                                        setSelectedId(null);
                                    }}
                                >
                                    <Trash2 size={11} /> Remove
                                </button>
                            </>
                        ) : (
                            <p style={{ color: 'var(--st-muted)', fontSize: 10 }}>Select a node to edit it.</p>
                        )}

                        <h4 style={{ marginTop: 14 }}>Shot context</h4>
                        <dl className="st-meta-grid">
                            <dt>Frames</dt><dd className="st-mono">{shot.numFrames}</dd>
                            <dt>Duration</dt><dd className="st-mono">{(shot.numFrames / fps).toFixed(2)}s</dd>
                            <dt>Type</dt><dd>{shot.shotType || 'medium'}</dd>
                            <dt>Talent</dt><dd>{shot.character || '—'}</dd>
                        </dl>
                    </aside>
                </div>
            </div>
        </div>
    );
}
