import { useRef } from 'react';
import { useShallow } from 'zustand/react/shallow';
import { Loader2, Play, RotateCcw, Sparkles, Star, Volume2, VolumeX, X } from 'lucide-react';
import { useStudioDocumentStore } from '../state/documentStore';
import { getAssetUrl } from '../../config';
import { MotionPanel } from './MotionPanel';
import { useStudioEditor } from '../state/editorStore';
import {
    useStudioCancel,
    useStudioElements,
    useStudioFavoriteTake,
    useStudioGenerate,
    useStudioSelectTake,
    useStudioSelection,
    useStudioTakes,
} from '../data/useStudioData';

const TABS = ['shot', 'prompt', 'motion', 'takes', 'audio', 'effects', 'review', 'history'] as const;

export function Inspector() {
    const { selectedShotId, updateShot, addToast } = useStudioSelection();
    const project = useStudioDocumentStore((state) => state.project);
    const shot = project.shots.find((item) => item.id === selectedShotId) || null;
    const elements = useStudioElements(project.id);
    const generate = useStudioGenerate();
    const cancel = useStudioCancel();
    const { takes, loading: takesLoading, refresh: refreshTakes } = useStudioTakes(shot?.id);
    const selectTake = useStudioSelectTake();
    const favoriteTake = useStudioFavoriteTake();
    const {
        inspectorTab, setInspectorTab, setInspectorOpen, inspectorOpen,
        inspectorWidth, setInspectorWidth,
    } = useStudioEditor(useShallow((state) => ({
        inspectorTab: state.inspectorTab,
        setInspectorTab: state.setInspectorTab,
        setInspectorOpen: state.setInspectorOpen,
        inspectorOpen: state.inspectorOpen,
        inspectorWidth: state.inspectorWidth,
        setInspectorWidth: state.setInspectorWidth,
    })));
    const resizeState = useRef<{ startX: number; startWidth: number } | null>(null);

    const matchedElements = !shot?.matchedElements?.length
        ? []
        : shot.matchedElements
            .map((match) => elements.find((element) => element.id === match.element_id) || null)
            .filter(Boolean)
            .slice(0, 6);

    if (!inspectorOpen) return null;

    return (
        <aside
            className="st-inspector"
            aria-label="Inspector"
            style={{ width: inspectorWidth, flexBasis: inspectorWidth }}
        >
            <div
                className="st-inspector-resize"
                role="separator"
                aria-orientation="vertical"
                aria-label="Resize inspector"
                onPointerDown={(event) => {
                    event.preventDefault();
                    resizeState.current = { startX: event.clientX, startWidth: inspectorWidth };
                    event.currentTarget.setPointerCapture(event.pointerId);
                }}
                onPointerMove={(event) => {
                    if (!resizeState.current) return;
                    const delta = resizeState.current.startX - event.clientX;
                    setInspectorWidth(resizeState.current.startWidth + delta);
                }}
                onPointerUp={(event) => {
                    resizeState.current = null;
                    event.currentTarget.releasePointerCapture(event.pointerId);
                }}
                onPointerCancel={() => { resizeState.current = null; }}
            />
            <div className="st-inspector-head">
                <div className="st-inspector-tabs" role="tablist" aria-label="Inspector sections">
                    {TABS.map((tab) => (
                        <button
                            key={tab}
                            type="button"
                            role="tab"
                            aria-selected={inspectorTab === tab}
                            className="st-inspector-tab"
                            data-active={inspectorTab === tab}
                            onClick={() => setInspectorTab(tab)}
                        >
                            {tab}
                        </button>
                    ))}
                </div>
                <button type="button" className="st-icon-button" onClick={() => setInspectorOpen(false)} aria-label="Close inspector">
                    <X size={14} />
                </button>
            </div>

            <div className="st-inspector-body">
                {!shot ? (
                    <div className="st-empty">
                        <strong>No shot selected</strong>
                        <span>Select a card on the board or a clip on the timeline.</span>
                    </div>
                ) : (
                    <>
                        <div className="st-inspector-section">
                            <span className="st-label">Shot {typeof shot.index === 'number' ? shot.index + 1 : ''}</span>
                            <div className="st-value" style={{ fontWeight: 650 }}>
                                {shot.action || shot.prompt || 'Untitled shot'}
                            </div>
                            <div className="st-chip-row" style={{ marginTop: 8 }}>
                                {shot.shotType && <span className="st-chip">{shot.shotType}</span>}
                                <span className="st-chip">{shot.width}×{shot.height}</span>
                                <span className="st-chip">{shot.numFrames}f</span>
                                <span className="st-chip">seed {shot.seed}</span>
                            </div>
                        </div>

                        {inspectorTab === 'shot' && (
                            <>
                                <div className="st-inspector-section">
                                    <label className="st-label" htmlFor="studio-prompt">Prompt</label>
                                    <textarea
                                        id="studio-prompt"
                                        className="st-textarea"
                                        value={shot.prompt}
                                        onChange={(event) => updateShot(shot.id, { prompt: event.target.value })}
                                    />
                                </div>
                                <div className="st-inspector-section">
                                    <label className="st-label" htmlFor="studio-negative">Negative prompt</label>
                                    <textarea
                                        id="studio-negative"
                                        className="st-textarea"
                                        style={{ minHeight: 58 }}
                                        value={shot.negativePrompt}
                                        onChange={(event) => updateShot(shot.id, { negativePrompt: event.target.value })}
                                    />
                                </div>
                                <div className="st-inspector-section">
                                    <span className="st-label">Parameters</span>
                                    <div className="st-row">
                                        <label>
                                            <span className="st-label">Seed</span>
                                            <input
                                                className="st-input"
                                                type="number"
                                                value={shot.seed}
                                                onChange={(event) => updateShot(shot.id, { seed: Number(event.target.value) })}
                                            />
                                        </label>
                                        <label>
                                            <span className="st-label">Frames</span>
                                            <input
                                                className="st-input"
                                                type="number"
                                                value={shot.numFrames}
                                                onChange={(event) => updateShot(shot.id, { numFrames: Number(event.target.value) })}
                                            />
                                        </label>
                                    </div>
                                    <div className="st-row" style={{ marginTop: 8 }}>
                                        <label>
                                            <span className="st-label">Width</span>
                                            <input
                                                className="st-input"
                                                type="number"
                                                value={shot.width}
                                                onChange={(event) => updateShot(shot.id, { width: Number(event.target.value) })}
                                            />
                                        </label>
                                        <label>
                                            <span className="st-label">Height</span>
                                            <input
                                                className="st-input"
                                                type="number"
                                                value={shot.height}
                                                onChange={(event) => updateShot(shot.id, { height: Number(event.target.value) })}
                                            />
                                        </label>
                                    </div>
                                    <label style={{ display: 'block', marginTop: 8 }}>
                                        <span className="st-label">Speed · timeline duration</span>
                                        <select
                                            className="st-select"
                                            value={shot.speed ?? 1}
                                            onChange={(event) => updateShot(shot.id, { speed: Number(event.target.value) })}
                                        >
                                            {[0.5, 0.75, 1, 1.25, 1.5, 2, 3, 4].map((rate) => (
                                                <option key={rate} value={rate}>{rate}×</option>
                                            ))}
                                        </select>
                                    </label>
                                </div>
                                <div className="st-inspector-section">
                                    <span className="st-label">Visual references</span>
                                    <div className="st-chip-row">
                                        {matchedElements.length === 0 && <span className="st-value" style={{ color: 'var(--st-muted)' }}>No matched elements.</span>}
                                        {matchedElements.map((element) => (
                                            <span className="st-chip" key={element!.id} title={element!.description}>
                                                @{element!.trigger_word.replace(/^@/, '')} · {element!.name}
                                            </span>
                                        ))}
                                    </div>
                                </div>
                            </>
                        )}

                        {inspectorTab === 'prompt' && (
                            <div className="st-inspector-section">
                                <span className="st-label">Effective prompt</span>
                                <div className="st-value">{shot.enhancedPromptResult || shot.prompt || 'No prompt yet.'}</div>
                                {shot.enhancedPromptResult && (
                                    <>
                                        <span className="st-label" style={{ marginTop: 12 }}>Raw prompt</span>
                                        <div className="st-value" style={{ color: 'var(--st-muted)' }}>{shot.prompt}</div>
                                    </>
                                )}
                            </div>
                        )}

                        {inspectorTab === 'motion' && (
                            <MotionPanel shot={shot} project={project} updateShot={updateShot} />
                        )}

                        {inspectorTab === 'takes' && (
                            <div className="st-inspector-section">
                                <span className="st-label">Takes {takesLoading ? '· loading…' : `· ${takes.length}`}</span>
                                <div className="st-take-grid" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(130px, 1fr))' }}>
                                    {takes.map((take) => (
                                        <article className="st-take" key={take.job_id} style={{ borderColor: take.selected ? 'var(--st-accent)' : undefined }}>
                                            <div className="st-take-media">
                                                {take.thumbnail_url || take.url ? (
                                                    <img src={getAssetUrl(take.thumbnail_url || take.url)} alt="" loading="lazy" />
                                                ) : (
                                                    <div className="st-stage-empty"><RotateCcw size={14} /></div>
                                                )}
                                            </div>
                                            <div className="st-take-body">
                                                <div className="st-mono" style={{ color: 'var(--st-secondary)' }}>{take.job_id.slice(0, 10)}</div>
                                                <div>{take.model || take.type} · seed {take.seed ?? '—'}</div>
                                                <button
                                                    type="button"
                                                    className="st-button"
                                                    data-size="sm"
                                                    style={{ marginTop: 6 }}
                                                    disabled={take.selected}
                                                    onClick={async () => {
                                                        if (await selectTake(shot, take)) await refreshTakes();
                                                    }}
                                                >
                                                    {take.selected ? 'Selected' : 'Select'}
                                                </button>
                                                <button
                                                    type="button"
                                                    className="st-button"
                                                    data-size="sm"
                                                    data-variant={take.favorite ? 'primary' : 'ghost'}
                                                    style={{ marginTop: 6 }}
                                                    aria-label={take.favorite ? 'Clear favorite take' : 'Favorite take'}
                                                    onClick={async () => {
                                                        if (await favoriteTake(shot, take.favorite ? null : take)) await refreshTakes();
                                                    }}
                                                >
                                                    <Star size={11} /> {take.favorite ? 'Favorite' : 'Favorite'}
                                                </button>
                                            </div>
                                        </article>
                                    ))}
                                    {takes.length === 0 && !takesLoading && (
                                        <div className="st-empty" style={{ minHeight: 100, gridColumn: '1 / -1' }}>
                                            <strong>No takes yet</strong>
                                            <span>Generate once; old outputs are kept as takes.</span>
                                        </div>
                                    )}
                                </div>
                                <div className="st-button-row" style={{ marginTop: 10 }}>
                                    <button
                                        type="button"
                                        className="st-button"
                                        data-variant="primary"
                                        disabled={shot.isGenerating}
                                        onClick={() => void generate(shot, true)}
                                    >
                                        {shot.isGenerating ? <Loader2 size={12} className="animate-spin" /> : <RotateCcw size={12} />}
                                        {shot.isGenerating ? 'Generating…' : 'Generate new take'}
                                    </button>
                                    {shot.isGenerating && (
                                        <button type="button" className="st-button" data-variant="danger" onClick={() => void cancel(shot)}>
                                            Cancel
                                        </button>
                                    )}
                                </div>
                                <p className="st-value" style={{ marginTop: 8, color: 'var(--st-faint)', fontSize: 10 }}>
                                    Takes are completed jobs; selecting one repoints the shot without deleting the others.
                                </p>
                            </div>
                        )}

                        {inspectorTab === 'effects' && (
                            <div className="st-inspector-section">
                                <span className="st-label">Clip effects</span>
                                <div className="st-meta-grid">
                                    <dt>Speed</dt><dd>{shot.speed ?? 1}×</dd>
                                    <dt>Opacity keys</dt><dd>{shot.keyframes?.length ?? 0}</dd>
                                    <dt>Fade in/out</dt><dd>{(shot.audioFadeIn ?? 0).toFixed(1)}s / {(shot.audioFadeOut ?? 0).toFixed(1)}s</dd>
                                    <dt>Gain</dt><dd>{Math.round((shot.audioGain ?? 1) * 100)}%{shot.audioMuted ? ' · muted' : ''}</dd>
                                    <dt>Transition</dt><dd>{shot.keyframes?.some((key) => (key.scale ?? 1) !== 1) ? 'transform' : 'none'}</dd>
                                </div>
                                <div className="st-button-row" style={{ marginTop: 10 }}>
                                    <button type="button" className="st-button" data-size="sm" onClick={() => setInspectorTab('motion')}>
                                        Edit keyframes
                                    </button>
                                    <button type="button" className="st-button" data-size="sm" onClick={() => setInspectorTab('audio')}>
                                        Edit audio
                                    </button>
                                    <button
                                        type="button"
                                        className="st-button"
                                        data-size="sm"
                                        data-variant="ghost"
                                        onClick={() => updateShot(shot.id, {
                                            speed: 1,
                                            audioGain: 1,
                                            audioMuted: false,
                                            audioFadeIn: 0,
                                            audioFadeOut: 0,
                                            keyframes: [],
                                        })}
                                    >
                                        Reset effects
                                    </button>
                                </div>
                                <p className="st-value" style={{ color: 'var(--st-faint)', fontSize: 9, marginTop: 8 }}>
                                    Effects are baked by the render: speed, fades, gain, and keyframe alpha/transform. No external effect service.
                                </p>
                            </div>
                        )}

                        {inspectorTab === 'review' && (
                            <div className="st-inspector-section">
                                <span className="st-label">Review</span>
                                <div className="st-value">
                                    Version stacks, frame comments, and approvals arrive in F4. The current shot is still the source of truth.
                                </div>
                                <div className="st-meta-grid" style={{ marginTop: 10 }}>
                                    <dt>Status</dt><dd>{shot.status || 'pending'}</dd>
                                    <dt>Last job</dt><dd className="st-mono">{shot.lastJobId || '—'}</dd>
                                </div>
                            </div>
                        )}

                        {inspectorTab === 'audio' && (
                            <div className="st-inspector-section">
                                <span className="st-label">Embedded audio</span>
                                <div className="st-value" style={{ color: 'var(--st-muted)', fontSize: 10 }}>
                                    Gain and mute are written into the computed cut and applied by FFmpeg.
                                </div>
                                <label style={{ display: 'block', marginTop: 10 }}>
                                    <span className="st-label">
                                        Gain · {Math.round((shot.audioGain ?? 1) * 100)}%
                                    </span>
                                    <input
                                        type="range"
                                        min={0}
                                        max={200}
                                        step={5}
                                        value={Math.round((shot.audioGain ?? 1) * 100)}
                                        onChange={(event) => updateShot(shot.id, {
                                            audioGain: Number(event.target.value) / 100,
                                        })}
                                        style={{ width: '100%', accentColor: 'var(--st-accent)' }}
                                    />
                                </label>
                                <div className="st-button-row" style={{ marginTop: 8 }}>
                                    <button
                                        type="button"
                                        className="st-button"
                                        data-size="sm"
                                        data-variant={shot.audioMuted ? 'danger' : undefined}
                                        onClick={() => updateShot(shot.id, { audioMuted: !shot.audioMuted })}
                                    >
                                        {shot.audioMuted ? <VolumeX size={12} /> : <Volume2 size={12} />}
                                        {shot.audioMuted ? 'Muted' : 'Mute clip'}
                                    </button>
                                    <button
                                        type="button"
                                        className="st-button"
                                        data-size="sm"
                                        data-variant="ghost"
                                        onClick={() => updateShot(shot.id, { audioGain: 1, audioMuted: false })}
                                    >
                                        Reset
                                    </button>
                                </div>
                                <div className="st-row" style={{ marginTop: 10 }}>
                                    <label>
                                        <span className="st-label">Fade in · {(shot.audioFadeIn ?? 0).toFixed(1)}s</span>
                                        <input
                                            type="range"
                                            min={0}
                                            max={2}
                                            step={0.1}
                                            value={shot.audioFadeIn ?? 0}
                                            style={{ width: '100%', accentColor: 'var(--st-accent)' }}
                                            onChange={(event) => updateShot(shot.id, {
                                                audioFadeIn: Number(event.target.value),
                                            })}
                                        />
                                    </label>
                                    <label>
                                        <span className="st-label">Fade out · {(shot.audioFadeOut ?? 0).toFixed(1)}s</span>
                                        <input
                                            type="range"
                                            min={0}
                                            max={2}
                                            step={0.1}
                                            value={shot.audioFadeOut ?? 0}
                                            style={{ width: '100%', accentColor: 'var(--st-accent)' }}
                                            onChange={(event) => updateShot(shot.id, {
                                                audioFadeOut: Number(event.target.value),
                                            })}
                                        />
                                    </label>
                                </div>
                                <div className="st-meta-grid" style={{ marginTop: 10 }}>
                                    <dt>Track</dt><dd>{shot.trackIndex === 2 ? 'A1' : shot.trackIndex === 1 ? 'V2' : 'V1'}</dd>
                                    <dt>Frames</dt><dd className="st-mono">{shot.numFrames}</dd>
                                    <dt>Trim</dt><dd className="st-mono">{shot.trimIn || 0} / {shot.trimOut || 0}</dd>
                                </div>
                            </div>
                        )}

                        {inspectorTab === 'history' && (
                            <div className="st-inspector-section">
                                <span className="st-label">History</span>
                                <div className="st-value">Project revision {project.revision ?? 0}. Undo is document-scoped in the legacy adapter; immutable takes are next.</div>
                            </div>
                        )}

                        <div className="st-inspector-section">
                            <div className="st-button-row">
                                <button type="button" className="st-button" data-variant="primary" disabled={shot.isGenerating} onClick={() => void generate(shot)}>
                                    {shot.isGenerating ? <Loader2 size={12} className="animate-spin" /> : <Play size={12} />}
                                    {shot.isGenerating ? 'Queued' : 'Generate'}
                                </button>
                                <button
                                    type="button"
                                    className="st-button"
                                    onClick={() => addToast('Open the legacy inspector for masking/tracking tools', 'info')}
                                >
                                    <Sparkles size={12} /> Edit frame
                                </button>
                            </div>
                        </div>
                    </>
                )}
            </div>
        </aside>
    );
}
