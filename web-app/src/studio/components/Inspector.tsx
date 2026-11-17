import { useRef } from 'react';
import { Loader2, Play, RotateCcw, Sparkles, Volume2, VolumeX, X } from 'lucide-react';
import { useTimelineStore } from '../../stores/timelineStore';
import { getAssetUrl } from '../../config';
import { computeTimelineLayout } from '../../utils/timelineUtils';
import {
    opacityAtFrame, removeKeyframeAt, sanitizeKeyframes, upsertKeyframe,
} from '../timeline/keyframes';
import { useStudioEditor } from '../state/editorStore';
import {
    useStudioCancel,
    useStudioElements,
    useStudioGenerate,
    useStudioSelectTake,
    useStudioSelection,
    useStudioTakes,
} from '../data/useStudioData';

const TABS = ['shot', 'prompt', 'motion', 'takes', 'audio', 'review', 'history'] as const;

export function Inspector() {
    const { selectedShotId, updateShot, addToast, currentTime } = useStudioSelection();
    const project = useTimelineStore((state) => state.project);
    const shot = project.shots.find((item) => item.id === selectedShotId) || null;
    const keyframes = sanitizeKeyframes(shot?.keyframes || []);
    const clip = shot ? computeTimelineLayout(project).find((item) => item.id === shot.id) : null;
    const fps = project.fps || 25;
    const localFrame = shot
        ? Math.max(0, Math.min(
            Math.max(0, (shot.numFrames || 1) - 1),
            clip ? Math.round((currentTime - clip.start) * fps) : 0,
        ))
        : 0;
    const elements = useStudioElements(project.id);
    const generate = useStudioGenerate();
    const cancel = useStudioCancel();
    const { takes, loading: takesLoading, refresh: refreshTakes } = useStudioTakes(shot?.id);
    const selectTake = useStudioSelectTake();
    const {
        inspectorTab, setInspectorTab, setInspectorOpen, inspectorOpen,
        inspectorWidth, setInspectorWidth,
    } = useStudioEditor();
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
                            <div className="st-inspector-section">
                                <span className="st-label">Opacity keyframes · frame {localFrame}</span>
                                <div className="st-value" style={{ color: 'var(--st-muted)', fontSize: 10 }}>
                                    Piecewise-linear alpha, burned into the render. V2 fades over V1; V1 fades through black.
                                </div>
                                <div className="st-button-row" style={{ marginTop: 8 }}>
                                    <button
                                        type="button"
                                        className="st-button"
                                        data-size="sm"
                                        data-variant="primary"
                                        onClick={() => updateShot(shot.id, {
                                            keyframes: upsertKeyframe(keyframes, {
                                                frame: localFrame,
                                                opacity: opacityAtFrame(keyframes, localFrame),
                                            }),
                                        })}
                                    >
                                        Add at {localFrame}
                                    </button>
                                    <button
                                        type="button"
                                        className="st-button"
                                        data-size="sm"
                                        data-variant="ghost"
                                        disabled={keyframes.length === 0}
                                        onClick={() => updateShot(shot.id, { keyframes: [] })}
                                    >
                                        Clear
                                    </button>
                                </div>
                                <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginTop: 10 }}>
                                    {keyframes.length === 0 && (
                                        <span className="st-value" style={{ color: 'var(--st-faint)', fontSize: 10 }}>
                                            No keyframes. The clip stays fully opaque.
                                        </span>
                                    )}
                                    {keyframes.map((keyframe) => (
                                        <div key={keyframe.frame} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                                            <span className="st-mono" style={{ width: 44, fontSize: 10 }}>
                                                f{keyframe.frame}
                                            </span>
                                            <input
                                                type="range"
                                                min={0}
                                                max={100}
                                                step={1}
                                                value={Math.round(keyframe.opacity * 100)}
                                                aria-label={`Opacity at frame ${keyframe.frame}`}
                                                style={{ flex: 1, accentColor: 'var(--st-accent)' }}
                                                onChange={(event) => updateShot(shot.id, {
                                                    keyframes: upsertKeyframe(keyframes, {
                                                        frame: keyframe.frame,
                                                        opacity: Number(event.target.value) / 100,
                                                    }),
                                                })}
                                            />
                                            <span className="st-mono" style={{ width: 34, fontSize: 10, textAlign: 'right' }}>
                                                {Math.round(keyframe.opacity * 100)}%
                                            </span>
                                            <button
                                                type="button"
                                                className="st-icon-button"
                                                aria-label={`Remove keyframe at ${keyframe.frame}`}
                                                onClick={() => updateShot(shot.id, {
                                                    keyframes: removeKeyframeAt(keyframes, keyframe.frame),
                                                })}
                                            >
                                                <X size={11} />
                                            </button>
                                        </div>
                                    ))}
                                </div>
                            </div>
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
