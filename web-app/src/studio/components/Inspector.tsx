import { Loader2, Play, RotateCcw, Sparkles, X } from 'lucide-react';
import { useTimelineStore } from '../../stores/timelineStore';
import { useStudioEditor } from '../state/editorStore';
import {
    useStudioCancel,
    useStudioElements,
    useStudioGenerate,
    useStudioSelection,
} from '../data/useStudioData';

const TABS = ['shot', 'prompt', 'takes', 'audio', 'review', 'history'] as const;

export function Inspector() {
    const { selectedShotId, updateShot, addToast } = useStudioSelection();
    const project = useTimelineStore((state) => state.project);
    const shot = project.shots.find((item) => item.id === selectedShotId) || null;
    const elements = useStudioElements(project.id);
    const generate = useStudioGenerate();
    const cancel = useStudioCancel();
    const { inspectorTab, setInspectorTab, setInspectorOpen, inspectorOpen } = useStudioEditor();

    const matchedElements = !shot?.matchedElements?.length
        ? []
        : shot.matchedElements
            .map((match) => elements.find((element) => element.id === match.element_id) || null)
            .filter(Boolean)
            .slice(0, 6);

    if (!inspectorOpen) return null;

    return (
        <aside className="st-inspector" aria-label="Inspector">
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

                        {inspectorTab === 'takes' && (
                            <div className="st-inspector-section">
                                <span className="st-label">Current take</span>
                                <div className="st-value">
                                    {shot.videoUrl ? 'One selected output. Immutable take history lands in F4.' : 'No generated take yet.'}
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
                                    Regeneration currently replaces the selected URL until the Take model lands.
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
                                <span className="st-label">Audio</span>
                                <div className="st-value">
                                    The render path now mixes V1/V2 embedded audio and A1/A2. Per-clip gain/fades arrive with the new timeline inspector.
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
