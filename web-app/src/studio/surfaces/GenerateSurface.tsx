import { useEffect, useState } from 'react';
import { Film, Image as ImageIcon, Loader2, Play, RotateCcw, Sparkles, Square } from 'lucide-react';
import { useTimelineStore } from '../../stores/timelineStore';
import { useStudioCancel, useStudioElements, useStudioGenerate, useStudioSelection, useAssetUrl } from '../data/useStudioData';

export function GenerateSurface() {
    const project = useTimelineStore((state) => state.project);
    const { selectedShotId, updateShot } = useStudioSelection();
    const elements = useStudioElements(project.id);
    const generate = useStudioGenerate();
    const cancel = useStudioCancel();
    const [promptDraft, setPromptDraft] = useState<string | null>(null);

    const shot = project.shots.find((item) => item.id === selectedShotId) || project.shots[0] || null;
    useEffect(() => { setPromptDraft(null); }, [shot?.id]);
    const mediaUrl = useAssetUrl(shot?.videoUrl || shot?.thumbnailUrl);
    const isImage = !!mediaUrl?.match(/\.(jpg|jpeg|png|webp)$/i);

    if (!shot) {
        return (
            <div className="st-page">
                <div className="st-empty">
                    <strong>No shot selected</strong>
                    <span>Add a shot in the storyboard first.</span>
                </div>
            </div>
        );
    }

    const matched = (shot.matchedElements || [])
        .map((match) => elements.find((element) => element.id === match.element_id) || null)
        .filter(Boolean);

    return (
        <div className="st-shot-workspace">
            <aside className="st-shot-rail">
                <div className="st-spine-head">
                    <div className="st-spine-title">Shot queue</div>
                    <div className="st-value" style={{ color: 'var(--st-muted)', fontSize: 10 }}>
                        {project.shots.filter((item) => item.status === 'completed').length}/{project.shots.length} with takes
                    </div>
                </div>
                <div className="st-spine-list">
                    {project.shots.map((item, index) => (
                        <button
                            key={item.id}
                            type="button"
                            className="st-spine-shot"
                            data-selected={item.id === shot.id}
                            onClick={() => useTimelineStore.getState().selectShot(item.id)}
                        >
                            <span className="st-spine-shot-index st-mono">{index + 1}</span>
                            <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                {item.action || item.prompt || 'Untitled shot'}
                            </span>
                            {item.isGenerating && <Loader2 size={10} className="animate-spin" style={{ marginLeft: 'auto' }} />}
                        </button>
                    ))}
                </div>
            </aside>

            <section className="st-shot-stage">
                <div className="st-page-head">
                    <div>
                        <h2 className="st-page-title">Shot {typeof shot.index === 'number' ? shot.index + 1 : ''}</h2>
                        <p className="st-page-subtitle">
                            {shot.action || shot.prompt || 'Untitled shot'}
                        </p>
                    </div>
                    <div className="st-button-row">
                        <button type="button" className="st-button" data-variant="primary" disabled={shot.isGenerating} onClick={() => void generate(shot)}>
                            {shot.isGenerating ? <Loader2 size={13} className="animate-spin" /> : <Sparkles size={13} />}
                            {shot.isGenerating ? 'Generating…' : 'Generate'}
                        </button>
                        <button type="button" className="st-button" disabled={shot.isGenerating} onClick={() => void generate(shot, true)}>
                            <RotateCcw size={13} /> New take
                        </button>
                        {shot.isGenerating && (
                            <button type="button" className="st-button" data-variant="danger" onClick={() => void cancel(shot)}>
                                <Square size={12} /> Cancel
                            </button>
                        )}
                    </div>
                </div>

                <div className="st-take-grid">
                    <article className="st-take">
                        <div className="st-take-media">
                            {mediaUrl ? (
                                isImage ? <img src={mediaUrl} alt="" /> : <video src={mediaUrl} muted playsInline controls />
                            ) : (
                                <div className="st-stage-empty"><ImageIcon size={18} /><span>No take yet</span></div>
                            )}
                        </div>
                        <div className="st-take-body">
                            <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8 }}>
                                <strong style={{ color: 'var(--st-text)' }}>Current selected take</strong>
                                <span className="st-mono">{shot.numFrames}f</span>
                            </div>
                            <div style={{ marginTop: 4 }}>seed {shot.seed} · {shot.width}×{shot.height} · {shot.pipelineOverride || 'auto'}</div>
                        </div>
                    </article>
                    <article className="st-take" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: 180, borderStyle: 'dashed' }}>
                        <div className="st-empty" style={{ minHeight: 0 }}>
                            <Film size={20} />
                            <strong>Immutable takes: F4</strong>
                            <span>New takes will sit beside this one, not replace it.</span>
                        </div>
                    </article>
                </div>

                <div className="st-panel" style={{ marginTop: 16 }}>
                    <h3 className="st-panel-title">Prompt</h3>
                    <textarea
                        className="st-textarea"
                        value={promptDraft ?? shot.prompt}
                        onChange={(event) => setPromptDraft(event.target.value)}
                        onBlur={() => {
                            if (promptDraft != null && promptDraft !== shot.prompt) {
                                updateShot(shot.id, { prompt: promptDraft });
                                setPromptDraft(null);
                            }
                        }}
                    />
                    <div className="st-button-row" style={{ marginTop: 8 }}>
                        <span className="st-chip">seed {shot.seed}</span>
                        <span className="st-chip">{shot.numFrames} frames</span>
                        <span className="st-chip">{shot.enhancePrompt ? 'enhance' : 'raw prompt'}</span>
                        <span className="st-chip">{shot.upscale ? 'upscale' : 'no upscale'}</span>
                    </div>
                </div>
            </section>

            <aside className="st-shot-params">
                <div className="st-inspector-section">
                    <span className="st-label">Ingredients</span>
                    <div className="st-chip-row">
                        {matched.length === 0 && <span className="st-value" style={{ color: 'var(--st-muted)' }}>No master references matched.</span>}
                        {matched.map((element) => (
                            <span className="st-chip" key={element!.id} title={element!.description}>
                                @{element!.trigger_word.replace(/^@/, '')} · {element!.name}
                            </span>
                        ))}
                    </div>
                </div>
                <div className="st-inspector-section">
                    <span className="st-label">Generation</span>
                    <div className="st-meta-grid">
                        <dt>Model</dt><dd>{shot.pipelineOverride || 'auto'}</dd>
                        <dt>Seed</dt><dd>{shot.seed}</dd>
                        <dt>Frames</dt><dd>{shot.numFrames}</dd>
                        <dt>Resolution</dt><dd>{shot.width}×{shot.height}</dd>
                        <dt>Status</dt><dd>{shot.status || 'pending'}</dd>
                    </div>
                </div>
                <div className="st-inspector-section">
                    <span className="st-label">Reference media</span>
                    <div className="st-button-row">
                        <button type="button" className="st-button" data-size="sm" onClick={() => useTimelineStore.getState().addToast('Drag media from the Library onto a shot in F1', 'info')}>
                            <ImageIcon size={12} /> Add reference
                        </button>
                        <button type="button" className="st-button" data-size="sm" onClick={() => window.location.assign('/')}>
                            <Play size={12} /> Legacy edit tools
                        </button>
                    </div>
                </div>
                <div className="st-inspector-section">
                    <span className="st-label">Honest status</span>
                    <p className="st-value" style={{ color: 'var(--st-faint)', fontSize: 10, lineHeight: 1.55 }}>
                        This surface uses the existing generation endpoints. Take history, approval, and cost preview arrive in F3/F4.
                    </p>
                </div>
            </aside>
        </div>
    );
}
