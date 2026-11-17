import { useEffect, useState } from 'react';
import {
    Film, Image as ImageIcon, ListChecks, Loader2, Play, RotateCcw, Sparkles, Square, Star, X,
} from 'lucide-react';
import { useStudioDocumentStore } from '../state/documentStore';
import { apiUrl, getAssetUrl } from '../../config';
import {
    useAssetUrl,
    useStudioActiveJobs,
    useStudioCancel,
    useStudioElements,
    useStudioFavoriteTake,
    useStudioGenerate,
    useStudioSelectTake,
    useStudioSelection,
    useStudioTakes,
} from '../data/useStudioData';
import type { StudioElement } from '../data/useStudioData';
import { PipelineLens } from '../components/PipelineLens';

export function GenerateSurface() {
    const project = useStudioDocumentStore((state) => state.project);
    const { selectedShotId, updateShot } = useStudioSelection();
    const elements = useStudioElements(project.id);
    const generate = useStudioGenerate();
    const cancel = useStudioCancel();
    const [promptDraft, setPromptDraft] = useState<string | null>(null);
    const [batchBusy, setBatchBusy] = useState(false);
    const [preview, setPreview] = useState<{ url: string | null; status: string } | null>(null);
    const [lensOpen, setLensOpen] = useState(false);
    const { jobs: activeJobs, refresh: refreshJobs } = useStudioActiveJobs(project.id);

    const videoShots = project.shots.filter((item) => item.trackIndex !== 2);
    const selectedVideoShot = videoShots.find((item) => item.id === selectedShotId);
    const shot = selectedVideoShot || videoShots[0] || null;
    useEffect(() => { setPromptDraft(null); }, [shot?.id]);
    const { takes, loading: takesLoading, refresh: refreshTakes } = useStudioTakes(shot?.id);
    const selectTake = useStudioSelectTake();
    const favoriteTake = useStudioFavoriteTake();
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

    const toggleIngredient = (element: StudioElement) => {
        const current = shot.matchedElements || [];
        const exists = current.some((match) => match.element_id === element.id);
        const next = exists
            ? current.filter((match) => match.element_id !== element.id)
            : [...current, {
                element_id: element.id,
                element_name: element.name,
                element_type: element.type as 'character' | 'location' | 'object',
                trigger_word: element.trigger_word,
                confidence: 1,
                match_source: 'manual',
            }];
        updateShot(shot.id, { matchedElements: next });
    };

    const insertMention = (element: StudioElement) => {
        const trigger = `@${element.trigger_word.replace(/^@/, '')}`;
        const current = promptDraft ?? shot.prompt ?? '';
        if (current.includes(trigger)) return;
        const next = `${current.trim()} ${trigger}`.trim();
        setPromptDraft(next);
        updateShot(shot.id, { prompt: next });
    };

    const previewFrame = async () => {
        setPreview({ url: null, status: 'queued' });
        try {
            const response = await fetch(apiUrl('/generate/image'), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    project_id: project.id,
                    prompt: promptDraft ?? shot.prompt,
                    negative_prompt: shot.negativePrompt,
                    width: shot.width,
                    height: shot.height,
                    seed: shot.seed,
                    reference_images: matched
                        .map((element) => element?.image_path)
                        .filter((path): path is string => !!path),
                }),
            });
            const data = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
            const jobId = data.job_id as string;
            for (let attempt = 0; attempt < 90; attempt += 1) {
                await new Promise((resolve) => setTimeout(resolve, 2000));
                const status = await fetch(apiUrl(`/status/${jobId}`)).then((res) => res.json()).catch(() => ({}));
                if (status.status === 'completed' && status.video_url) {
                    const path = status.video_url as string;
                    updateShot(shot.id, {
                        timeline: [{
                            id: `preview_${jobId}`,
                            type: 'image',
                            path,
                            frameIndex: 0,
                            strength: 1,
                        }, ...shot.timeline],
                    });
                    setPreview({ url: getAssetUrl(path) || null, status: 'completed' });
                    useStudioDocumentStore.getState().addToast('Preview frame attached at frame 0', 'success');
                    return;
                }
                if (status.status === 'failed' || status.status === 'cancelled') {
                    throw new Error(status.error || 'Preview generation failed');
                }
                setPreview({ url: null, status: status.status_message || 'generating' });
            }
            throw new Error('Preview timed out');
        } catch (exc) {
            setPreview(null);
            useStudioDocumentStore.getState().addToast(
                exc instanceof Error ? exc.message : 'Preview failed',
                'error',
            );
        }
    };

    const queueReadyShots = async () => {
        const shotIds = videoShots
            .filter((item) => item.status !== 'completed' && (item.prompt || item.action))
            .map((item) => item.id);
        if (shotIds.length === 0) {
            useStudioDocumentStore.getState().addToast('No pending shots to queue', 'info');
            return;
        }
        setBatchBusy(true);
        try {
            const response = await fetch(apiUrl(`/projects/${project.id}/storyboard/batch-generate`), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ shot_ids: shotIds }),
            });
            const data = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
            const queued = (data.results || []).filter((item: { status: string }) => item.status === 'queued').length;
            useStudioDocumentStore.getState().addToast(`Queued ${queued} shot(s)`, 'success');
            await refreshJobs();
        } catch (exc) {
            useStudioDocumentStore.getState().addToast(
                exc instanceof Error ? exc.message : 'Batch queue failed',
                'error',
            );
        } finally {
            setBatchBusy(false);
        }
    };

    return (
        <div className="st-shot-workspace">
            <aside className="st-shot-rail">
                <div className="st-spine-head">
                    <div className="st-spine-title">Shot queue</div>
                    <div className="st-value" style={{ color: 'var(--st-muted)', fontSize: 10 }}>
                        {videoShots.filter((item) => item.status === 'completed').length}/{videoShots.length} with takes
                    </div>
                </div>
                <div className="st-spine-list">
                    {videoShots.map((item, index) => (
                        <button
                            key={item.id}
                            type="button"
                            className="st-spine-shot"
                            data-selected={item.id === shot.id}
                            onClick={() => useStudioDocumentStore.getState().selectShot(item.id)}
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
                        <button
                            type="button"
                            className="st-button"
                            disabled={preview?.status === 'queued' || preview?.status === 'generating'}
                            onClick={() => void previewFrame()}
                            title="Generate a still at frame 0 before spending video time"
                        >
                            {preview?.status ? <Loader2 size={12} className="animate-spin" /> : <ImageIcon size={12} />}
                            Preview frame
                        </button>
                        <button type="button" className="st-button" disabled={batchBusy} onClick={() => void queueReadyShots()}>
                            {batchBusy ? <Loader2 size={12} className="animate-spin" /> : <ListChecks size={12} />}
                            Queue pending
                        </button>
                        <button type="button" className="st-button" data-variant="ghost" onClick={() => setLensOpen(true)}>
                            <Film size={12} /> Lens
                        </button>
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
                    {takes.filter((take) => !take.selected).map((take) => (
                        <article className="st-take" key={take.job_id}>
                            <div className="st-take-media">
                                {take.thumbnail_url || take.url ? (
                                    <img src={getAssetUrl(take.thumbnail_url || take.url)} alt="" loading="lazy" />
                                ) : (
                                    <div className="st-stage-empty"><Film size={16} /></div>
                                )}
                            </div>
                            <div className="st-take-body">
                                <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8 }}>
                                    <strong style={{ color: 'var(--st-text)' }}>{take.model || take.type}</strong>
                                    <span className="st-mono">seed {take.seed ?? '—'}</span>
                                </div>
                                <div className="st-mono" style={{ marginTop: 4 }}>{take.job_id.slice(0, 12)}</div>
                                <button
                                    type="button"
                                    className="st-button"
                                    data-size="sm"
                                    style={{ marginTop: 6 }}
                                    onClick={async () => {
                                        if (shot && await selectTake(shot, take)) await refreshTakes();
                                    }}
                                >
                                    Select take
                                </button>
                                <button
                                    type="button"
                                    className="st-button"
                                    data-size="sm"
                                    data-variant={take.favorite ? 'primary' : 'ghost'}
                                    style={{ marginTop: 6 }}
                                    onClick={async () => {
                                        if (await favoriteTake(shot, take.favorite ? null : take)) await refreshTakes();
                                    }}
                                >
                                    <Star size={11} /> {take.favorite ? 'Favorited' : 'Favorite'}
                                </button>
                            </div>
                        </article>
                    ))}
                    {!takesLoading && takes.length <= 1 && (
                        <article className="st-take" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: 180, borderStyle: 'dashed' }}>
                            <div className="st-empty" style={{ minHeight: 0 }}>
                                <Film size={20} />
                                <strong>No older takes</strong>
                                <span>Generate again; the previous output stays available here.</span>
                            </div>
                        </article>
                    )}
                </div>

                {preview && (
                    <div className="st-panel" style={{ marginTop: 14 }}>
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8 }}>
                            <h3 className="st-panel-title" style={{ margin: 0 }}>Preview frame · {preview.status}</h3>
                            <button type="button" className="st-icon-button" aria-label="Close preview" onClick={() => setPreview(null)}>
                                <X size={12} />
                            </button>
                        </div>
                        {preview.url ? (
                            <img
                                src={preview.url}
                                alt="Preview frame"
                                style={{ width: '100%', maxHeight: 260, objectFit: 'contain', marginTop: 8, background: '#000', borderRadius: 6 }}
                            />
                        ) : (
                            <div className="st-empty" style={{ minHeight: 90 }}>
                                <Loader2 size={16} className="animate-spin" />
                                <span>{preview.status}</span>
                            </div>
                        )}
                        <p className="st-value" style={{ color: 'var(--st-faint)', fontSize: 9, marginTop: 6 }}>
                            Attached as conditioning at frame 0. Video generation still requires approval.
                        </p>
                    </div>
                )}

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

                <details className="st-panel" style={{ marginTop: 14 }}>
                    <summary className="st-panel-title" style={{ cursor: 'pointer', margin: 0 }}>
                        Batch matrix · {videoShots.length} shots
                    </summary>
                    <table className="st-table" style={{ marginTop: 8 }}>
                        <thead>
                            <tr><th>#</th><th>Shot</th><th>Status</th><th>Progress</th><th>Actions</th></tr>
                        </thead>
                        <tbody>
                            {videoShots.map((item, index) => (
                                <tr
                                    key={item.id}
                                    data-selected={item.id === shot.id}
                                    onClick={() => useStudioDocumentStore.getState().selectShot(item.id)}
                                >
                                    <td className="st-mono">{index + 1}</td>
                                    <td>{item.action || item.prompt || 'Untitled'}</td>
                                    <td>{item.status || 'pending'}</td>
                                    <td className="st-mono">{item.progress != null ? `${item.progress}%` : '—'}</td>
                                    <td onClick={(event) => event.stopPropagation()}>
                                        <div className="st-button-row">
                                            <button
                                                type="button"
                                                className="st-button"
                                                data-size="sm"
                                                disabled={item.isGenerating}
                                                onClick={() => void generate(item)}
                                            >
                                                {item.isGenerating ? <Loader2 size={11} className="animate-spin" /> : <Sparkles size={11} />}
                                                {item.isGenerating ? 'Running' : 'Generate'}
                                            </button>
                                            {item.isGenerating && (
                                                <button
                                                    type="button"
                                                    className="st-button"
                                                    data-size="sm"
                                                    data-variant="danger"
                                                    onClick={() => void cancel(item)}
                                                >
                                                    Cancel
                                                </button>
                                            )}
                                        </div>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </details>
            </section>

            <aside className="st-shot-params">
                <div className="st-inspector-section">
                    <span className="st-label">Ingredients · {matched.length}</span>
                    <div className="st-chip-row">
                        {elements.length === 0 && (
                            <span className="st-value" style={{ color: 'var(--st-muted)' }}>
                                No master references in this project.
                            </span>
                        )}
                        {elements.map((element) => {
                            const active = matched.some((item) => item?.id === element.id);
                            return (
                                <button
                                    key={element.id}
                                    type="button"
                                    className="st-chip"
                                    data-active={active}
                                    style={{ cursor: 'pointer' }}
                                    title={`${element.description}${active ? ' · click to remove' : ' · click to add'}`}
                                    onClick={() => toggleIngredient(element)}
                                >
                                    {active ? '✓ ' : '+ '}@{element.trigger_word.replace(/^@/, '')} · {element.name}
                                </button>
                            );
                        })}
                    </div>
                    {matched.length > 0 && (
                        <div className="st-button-row" style={{ marginTop: 6 }}>
                            {matched.map((element) => (
                                <button
                                    key={`mention-${element!.id}`}
                                    type="button"
                                    className="st-button"
                                    data-size="sm"
                                    data-variant="ghost"
                                    onClick={() => insertMention(element!)}
                                    title="Insert the trigger word into the prompt"
                                >
                                    Insert @{element!.trigger_word.replace(/^@/, '')}
                                </button>
                            ))}
                        </div>
                    )}
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
                    <details className="st-advanced">
                        <summary>Advanced parameters</summary>
                        <div className="st-inspector-section" style={{ marginTop: 10, borderBottom: 0 }}>
                            <label className="st-label" htmlFor="gen-negative">Negative prompt</label>
                            <textarea
                                id="gen-negative"
                                className="st-textarea"
                                style={{ minHeight: 54 }}
                                value={shot.negativePrompt}
                                onChange={(event) => updateShot(shot.id, { negativePrompt: event.target.value })}
                            />
                            <div className="st-row" style={{ marginTop: 8 }}>
                                <label>
                                    <span className="st-label">Seed</span>
                                    <input className="st-input" type="number" value={shot.seed} onChange={(event) => updateShot(shot.id, { seed: Number(event.target.value) })} />
                                </label>
                                <label>
                                    <span className="st-label">Frames</span>
                                    <input className="st-input" type="number" value={shot.numFrames} onChange={(event) => updateShot(shot.id, { numFrames: Number(event.target.value) })} />
                                </label>
                            </div>
                            <div className="st-row" style={{ marginTop: 8 }}>
                                <label>
                                    <span className="st-label">Width</span>
                                    <input className="st-input" type="number" value={shot.width} onChange={(event) => updateShot(shot.id, { width: Number(event.target.value) })} />
                                </label>
                                <label>
                                    <span className="st-label">Height</span>
                                    <input className="st-input" type="number" value={shot.height} onChange={(event) => updateShot(shot.id, { height: Number(event.target.value) })} />
                                </label>
                            </div>
                            <div className="st-row" style={{ marginTop: 8 }}>
                                <label>
                                    <span className="st-label">CFG</span>
                                    <input className="st-input" type="number" step={0.1} value={shot.cfgScale} onChange={(event) => updateShot(shot.id, { cfgScale: Number(event.target.value) })} />
                                </label>
                                <label>
                                    <span className="st-label">Pipeline</span>
                                    <select className="st-select" value={shot.pipelineOverride || 'auto'} onChange={(event) => updateShot(shot.id, { pipelineOverride: event.target.value as typeof shot.pipelineOverride })}>
                                        <option value="auto">auto</option>
                                        <option value="ti2vid">ti2vid</option>
                                        <option value="ic_lora">ic_lora</option>
                                        <option value="keyframe">keyframe</option>
                                    </select>
                                </label>
                            </div>
                            <div className="st-button-row" style={{ marginTop: 8 }}>
                                <label className="st-chip">
                                    <input type="checkbox" checked={shot.enhancePrompt} onChange={(event) => updateShot(shot.id, { enhancePrompt: event.target.checked })} />
                                    enhance
                                </label>
                                <label className="st-chip">
                                    <input type="checkbox" checked={shot.upscale} onChange={(event) => updateShot(shot.id, { upscale: event.target.checked })} />
                                    upscale
                                </label>
                            </div>
                        </div>
                    </details>
                </div>
                <div className="st-inspector-section">
                    <span className="st-label">Active queue · {activeJobs.length}</span>
                    {activeJobs.length === 0 ? (
                        <span className="st-value" style={{ color: 'var(--st-muted)', fontSize: 10 }}>No active jobs.</span>
                    ) : (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
                            {activeJobs.map((job) => (
                                <div key={job.job_id} className="st-link-row">
                                    <span className="st-mono">{job.job_id.slice(0, 10)}</span>
                                    <span style={{ flex: 1 }}>
                                        {job.position ? `#${job.position} · ` : ''}
                                        {job.status_message || job.status} · {job.progress}%
                                        {job.estimate_seconds ? ` · ~${job.estimate_seconds}s` : ''}
                                    </span>
                                    <button
                                        type="button"
                                        className="st-button"
                                        data-size="sm"
                                        data-variant="danger"
                                        onClick={async () => {
                                            await fetch(apiUrl(`/jobs/${job.job_id}/cancel`), { method: 'POST' });
                                            await refreshJobs();
                                        }}
                                    >
                                        Cancel
                                    </button>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
                <div className="st-inspector-section">
                    <span className="st-label">Reference media</span>
                    <div className="st-button-row">
                        <button type="button" className="st-button" data-size="sm" onClick={() => window.location.assign('/')}>
                            <Play size={12} /> Legacy edit tools
                        </button>
                    </div>
                    <p className="st-value" style={{ color: 'var(--st-faint)', fontSize: 9, marginTop: 6 }}>
                        Preview-first generates a still you can inspect before video. Batch queue uses the durable GPU queue.
                    </p>
                </div>
            </aside>

            {lensOpen && (
                <PipelineLens
                    shot={shot}
                    elements={elements}
                    updateShot={updateShot}
                    onClose={() => setLensOpen(false)}
                />
            )}
        </div>
    );
}
