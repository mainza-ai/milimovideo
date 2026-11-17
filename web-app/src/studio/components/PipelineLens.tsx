import { useRef, useState } from 'react';
import { CheckCircle2, Copy, Loader2, TriangleAlert, Upload, X } from 'lucide-react';
import type { Shot } from '../state/documentStore';
import { saveTextFile } from '../../desktop';
import type { StudioElement } from '../data/useStudioData';
import { parsePipelineRecipe } from '../lens/recipe';

interface Props {
    shot: Shot;
    elements: StudioElement[];
    updateShot: (id: string, updates: Partial<Shot>) => void;
    onClose: () => void;
}

interface Node {
    id: string;
    label: string;
    detail: string;
    ok: boolean;
}

export function PipelineLens({ shot, elements, updateShot, onClose }: Props) {
    const [estimate, setEstimate] = useState<string | null>(null);
    const [loading, setLoading] = useState(false);
    const [importError, setImportError] = useState<string | null>(null);
    const importRef = useRef<HTMLInputElement>(null);
    const [draft, setDraft] = useState({
        prompt: shot.prompt || '',
        negative: shot.negativePrompt || '',
        frames: shot.numFrames,
        width: shot.width,
        height: shot.height,
        seed: shot.seed,
        model: shot.pipelineOverride || 'auto',
        enhance: !!shot.enhancePrompt,
        upscale: !!shot.upscale,
    });
    const matched = (shot.matchedElements || [])
        .map((match) => elements.find((element) => element.id === match.element_id))
        .filter(Boolean);
    const framesValid = (draft.frames - 1) % 8 === 0;
    const nodes: Node[] = [
        { id: 'text', label: 'Prompt', detail: `${draft.prompt.length} chars`, ok: !!draft.prompt.trim() },
        { id: 'refs', label: 'Ingredients', detail: `${matched.length} references`, ok: true },
        { id: 'model', label: 'Model', detail: draft.model, ok: true },
        { id: 'frames', label: 'Frames', detail: `${draft.frames} frames`, ok: framesValid },
        { id: 'resolution', label: 'Resolution', detail: `${draft.width}×${draft.height}`, ok: draft.width % 2 === 0 && draft.height % 2 === 0 },
        { id: 'upscale', label: 'Upscale', detail: draft.upscale ? 'on' : 'off', ok: true },
        { id: 'output', label: 'Take', detail: shot.videoUrl ? 'existing' : 'new', ok: true },
    ];
    const warnings = nodes.filter((node) => !node.ok);

    const recipe = {
        shot: shot.id,
        prompt: draft.prompt,
        negative_prompt: draft.negative,
        model: draft.model,
        seed: draft.seed,
        frames: draft.frames,
        resolution: `${draft.width}x${draft.height}`,
        ingredients: matched.map((element) => element?.trigger_word),
        enhance: draft.enhance,
        upscale: draft.upscale,
    };

    const applyDraft = () => {
        updateShot(shot.id, {
            prompt: draft.prompt,
            negativePrompt: draft.negative,
            numFrames: draft.frames,
            width: draft.width,
            height: draft.height,
            seed: draft.seed,
            pipelineOverride: draft.model as Shot['pipelineOverride'],
            enhancePrompt: draft.enhance,
            upscale: draft.upscale,
        });
        onClose();
    };

    const estimatePlan = () => {
        setLoading(true);
        const workload = Math.max(0.08, (draft.frames / 121) * ((draft.width * draft.height) / (768 * 512)));
        const seconds = Math.round((8 + 22 * workload) * 10) / 10;
        const vram = Math.min(30000, Math.round(11000 + 6000 * workload));
        setEstimate(`${draft.model} · ~${seconds}s GPU · ~${(vram / 1024).toFixed(1)} GB VRAM`);
        setLoading(false);
    };

    return (
        <div className="st-lens-backdrop" role="dialog" aria-modal="true" aria-label="Pipeline lens">
            <div className="st-lens">
                <div className="st-lens-head">
                    <div>
                        <strong>Pipeline lens · {shot.id.slice(0, 10)}</strong>
                        <div style={{ color: 'var(--st-faint)', fontSize: 9 }}>
                            Recipe graph and validation for this shot. No cloud calls.
                        </div>
                    </div>
                    <button type="button" className="st-icon-button" aria-label="Close pipeline lens" onClick={onClose}>
                        <X size={14} />
                    </button>
                </div>

                <div className="st-lens-graph">
                    {nodes.map((node) => (
                        <div key={node.id} className="st-lens-node" data-ok={node.ok}>
                            <span>{node.label}</span>
                            <strong>{node.detail}</strong>
                            {node.ok ? <CheckCircle2 size={11} /> : <TriangleAlert size={11} />}
                        </div>
                    ))}
                </div>

                <div className="st-lens-body">
                    <div>
                        <h4>Recipe</h4>
                        <textarea
                            className="st-textarea"
                            style={{ minHeight: 58 }}
                            aria-label="Lens prompt"
                            value={draft.prompt}
                            onChange={(event) => setDraft({ ...draft, prompt: event.target.value })}
                        />
                        <label className="st-label" style={{ marginTop: 7 }}>Negative prompt</label>
                        <textarea
                            className="st-textarea"
                            style={{ minHeight: 40 }}
                            aria-label="Lens negative prompt"
                            value={draft.negative}
                            onChange={(event) => setDraft({ ...draft, negative: event.target.value })}
                        />
                        <div className="st-row" style={{ marginTop: 7 }}>
                            <label>
                                <span className="st-label">Frames · 8k+1</span>
                                <input
                                    className="st-input"
                                    type="number"
                                    value={draft.frames}
                                    onChange={(event) => setDraft({ ...draft, frames: Number(event.target.value) })}
                                />
                            </label>
                            <label>
                                <span className="st-label">Seed</span>
                                <input
                                    className="st-input"
                                    type="number"
                                    value={draft.seed}
                                    onChange={(event) => setDraft({ ...draft, seed: Number(event.target.value) })}
                                />
                            </label>
                        </div>
                        <div className="st-row" style={{ marginTop: 7 }}>
                            <label>
                                <span className="st-label">Width</span>
                                <input
                                    className="st-input"
                                    type="number"
                                    value={draft.width}
                                    onChange={(event) => setDraft({ ...draft, width: Number(event.target.value) })}
                                />
                            </label>
                            <label>
                                <span className="st-label">Height</span>
                                <input
                                    className="st-input"
                                    type="number"
                                    value={draft.height}
                                    onChange={(event) => setDraft({ ...draft, height: Number(event.target.value) })}
                                />
                            </label>
                        </div>
                        <div className="st-row" style={{ marginTop: 7 }}>
                            <label>
                                <span className="st-label">Model</span>
                                <select
                                    className="st-select"
                                    value={draft.model}
                                    onChange={(event) => setDraft({ ...draft, model: event.target.value as typeof draft.model })}
                                >
                                    <option value="auto">auto</option>
                                    <option value="ti2vid">ti2vid</option>
                                    <option value="ic_lora">ic_lora</option>
                                    <option value="keyframe">keyframe</option>
                                </select>
                            </label>
                            <div className="st-button-row" style={{ alignSelf: 'end' }}>
                                <label className="st-chip">
                                    <input
                                        type="checkbox"
                                        checked={draft.enhance}
                                        onChange={(event) => setDraft({ ...draft, enhance: event.target.checked })}
                                    />
                                    enhance
                                </label>
                                <label className="st-chip">
                                    <input
                                        type="checkbox"
                                        checked={draft.upscale}
                                        onChange={(event) => setDraft({ ...draft, upscale: event.target.checked })}
                                    />
                                    upscale
                                </label>
                            </div>
                        </div>

                        <h4 style={{ marginTop: 12 }}>Validation</h4>
                        {warnings.length === 0 ? (
                            <div className="st-lens-ok"><CheckCircle2 size={12} /> Recipe is valid.</div>
                        ) : (
                            <ul className="st-lens-warnings">
                                {warnings.map((node) => (
                                    <li key={node.id}>
                                        <TriangleAlert size={11} /> {node.label}: {node.detail}
                                        {node.id === 'frames' && ' (LTX needs 8k+1 frames)'}
                                    </li>
                                ))}
                            </ul>
                        )}
                        <h4 style={{ marginTop: 12 }}>Reproducible recipe</h4>
                        <pre className="st-lens-code">{JSON.stringify(recipe, null, 2)}</pre>
                        <div className="st-button-row">
                            <button
                                type="button"
                                className="st-button"
                                data-size="sm"
                                data-variant="primary"
                                disabled={warnings.length > 0}
                                onClick={applyDraft}
                            >
                                <CheckCircle2 size={11} /> Apply to shot
                            </button>
                            <button
                                type="button"
                                className="st-button"
                                data-size="sm"
                                onClick={() => void navigator.clipboard.writeText(JSON.stringify(recipe, null, 2))}
                            >
                                <Copy size={11} /> Copy JSON
                            </button>
                            <button
                                type="button"
                                className="st-button"
                                data-size="sm"
                                onClick={() => void saveTextFile(
                                    `recipe-${shot.id}.json`,
                                    JSON.stringify(recipe, null, 2),
                                )}
                            >
                                Save JSON
                            </button>
                            <button type="button" className="st-button" data-size="sm" onClick={() => void estimatePlan()}>
                                {loading ? <Loader2 size={11} className="animate-spin" /> : <CheckCircle2 size={11} />}
                                Estimate
                            </button>
                            <button type="button" className="st-button" data-size="sm" onClick={() => importRef.current?.click()}>
                                <Upload size={11} /> Import JSON
                            </button>
                            <input
                                ref={importRef}
                                type="file"
                                accept="application/json,.json"
                                style={{ display: 'none' }}
                                aria-label="Import pipeline recipe"
                                onChange={async (event) => {
                                    const file = event.target.files?.[0];
                                    event.target.value = '';
                                    if (!file) return;
                                    setImportError(null);
                                    try {
                                        const parsed = parsePipelineRecipe(await file.text());
                                        setDraft({
                                            prompt: parsed.prompt,
                                            negative: parsed.negativePrompt,
                                            frames: parsed.frames,
                                            width: parsed.width,
                                            height: parsed.height,
                                            seed: parsed.seed,
                                            model: parsed.model,
                                            enhance: parsed.enhance,
                                            upscale: parsed.upscale,
                                        });
                                    } catch (error) {
                                        setImportError(error instanceof Error ? error.message : 'Could not read recipe');
                                    }
                                }}
                            />
                        </div>
                        {importError && (
                            <div className="st-lens-warnings" role="alert">
                                <TriangleAlert size={11} /> {importError}
                            </div>
                        )}
                        {estimate && (
                            <div className="st-lens-estimate">{estimate}</div>
                        )}
                    </div>
                    <div>
                        <h4>Ingredients</h4>
                        {matched.length === 0 ? (
                            <div style={{ color: 'var(--st-muted)', fontSize: 10 }}>No references attached.</div>
                        ) : (
                            <div className="st-chip-row">
                                {matched.map((element) => (
                                    <span className="st-chip" key={element!.id}>
                                        @{element!.trigger_word.replace(/^@/, '')} · {element!.name}
                                    </span>
                                ))}
                            </div>
                        )}
                        <h4 style={{ marginTop: 12 }}>Notes</h4>
                        <p style={{ color: 'var(--st-muted)', fontSize: 10, lineHeight: 1.55 }}>
                            Apply writes the recipe back to the shot through the undoable document store.
                            Reproducing it needs the same model, seed, frames, resolution, and ingredients.
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
}
