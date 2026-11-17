import { useState } from 'react';
import { CheckCircle2, Copy, Loader2, TriangleAlert, X } from 'lucide-react';
import type { Shot } from '../state/documentStore';
import { apiUrl } from '../../config';
import { saveTextFile } from '../../desktop';
import type { StudioElement } from '../data/useStudioData';

interface Props {
    projectId: string;
    shot: Shot;
    elements: StudioElement[];
    onClose: () => void;
}

interface Node {
    id: string;
    label: string;
    detail: string;
    ok: boolean;
}

export function PipelineLens({ projectId, shot, elements, onClose }: Props) {
    const [estimate, setEstimate] = useState<string | null>(null);
    const [loading, setLoading] = useState(false);
    const matched = (shot.matchedElements || [])
        .map((match) => elements.find((element) => element.id === match.element_id))
        .filter(Boolean);
    const framesValid = (shot.numFrames - 1) % 8 === 0;
    const nodes: Node[] = [
        { id: 'text', label: 'Prompt', detail: `${(shot.prompt || '').length} chars`, ok: !!shot.prompt?.trim() },
        { id: 'refs', label: 'Ingredients', detail: `${matched.length} references`, ok: true },
        { id: 'model', label: 'Model', detail: shot.pipelineOverride || 'auto', ok: true },
        { id: 'frames', label: 'Frames', detail: `${shot.numFrames} frames`, ok: framesValid },
        { id: 'resolution', label: 'Resolution', detail: `${shot.width}×${shot.height}`, ok: shot.width % 2 === 0 && shot.height % 2 === 0 },
        { id: 'upscale', label: 'Upscale', detail: shot.upscale ? 'on' : 'off', ok: true },
        { id: 'output', label: 'Take', detail: shot.videoUrl ? 'existing' : 'new', ok: true },
    ];
    const warnings = nodes.filter((node) => !node.ok);

    const recipe = {
        shot: shot.id,
        prompt: shot.prompt,
        negative_prompt: shot.negativePrompt,
        model: shot.pipelineOverride || 'auto',
        seed: shot.seed,
        frames: shot.numFrames,
        resolution: `${shot.width}x${shot.height}`,
        ingredients: matched.map((element) => element?.trigger_word),
        enhance: shot.enhancePrompt,
        upscale: shot.upscale,
    };

    const estimatePlan = async () => {
        setLoading(true);
        try {
            const response = await fetch(apiUrl(`/projects/${projectId}/plan`), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mode: 'generate', text: shot.prompt || 'take', shot_id: shot.id }),
            });
            const data = await response.json().catch(() => ({}));
            const info = data.estimate;
            setEstimate(info
                ? `${info.model || 'auto'} · ~${info.gpu_seconds}s · ~${(info.vram_mb / 1024).toFixed(1)} GB`
                : 'No estimate available');
        } catch {
            setEstimate('Estimate failed');
        } finally {
            setLoading(false);
        }
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
                        <h4>Validation</h4>
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
                        </div>
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
                            The lens is read-only: edits still go through Inspector or the command dock.
                            Reproducing this recipe needs the same model, seed, frames, resolution, and ingredients.
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
}
