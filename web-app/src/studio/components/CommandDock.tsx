import { useEffect, useMemo, useState } from 'react';
import { ArrowUp, Check, CornerDownLeft, Loader2, Sparkles, TriangleAlert, X } from 'lucide-react';
import type { Shot } from '../../stores/timelineStore';
import { apiUrl } from '../../config';
import { useStudioEditor } from '../state/editorStore';
import {
    useStudioElements,
    useStudioGenerate,
    useStudioProject,
    useStudioSelection,
} from '../data/useStudioData';

const MODES = [
    { id: 'ask', label: 'Ask', placeholder: 'Ask about the cut, a shot, or a reference…' },
    { id: 'direct', label: 'Direct', placeholder: 'prompt: … · seed 42 · frames 121 · type close-up · add rain' },
    { id: 'generate', label: 'Generate', placeholder: 'Describe the next take for the selected shot…' },
    { id: 'find', label: 'Find', placeholder: 'Find a shot, asset, or reference…' },
] as const;

interface PlanStep {
    kind: string;
    label: string;
    target?: string | null;
    before?: unknown;
    after?: unknown;
    detail?: string | null;
}

interface PlanEstimate {
    model?: string | null;
    gpu_seconds?: number | null;
    vram_mb?: number | null;
    note?: string | null;
}

interface Plan {
    mode: string;
    summary: string;
    steps: PlanStep[];
    estimate?: PlanEstimate | null;
    warnings: string[];
    shot_id?: string | null;
}

function renderValue(value: unknown): string {
    if (value === null || value === undefined || value === '') return '—';
    const text = String(value);
    return text.length > 120 ? `${text.slice(0, 117)}…` : text;
}

export function CommandDock() {
    const project = useStudioProject();
    const elements = useStudioElements(project.id);
    const { selectedShotId, addToast, updateShot } = useStudioSelection();
    const generate = useStudioGenerate();
    const {
        commandOpen, setCommandOpen, commandMode, setCommandMode,
        setBoardQuery, setSurface, commandDraft, setCommandDraft,
    } = useStudioEditor();
    const [value, setValue] = useState('');
    const [plan, setPlan] = useState<Plan | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const shot = project.shots.find((item) => item.id === selectedShotId) || null;
    const mode = MODES.find((item) => item.id === commandMode) || MODES[1];
    const matched = useMemo(() => shot?.matchedElements || [], [shot]);

    useEffect(() => {
        if (!commandDraft) return;
        setValue(commandDraft);
        setCommandOpen(true);
        setPlan(null);
        setCommandDraft('');
    }, [commandDraft, setCommandDraft, setCommandOpen]);

    if (!commandOpen) {
        return (
            <div className="st-command-dock">
                <div className="st-command-main">
                    <button type="button" className="st-button" data-variant="primary" onClick={() => setCommandOpen(true)}>
                        <Sparkles size={13} /> Open command dock
                    </button>
                    <span className="st-command-hint">⌘K</span>
                </div>
            </div>
        );
    }

    const submit = async (event?: React.FormEvent) => {
        event?.preventDefault();
        const text = value.trim();
        if (!text) return;
        setLoading(true);
        setError(null);
        try {
            const response = await fetch(apiUrl(`/projects/${project.id}/plan`), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text, mode: commandMode, shot_id: selectedShotId }),
            });
            const data = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
            setPlan(data as Plan);
        } catch (exc) {
            const message = exc instanceof Error ? exc.message : 'Planning failed';
            setError(message);
            addToast(message, 'error');
        } finally {
            setLoading(false);
        }
    };

    const approve = async () => {
        if (!plan) return;

        if (plan.mode === 'ask') {
            setPlan(null);
            setValue('');
            return;
        }

        if (plan.mode === 'find') {
            const query = String(plan.steps[0]?.after || value);
            setBoardQuery(query);
            setSurface('story');
            addToast(`Filtering the board for “${query}”`, 'info');
            setPlan(null);
            setValue('');
            return;
        }

        if (!shot) {
            addToast('Select a shot first', 'info');
            return;
        }

        const patch: Partial<Shot> = {};
        let generateStep: PlanStep | null = null;
        for (const step of plan.steps) {
            switch (step.kind) {
                case 'set_prompt':
                    patch.prompt = String(step.after ?? '');
                    break;
                case 'set_negative_prompt':
                    patch.negativePrompt = String(step.after ?? '');
                    break;
                case 'set_seed':
                    patch.seed = Number(step.after);
                    break;
                case 'set_frames':
                    patch.numFrames = Number(step.after);
                    break;
                case 'set_shot_type':
                    patch.shotType = String(step.after ?? '') as Shot['shotType'];
                    break;
                case 'generate':
                    generateStep = step;
                    break;
                default:
                    break;
            }
        }

        const hasPatch = Object.keys(patch).length > 0;
        if (hasPatch) updateShot(shot.id, patch);

        if (generateStep) {
            const nextShot: Shot = {
                ...shot,
                ...patch,
                prompt: String(generateStep.after ?? patch.prompt ?? shot.prompt),
            };
            await generate(nextShot, true);
        } else if (hasPatch) {
            addToast(`${Object.keys(patch).length} edit(s) applied — undo is available`, 'success');
        }

        setPlan(null);
        setValue('');
    };

    const approveLabel = plan?.mode === 'ask'
        ? 'Close'
        : plan?.mode === 'find'
            ? 'Apply filter'
            : plan?.steps.some((step) => step.kind === 'generate') ? 'Approve & generate' : 'Approve & apply';
    const canApprove = !!plan && (plan.mode === 'ask' || plan.mode === 'find' || plan.steps.length > 0);

    return (
        <section className="st-command-dock" aria-label="Command dock">
            <div className="st-context-row">
                <span className="st-context-chip" data-kind="selection">
                    {shot ? `Shot ${shot.index != null ? shot.index + 1 : ''}` : 'Project'}
                </span>
                <span className="st-context-chip">{project.name}</span>
                {shot?.sceneId && <span className="st-context-chip">scene {shot.sceneId.slice(0, 6)}</span>}
                {matched.slice(0, 3).map((match) => (
                    <span className="st-context-chip" key={match.element_id}>@{match.trigger_word.replace(/^@/, '')}</span>
                ))}
                {elements.length > 0 && <span className="st-context-chip">{elements.length} master refs</span>}
                <span className="st-context-chip" style={{ marginLeft: 'auto' }}>plan is local · nothing runs before approval</span>
            </div>

            {plan && (
                <div className="st-plan-preview">
                    <h4>Plan preview · {plan.mode}</h4>
                    <p className="st-plan-summary">{plan.summary}</p>
                    {plan.steps.length > 0 && (
                        <ul className="st-plan-steps">
                            {plan.steps.map((step, index) => (
                                <li key={`${step.kind}-${index}`}>
                                    <strong>{step.label}</strong>
                                    {step.kind !== 'answer' && step.kind !== 'search' && (
                                        <span className="st-plan-diff">
                                            <span>{renderValue(step.before)}</span>
                                            <span aria-hidden="true">→</span>
                                            <span>{renderValue(step.after)}</span>
                                        </span>
                                    )}
                                    {step.kind === 'answer' && <span className="st-plan-diff">{step.detail}</span>}
                                    {step.detail && step.kind !== 'answer' && (
                                        <span className="st-plan-detail">{step.detail}</span>
                                    )}
                                </li>
                            ))}
                        </ul>
                    )}
                    {plan.estimate && (
                        <div className="st-plan-estimate">
                            <span className="st-chip">model {plan.estimate.model || 'auto'}</span>
                            {plan.estimate.gpu_seconds != null && (
                                <span className="st-chip">~{plan.estimate.gpu_seconds}s GPU</span>
                            )}
                            {plan.estimate.vram_mb != null && (
                                <span className="st-chip">~{(plan.estimate.vram_mb / 1024).toFixed(1)} GB VRAM</span>
                            )}
                            {plan.estimate.note && <span className="st-plan-detail">{plan.estimate.note}</span>}
                        </div>
                    )}
                    {plan.warnings.length > 0 && (
                        <ul className="st-plan-warnings">
                            {plan.warnings.map((warning) => (
                                <li key={warning}><TriangleAlert size={10} /> {warning}</li>
                            ))}
                        </ul>
                    )}
                    <div className="st-plan-actions">
                        <button
                            type="button"
                            className="st-button"
                            data-variant="primary"
                            disabled={!canApprove}
                            onClick={() => void approve()}
                        >
                            <Check size={12} /> {approveLabel}
                        </button>
                        <button type="button" className="st-button" data-variant="ghost" onClick={() => setPlan(null)}>
                            <X size={12} /> Cancel
                        </button>
                    </div>
                </div>
            )}

            {error && (
                <div className="st-plan-error">
                    <TriangleAlert size={12} /> {error}
                </div>
            )}

            <form className="st-command-main" onSubmit={submit}>
                <div className="st-command-modes" role="group" aria-label="Command mode">
                    {MODES.map((item) => (
                        <button
                            key={item.id}
                            type="button"
                            className="st-command-mode"
                            data-active={commandMode === item.id}
                            onClick={() => {
                                setCommandMode(item.id);
                                setPlan(null);
                            }}
                        >
                            {item.label}
                        </button>
                    ))}
                </div>
                <textarea
                    className="st-command-input"
                    value={value}
                    onChange={(event) => setValue(event.target.value)}
                    placeholder={mode.placeholder}
                    rows={1}
                    onKeyDown={(event) => {
                        if (event.key === 'Enter' && !event.shiftKey) {
                            event.preventDefault();
                            void submit();
                        }
                    }}
                />
                <button type="submit" className="st-button st-command-send" data-variant="primary" disabled={loading}>
                    {loading ? <Loader2 size={14} className="animate-spin" /> : <ArrowUp size={14} />} Plan
                </button>
            </form>
            <div className="st-command-hint">
                <CornerDownLeft size={10} /> Enter to plan · Shift+Enter for a new line · approve before spend
            </div>
        </section>
    );
}
