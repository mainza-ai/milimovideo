import { useMemo, useState } from 'react';
import { ArrowUp, Check, CornerDownLeft, Sparkles, X } from 'lucide-react';
import { useStudioEditor } from '../state/editorStore';
import {
    useStudioElements,
    useStudioGenerate,
    useStudioProject,
    useStudioSelection,
} from '../data/useStudioData';

const MODES = [
    { id: 'ask', label: 'Ask', placeholder: 'Ask about the cut, a shot, or a reference…' },
    { id: 'direct', label: 'Direct', placeholder: 'Describe an edit. A plan will be previewed before anything changes.' },
    { id: 'generate', label: 'Generate', placeholder: 'Describe the next take for the selected shot…' },
    { id: 'find', label: 'Find', placeholder: 'Find a shot, asset, or reference…' },
] as const;

export function CommandDock() {
    const project = useStudioProject();
    const elements = useStudioElements(project.id);
    const { selectedShotId, addToast } = useStudioSelection();
    const generate = useStudioGenerate();
    const { commandOpen, setCommandOpen, commandMode, setCommandMode } = useStudioEditor();
    const [value, setValue] = useState('');
    const [plan, setPlan] = useState<string[] | null>(null);

    const shot = project.shots.find((item) => item.id === selectedShotId) || null;
    const mode = MODES.find((item) => item.id === commandMode) || MODES[1];
    const matched = useMemo(() => shot?.matchedElements || [], [shot]);

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

    const submit = (event?: React.FormEvent) => {
        event?.preventDefault();
        const text = value.trim();
        if (!text) return;
        const lines = commandMode === 'generate'
            ? [
                `Target: ${shot ? `Shot ${shot.index != null ? shot.index + 1 : ''} · ${shot.action || shot.prompt}` : 'no shot selected'}`,
                `Prompt: ${text}`,
                `Model: project default · seed ${shot?.seed ?? project.seed}`,
                `Cost: local GPU time (estimate shown by the queue)`,
            ]
            : commandMode === 'find'
                ? [`Local search: ${text}`, 'Search across cards, assets, and references in this project.']
                : [
                    `Context: ${shot ? shot.action || shot.prompt : 'project'} · ${matched.length} matched reference(s)`,
                    `Request: ${text}`,
                    'Effect: preview only — approve to apply an undoable edit',
                ];
        setPlan(lines);
    };

    const approve = async () => {
        if (!shot) {
            addToast('Select a shot first', 'info');
            return;
        }
        if (commandMode === 'generate' || commandMode === 'direct') {
            await generate(shot, commandMode === 'generate');
        }
        setPlan(null);
        setValue('');
    };

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
                <span className="st-context-chip" style={{ marginLeft: 'auto' }}>context is visible to the agent</span>
            </div>

            {plan && (
                <div className="st-plan-preview">
                    <h4>Plan preview</h4>
                    <ul>{plan.map((line) => <li key={line}>{line}</li>)}</ul>
                    <div className="st-plan-actions">
                        <button type="button" className="st-button" data-variant="primary" onClick={() => void approve()}>
                            <Check size={12} /> Approve
                        </button>
                        <button type="button" className="st-button" data-variant="ghost" onClick={() => setPlan(null)}>
                            <X size={12} /> Cancel
                        </button>
                    </div>
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
                            onClick={() => setCommandMode(item.id)}
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
                            submit();
                        }
                    }}
                />
                <button type="submit" className="st-button st-command-send" data-variant="primary">
                    <ArrowUp size={14} /> Plan
                </button>
            </form>
            <div className="st-command-hint">
                <CornerDownLeft size={10} /> Enter to plan · Shift+Enter for a new line · Ask-first is the default
            </div>
        </section>
    );
}
