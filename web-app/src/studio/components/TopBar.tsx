import {
    CheckCircle2, CloudOff, HardDrive, MonitorPlay, Redo2, Save, Share2, TriangleAlert, Undo2,
} from 'lucide-react';
import { useStore } from 'zustand';
import { useStudioData } from '../data/useStudioData';
import { useStudioEditor, type StudioSurface } from '../state/editorStore';
import { useTimelineStore } from '../../stores/timelineStore';

const SURFACES: Array<{ id: StudioSurface; label: string }> = [
    { id: 'story', label: 'Story' },
    { id: 'generate', label: 'Generate' },
    { id: 'edit', label: 'Edit' },
    { id: 'review', label: 'Review' },
    { id: 'library', label: 'Library' },
    { id: 'deliver', label: 'Deliver' },
];

export function TopBar() {
    const { project, health, error } = useStudioData();
    const { activeSurface, setSurface, inspectorOpen, setInspectorOpen } = useStudioEditor();
    const { saveStatus, saveError, saveProject } = useTimelineStore();
    // zundo's temporal store is attached to the document store.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const temporalStore = (useTimelineStore as any).temporal;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const canUndo = useStore(temporalStore, (state: any) => state.pastStates.length > 0);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const canRedo = useStore(temporalStore, (state: any) => state.futureStates.length > 0);

    const saveLabel = saveStatus === 'saving' ? 'Saving…'
        : saveStatus === 'conflict' ? 'Conflict'
            : saveStatus === 'error' ? 'Save failed'
                : saveStatus === 'saved' ? 'Saved' : 'Unsaved';
    const saveTone = saveStatus === 'error' || saveStatus === 'conflict' ? 'bad'
        : saveStatus === 'saving' ? 'warn' : 'good';

    return (
        <header className="st-topbar">
            <div className="st-brand">
                <img src="/logo.png" alt="" />
                <span className="st-brand-name">Milimo</span>
                <span className="st-status-pill" data-tone={health ? 'good' : 'warn'}>
                    {health ? <MonitorPlay size={11} /> : <CloudOff size={11} />}
                    {health ? health.device.toUpperCase() : 'OFFLINE'}
                </span>
                <input
                    className="st-project-name"
                    value={project.name}
                    aria-label="Project name"
                    onChange={(event) => useTimelineStore.getState().setProject({ ...project, name: event.target.value })}
                />
            </div>

            <nav className="st-surface-tabs" aria-label="Studio surfaces">
                {SURFACES.map((surface) => (
                    <button
                        key={surface.id}
                        type="button"
                        className="st-surface-tab"
                        data-active={activeSurface === surface.id}
                        onClick={() => setSurface(surface.id)}
                    >
                        {surface.label}
                    </button>
                ))}
            </nav>

            <div className="st-topbar-right">
                <button
                    type="button"
                    className="st-icon-button"
                    disabled={!canUndo}
                    onClick={() => temporalStore.getState().undo()}
                    title="Undo (⌘Z)"
                    aria-label="Undo"
                >
                    <Undo2 size={15} />
                </button>
                <button
                    type="button"
                    className="st-icon-button"
                    disabled={!canRedo}
                    onClick={() => temporalStore.getState().redo()}
                    title="Redo (⇧⌘Z)"
                    aria-label="Redo"
                >
                    <Redo2 size={15} />
                </button>
                <button
                    type="button"
                    className="st-status-pill"
                    data-tone={saveTone}
                    onClick={() => void saveProject()}
                    title={saveError || 'Save project'}
                >
                    {saveStatus === 'saved' ? <CheckCircle2 size={11} /> : saveStatus === 'error' || saveStatus === 'conflict' ? <TriangleAlert size={11} /> : <Save size={11} />}
                    {saveLabel}
                </button>
                {health && (
                    <span className="st-status-pill" data-tone={health.gpu.busy ? 'warn' : 'good'} title="GPU slot / queue">
                        <HardDrive size={11} />
                        {health.gpu.busy ? 'GPU busy' : 'GPU free'}
                        {health.gpu.queued > 0 ? ` · ${health.gpu.queued} queued` : ''}
                    </span>
                )}
                {error && <span className="st-status-pill" data-tone="bad">Backend offline</span>}
                <button type="button" className="st-icon-button" title="Toggle inspector" onClick={() => setInspectorOpen(!inspectorOpen)}>
                    <MonitorPlay size={15} />
                </button>
                <button type="button" className="st-icon-button" title="Share / review link (F4)" disabled>
                    <Share2 size={15} />
                </button>
            </div>
        </header>
    );
}
