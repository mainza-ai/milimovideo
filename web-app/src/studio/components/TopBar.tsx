import { useState } from 'react';
import {
    CheckCircle2, ChevronDown, FolderOpen, MessageSquare, MonitorPlay,
    Plus, Redo2, Save, Share2, TriangleAlert, Undo2,
} from 'lucide-react';
import { useStore } from 'zustand';
import { useShallow } from 'zustand/react/shallow';
import { apiUrl } from '../../config';
import { isDesktop, startDesktopBackend } from '../../desktop';
import { useStudioData, useStudioReview } from '../data/useStudioData';
import { useStudioEditor, type StudioSurface } from '../state/editorStore';
import { useStudioDocumentStore } from '../state/documentStore';

const SURFACES: Array<{ id: StudioSurface; label: string }> = [
    { id: 'story', label: 'Story' },
    { id: 'generate', label: 'Generate' },
    { id: 'edit', label: 'Edit' },
    { id: 'review', label: 'Review' },
    { id: 'library', label: 'Library' },
    { id: 'deliver', label: 'Deliver' },
    { id: 'shorts', label: 'Shorts' },
];

export function TopBar() {
    const { project, health, error } = useStudioData();
    const { review } = useStudioReview(project.id);
    const needsYou = review.summary.open + review.summary.changes_requested;
    const { activeSurface, setSurface, inspectorOpen, setInspectorOpen } = useStudioEditor(
        useShallow((state) => ({
            activeSurface: state.activeSurface,
            setSurface: state.setSurface,
            inspectorOpen: state.inspectorOpen,
            setInspectorOpen: state.setInspectorOpen,
        })),
    );
    const { saveStatus, saveError, saveProject } = useStudioDocumentStore();
    const addToast = useStudioDocumentStore((state) => state.addToast);
    const loadProject = useStudioDocumentStore((state) => state.loadProject);
    const createNewProject = useStudioDocumentStore((state) => state.createNewProject);
    const deleteProject = useStudioDocumentStore((state) => state.deleteProject);
    const [menuOpen, setMenuOpen] = useState(false);
    const [surfaceMenuOpen, setSurfaceMenuOpen] = useState(false);
    const [projects, setProjects] = useState<Array<{ id: string; name: string }>>([]);
    // zundo's temporal store is attached to the document store.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const temporalStore = (useStudioDocumentStore as any).temporal;
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

    const createShareLink = async () => {
        const response = await fetch(apiUrl(`/projects/${project.id}/share`), { method: 'POST' });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
            addToast(data.detail || 'Could not create a share link', 'error');
            return;
        }
        const url = `${window.location.origin}${data.path}`;
        try {
            await navigator.clipboard.writeText(url);
            addToast('Review link copied', 'success');
        } catch {
            window.prompt('Review link (copy it):', url);
        }
    };

    const toggleProjects = async () => {
        const next = !menuOpen;
        setMenuOpen(next);
        if (!next) return;
        try {
            const response = await fetch(apiUrl('/projects'));
            const data = response.ok ? await response.json() : [];
            setProjects(Array.isArray(data) ? data : []);
        } catch {
            setProjects([]);
        }
    };

    const newProject = async () => {
        const name = window.prompt('New project name', 'Untitled Project');
        if (!name?.trim()) return;
        await createNewProject(name.trim());
        setMenuOpen(false);
        addToast(`Created ${name.trim()}`, 'success');
    };

    return (
        <header className="st-topbar">
            <div className="st-brand">
                <img src="/logo.png" alt="" />
                <span className="st-brand-name">Milimo</span>
                <div className="st-project-switcher">
                    <input
                        className="st-project-name"
                        value={project.name}
                        aria-label="Project name"
                        onChange={(event) => useStudioDocumentStore.getState().setProject({ ...project, name: event.target.value })}
                        onBlur={() => void saveProject()}
                    />
                    <button
                        type="button"
                        className="st-icon-button"
                        onClick={() => void toggleProjects()}
                        aria-label="Switch project"
                        aria-expanded={menuOpen}
                        title="Projects"
                    >
                        <ChevronDown size={14} />
                    </button>
                    {menuOpen && (
                        <div className="st-project-menu" role="menu">
                            <div className="st-project-menu-head">
                                <span>Projects</span>
                                <button type="button" className="st-button" data-size="sm" onClick={() => void newProject()}>
                                    <Plus size={11} /> New
                                </button>
                            </div>
                            <div className="st-project-menu-list">
                                {projects.length === 0 && (
                                    <div className="st-project-menu-empty">No projects on this machine yet.</div>
                                )}
                                {projects.map((item) => (
                                    <button
                                        key={item.id}
                                        type="button"
                                        className="st-project-menu-item"
                                        data-active={item.id === project.id}
                                        onClick={async () => {
                                            await loadProject(item.id);
                                            setMenuOpen(false);
                                        }}
                                    >
                                        <FolderOpen size={12} />
                                        <span>{item.name}</span>
                                        <span className="st-mono">{item.id.slice(0, 8)}</span>
                                    </button>
                                ))}
                            </div>
                            <div style={{ marginTop: 8, paddingTop: 8, borderTop: '1px solid var(--st-border)' }}>
                                <button
                                    type="button"
                                    className="st-button"
                                    data-size="sm"
                                    data-variant="danger"
                                    onClick={async () => {
                                        if (!window.confirm(`Delete “${project.name}”? This removes its files and records.`)) return;
                                        await deleteProject(project.id);
                                        setMenuOpen(false);
                                    }}
                                >
                                    Delete current project
                                </button>
                            </div>
                        </div>
                    )}
                </div>
            </div>

            <div className="st-surface-compact">
                <button
                    type="button"
                    className="st-button"
                    data-size="sm"
                    aria-label="Switch studio surface"
                    aria-expanded={surfaceMenuOpen}
                    onClick={() => setSurfaceMenuOpen(!surfaceMenuOpen)}
                >
                    {SURFACES.find((surface) => surface.id === activeSurface)?.label || 'Story'}
                    <ChevronDown size={13} />
                </button>
                {surfaceMenuOpen && (
                    <div className="st-surface-compact-menu" role="menu">
                        {SURFACES.map((surface) => (
                            <button
                                key={surface.id}
                                type="button"
                                role="menuitem"
                                className="st-project-menu-item"
                                data-active={activeSurface === surface.id}
                                onClick={() => {
                                    setSurface(surface.id);
                                    setSurfaceMenuOpen(false);
                                }}
                            >
                                {surface.label}
                            </button>
                        ))}
                    </div>
                )}
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
                {needsYou > 0 && (
                    <button
                        type="button"
                        className="st-status-pill"
                        data-tone="warn"
                        onClick={() => setSurface('review')}
                        title="Open comments and change requests"
                    >
                        <MessageSquare size={11} /> {needsYou} needs you
                    </button>
                )}
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
                {error && (
                    <button
                        type="button"
                        className="st-status-pill"
                        data-tone="bad"
                        title={error}
                        onClick={async () => {
                            if (!isDesktop()) {
                                addToast('Start the backend with ./run_backend.sh', 'info');
                                return;
                            }
                            try {
                                const root = await startDesktopBackend();
                                addToast(`Starting backend from ${root}`, 'success');
                            } catch (exc) {
                                addToast(exc instanceof Error ? exc.message : String(exc), 'error');
                            }
                        }}
                    >
                        {isDesktop() ? 'Start backend' : 'Backend offline'}
                    </button>
                )}
                <button
                    type="button"
                    className="st-icon-button"
                    title="Toggle inspector"
                    aria-label="Toggle inspector"
                    onClick={() => setInspectorOpen(!inspectorOpen)}
                >
                    <MonitorPlay size={15} />
                </button>
                <button
                    type="button"
                    className="st-icon-button"
                    title={health?.auth_required ? 'Share links need local no-token mode' : 'Copy review link'}
                    aria-label="Copy review link"
                    disabled={health?.auth_required}
                    onClick={() => void createShareLink()}
                >
                    <Share2 size={15} />
                </button>
            </div>
        </header>
    );
}
