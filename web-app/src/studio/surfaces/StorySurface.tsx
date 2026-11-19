import { useEffect, useMemo, useRef, useState } from 'react';
import { ChevronLeft, ChevronRight, Search, Sparkles } from 'lucide-react';
import { useShallow } from 'zustand/react/shallow';
import { useStudioDocumentStore } from '../state/documentStore';
import { useStudioEditor } from '../state/editorStore';
import { useStudioElements } from '../data/useStudioData';
import { BoardCanvas } from '../components/BoardCanvas';

export function StorySurface() {
    const project = useStudioDocumentStore((state) => state.project);
    const { selectedShotId, selectShot, setCurrentTime } = useStudioDocumentStore(useShallow((state) => ({
        selectedShotId: state.selectedShotId,
        selectShot: state.selectShot,
        setCurrentTime: state.setCurrentTime,
    })));
    const {
        selectedSceneId, selectScene, spineCollapsed, setSpineCollapsed,
        mobileSpineOpen, setMobileSpineOpen,
    } = useStudioEditor(useShallow((state) => ({
        selectedSceneId: state.selectedSceneId,
        selectScene: state.selectScene,
        spineCollapsed: state.spineCollapsed,
        setSpineCollapsed: state.setSpineCollapsed,
        mobileSpineOpen: state.mobileSpineOpen,
        setMobileSpineOpen: state.setMobileSpineOpen,
    })));
    const { elements } = useStudioElements(project.id);
    const [query, setQuery] = useState('');
    const [elementFilter, setElementFilter] = useState<string | null>(null);
    const spineRef = useRef<HTMLElement | null>(null);

    useEffect(() => {
        if (!mobileSpineOpen) return undefined;
        const host = spineRef.current;
        const board = document.querySelector('.st-board-wrap');
        const previous = document.activeElement as HTMLElement | null;
        board?.setAttribute('inert', '');
        const focusable = () => Array.from(host?.querySelectorAll<HTMLElement>(
            'button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
        ) || []).filter((element) => element.offsetParent !== null);
        focusable()[0]?.focus();
        const onKeyDown = (event: KeyboardEvent) => {
            if (event.key === 'Escape') {
                event.preventDefault();
                event.stopPropagation();
                setMobileSpineOpen(false);
                return;
            }
            if (event.key !== 'Tab') return;
            const items = focusable();
            if (items.length === 0) return;
            const first = items[0];
            const last = items[items.length - 1];
            if (event.shiftKey && document.activeElement === first) {
                event.preventDefault();
                last.focus();
            } else if (!event.shiftKey && document.activeElement === last) {
                event.preventDefault();
                first.focus();
            }
        };
        document.addEventListener('keydown', onKeyDown, true);
        return () => {
            document.removeEventListener('keydown', onKeyDown, true);
            board?.removeAttribute('inert');
            previous?.focus?.();
        };
    }, [mobileSpineOpen, setMobileSpineOpen]);

    const groups = useMemo(() => {
        const scenes = project.scenes || [];
        const q = query.trim().toLowerCase();
        const visible = project.shots.filter((shot) => {
            if (q && !`${shot.action || ''} ${shot.prompt || ''} ${shot.character || ''}`.toLowerCase().includes(q)) {
                return false;
            }
            if (elementFilter && !shot.matchedElements?.some((match) => match.element_id === elementFilter)) {
                return false;
            }
            return true;
        });
        return scenes.map((scene) => ({
            id: scene.id,
            name: scene.name,
            shots: visible.filter((shot) => shot.sceneId === scene.id),
        })).concat([{
            id: '__unassigned',
            name: 'Unassigned',
            shots: visible.filter((shot) => !shot.sceneId),
        }]).filter((group) => group.shots.length > 0 || group.id !== '__unassigned');
    }, [elementFilter, project.shots, project.scenes, query]);

    return (
        <div className="st-story">
            {spineCollapsed ? (
                <aside className="st-spine st-spine-collapsed" aria-label="Story spine collapsed">
                    <button
                        type="button"
                        className="st-icon-button"
                        aria-label="Show story spine"
                        title="Show story spine"
                        onClick={() => setSpineCollapsed(false)}
                    >
                        <ChevronRight size={15} />
                    </button>
                </aside>
            ) : (
            <aside
                className="st-spine"
                aria-label="Story spine"
                role={mobileSpineOpen ? 'dialog' : undefined}
                aria-modal={mobileSpineOpen ? true : undefined}
                data-mobile-open={mobileSpineOpen}
                ref={spineRef}
            >
                <div className="st-spine-head">
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8 }}>
                        <div className="st-spine-title" style={{ marginBottom: 0 }}>Story spine</div>
                        <button
                            type="button"
                            className="st-icon-button"
                            aria-label="Hide story spine"
                            title="Hide story spine"
                            onClick={() => {
                            setSpineCollapsed(true);
                            setMobileSpineOpen(false);
                        }}
                        >
                            <ChevronLeft size={15} />
                        </button>
                    </div>
                    <label style={{ position: 'relative', display: 'block', marginTop: 8 }}>
                        <Search size={12} style={{ position: 'absolute', left: 8, top: 9, color: 'var(--st-faint)' }} />
                        <input
                            className="st-input"
                            style={{ paddingLeft: 26, height: 30 }}
                            placeholder="Search scenes and shots"
                            value={query}
                            onChange={(event) => setQuery(event.target.value)}
                        />
                    </label>
                </div>
                <div className="st-spine-list">
                    {groups.length === 0 && (
                        <div className="st-empty" style={{ minHeight: 120 }}>
                            <strong>No story yet</strong>
                            <span>Use Add shot on the board, or import a screenplay in Story → Script.</span>
                        </div>
                    )}
                    {groups.map((group) => (
                        <div className="st-spine-scene" key={group.id}>
                            <button
                                type="button"
                                className="st-spine-scene-name"
                                data-active={selectedSceneId === group.id}
                                onClick={() => selectScene(group.id === '__unassigned' ? null : group.id)}
                            >
                                <ChevronRight size={11} />
                                {group.name || 'Scene'}
                                <span style={{ marginLeft: 'auto', color: 'var(--st-faint)' }}>{group.shots.length}</span>
                            </button>
                            {group.shots.map((shot) => (
                                <button
                                    key={shot.id}
                                    type="button"
                                    className="st-spine-shot"
                                    data-selected={selectedShotId === shot.id}
                                    onClick={() => {
                                        selectShot(shot.id);
                                        setCurrentTime(0);
                                        setMobileSpineOpen(false);
                                    }}
                                >
                                    <span className="st-spine-shot-index st-mono">{project.shots.indexOf(shot) + 1}</span>
                                    <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                        {shot.action || shot.prompt || 'Untitled shot'}
                                    </span>
                                </button>
                            ))}
                        </div>
                    ))}
                    <div className="st-spine-refs">
                        <div className="st-spine-title" style={{ marginBottom: 6 }}>
                            <Sparkles size={10} style={{ verticalAlign: -1, marginRight: 4 }} />
                            Master references · {elements.length}
                        </div>
                        {elements.length === 0 ? (
                            <div style={{ color: 'var(--st-faint)', fontSize: 11, lineHeight: 1.5 }}>
                                Characters, locations, and objects appear here once created.
                            </div>
                        ) : (
                            <div className="st-chip-row">
                                {elements.map((element) => (
                                    <button
                                        key={element.id}
                                        type="button"
                                        className="st-chip"
                                        data-active={elementFilter === element.id}
                                        style={{ cursor: 'pointer' }}
                                        title={element.description}
                                        onClick={() => setElementFilter(elementFilter === element.id ? null : element.id)}
                                    >
                                        @{element.trigger_word.replace(/^@/, '')}
                                    </button>
                                ))}
                            </div>
                        )}
                        {elementFilter && (
                            <button
                                type="button"
                                className="st-button"
                                data-size="sm"
                                data-variant="ghost"
                                style={{ marginTop: 6 }}
                                onClick={() => setElementFilter(null)}
                            >
                                Clear reference filter
                            </button>
                        )}
                    </div>
                </div>
            </aside>
            )}
            {mobileSpineOpen && (
                <button
                    type="button"
                    className="st-mobile-backdrop"
                    aria-label="Close story spine"
                    onClick={() => setMobileSpineOpen(false)}
                />
            )}
            <BoardCanvas />
        </div>
    );
}
