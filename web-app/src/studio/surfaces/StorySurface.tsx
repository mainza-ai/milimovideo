import { useMemo, useState } from 'react';
import { ChevronRight, Search, Sparkles } from 'lucide-react';
import { useShallow } from 'zustand/react/shallow';
import { useTimelineStore } from '../../stores/timelineStore';
import { useStudioEditor } from '../state/editorStore';
import { useStudioElements } from '../data/useStudioData';
import { BoardCanvas } from '../components/BoardCanvas';

export function StorySurface() {
    const project = useTimelineStore((state) => state.project);
    const { selectedShotId, selectShot, setCurrentTime } = useTimelineStore(useShallow((state) => ({
        selectedShotId: state.selectedShotId,
        selectShot: state.selectShot,
        setCurrentTime: state.setCurrentTime,
    })));
    const { selectedSceneId, selectScene } = useStudioEditor(useShallow((state) => ({
        selectedSceneId: state.selectedSceneId,
        selectScene: state.selectScene,
    })));
    const elements = useStudioElements(project.id);
    const [query, setQuery] = useState('');
    const [elementFilter, setElementFilter] = useState<string | null>(null);

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
            <aside className="st-spine" aria-label="Story spine">
                <div className="st-spine-head">
                    <div className="st-spine-title">Story spine</div>
                    <label style={{ position: 'relative', display: 'block' }}>
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
                            <span>Parse a script in the legacy editor; the spine will populate here.</span>
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
                            <div style={{ color: 'var(--st-faint)', fontSize: 9, lineHeight: 1.5 }}>
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
            <BoardCanvas />
        </div>
    );
}
