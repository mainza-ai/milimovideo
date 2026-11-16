import { useMemo, useState } from 'react';
import { ChevronRight, Search, Sparkles } from 'lucide-react';
import { useShallow } from 'zustand/react/shallow';
import { useTimelineStore } from '../../stores/timelineStore';
import { useStudioEditor } from '../state/editorStore';
import { BoardCanvas } from '../components/BoardCanvas';

export function StorySurface() {
    const project = useTimelineStore((state) => state.project);
    const { selectedShotId, selectShot, setCurrentTime } = useTimelineStore(useShallow((state) => ({
        selectedShotId: state.selectedShotId,
        selectShot: state.selectShot,
        setCurrentTime: state.setCurrentTime,
    })));
    const { selectedSceneId, selectScene } = useStudioEditor();
    const [query, setQuery] = useState('');

    const groups = useMemo(() => {
        const scenes = project.scenes || [];
        const q = query.trim().toLowerCase();
        const visible = q
            ? project.shots.filter((shot) =>
                `${shot.action || ''} ${shot.prompt || ''} ${shot.character || ''}`
                    .toLowerCase()
                    .includes(q),
            )
            : project.shots;
        return scenes.map((scene) => ({
            id: scene.id,
            name: scene.name,
            shots: visible.filter((shot) => shot.sceneId === scene.id),
        })).concat([{
            id: '__unassigned',
            name: 'Unassigned',
            shots: visible.filter((shot) => !shot.sceneId),
        }]).filter((group) => group.shots.length > 0 || group.id !== '__unassigned');
    }, [project.shots, project.scenes, query]);

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
                    <div style={{ padding: '10px 7px 4px', color: 'var(--st-faint)', fontSize: 9, lineHeight: 1.5 }}>
                        <Sparkles size={10} /> Master References area and typed links are next in F1.
                    </div>
                </div>
            </aside>
            <BoardCanvas />
        </div>
    );
}
