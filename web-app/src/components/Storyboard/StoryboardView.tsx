import { useTimelineStore } from '../../stores/timelineStore';
import { useShallow } from 'zustand/react/shallow';
import { ScriptInput } from './ScriptInput';
import { StoryboardSceneGroup } from './StoryboardSceneGroup';
import type { Shot, Scene } from '../../stores/types';
import { useMemo } from 'react';
import { Film, Layers, ArrowRightToLine, ChevronDown } from 'lucide-react';

export const StoryboardView = () => {
    const { project, pushStoryboardToTimeline } = useTimelineStore(useShallow(state => ({
        project: state.project,
        pushStoryboardToTimeline: state.pushStoryboardToTimeline,
    })));

    // Group shots by sceneId into Scene objects
    const sceneGroups = useMemo(() => {
        const scenes = project.scenes || [];
        const shotsByScene = new Map<string, Shot[]>();

        // Build scene → shots mapping
        for (const shot of project.shots) {
            const key = shot.sceneId || '__unassigned__';
            if (!shotsByScene.has(key)) shotsByScene.set(key, []);
            shotsByScene.get(key)!.push(shot);
        }

        // Build scene list with shots attached
        const result: { scene: Scene; shots: Shot[] }[] = [];

        for (const scene of scenes) {
            const sceneShots = shotsByScene.get(scene.id) || [];
            // Sort by index
            sceneShots.sort((a, b) => (a.index ?? 0) - (b.index ?? 0));
            result.push({ scene, shots: sceneShots });
            shotsByScene.delete(scene.id);
        }

        // Handle unassigned shots (shots without a scene)
        const unassigned = shotsByScene.get('__unassigned__') || [];
        if (unassigned.length > 0) {
            result.push({
                scene: {
                    id: '__unassigned__',
                    index: result.length,
                    name: 'Unassigned Shots',
                    shots: unassigned,
                },
                shots: unassigned,
            });
        }

        return result;
    }, [project.shots, project.scenes]);

    const totalShots = project.shots.length;
    const totalScenes = sceneGroups.length;
    const completedShots = project.shots.filter(s => s.videoUrl).length;

    return (
        <div className="h-full bg-[#111] p-8 overflow-y-auto custom-scrollbar">
            <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold text-white tracking-tight">Storyboard Engine</h2>
                <div className="flex items-center gap-4">
                    {completedShots > 0 && (
                        <button
                            onClick={pushStoryboardToTimeline}
                            className="px-4 py-1.5 bg-milimo-500/20 hover:bg-milimo-500/30 text-milimo-400 text-xs font-semibold rounded-lg flex items-center gap-2 transition-colors"
                        >
                            <ArrowRightToLine size={14} />
                            Push to Timeline ({completedShots})
                        </button>
                    )}
                    <div className="flex items-center gap-4 text-xs font-mono text-white/30">
                        <span className="flex items-center gap-1"><Layers size={12} /> {totalScenes} scenes</span>
                        <span className="flex items-center gap-1"><Film size={12} /> {totalShots} shots</span>
                    </div>
                </div>
            </div>

            {/* Script Input */}
            <ScriptInput />

            {/* Scene Groups */}
            <div className="border-t border-white/10 pt-6 mt-2">
                {sceneGroups.length === 0 ? (
                    <div className="text-center py-16 text-white/20">
                        <Layers size={48} className="mx-auto mb-4 opacity-30" />
                        <p className="text-lg font-medium mb-1">No storyboard yet</p>
                        <p className="text-sm">Write a script above and click "Analyze Script" to get started.</p>
                    </div>
                ) : (
                    (() => {
                        let shotOffset = 0;
                        return sceneGroups.map(({ scene, shots }, groupIdx) => {
                            const groupOffset = shotOffset;
                            shotOffset += shots.length;
                            return (
                                <div key={scene.id}>
                                    {/* Scene boundary connector */}
                                    {groupIdx > 0 && (
                                        <div className="flex items-center gap-3 py-2 px-4">
                                            <div className="flex-1 border-t border-dashed border-white/10" />
                                            <ChevronDown size={12} className="text-white/20" />
                                            <span className="text-[10px] font-mono text-white/20">
                                                Scene {groupIdx} → {groupIdx + 1}
                                            </span>
                                            <div className="flex-1 border-t border-dashed border-white/10" />
                                        </div>
                                    )}
                                    <StoryboardSceneGroup
                                        scene={scene}
                                        shots={shots}
                                        shotOffset={groupOffset}
                                    />
                                </div>
                            );
                        });
                    })()
                )}
            </div>
        </div>
    );
};
