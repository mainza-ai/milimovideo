import { useTimelineStore } from '../../stores/timelineStore';
import { useShallow } from 'zustand/react/shallow';
import { Settings, Play, Trash2 } from 'lucide-react';
import { useEffect } from 'react';
import { pollJobStatus } from '../../utils/jobPoller';
import { NarrativeDirector } from './NarrativeDirector';
import { ShotParameters } from './ShotParameters';
import { AdvancedSettings } from './AdvancedSettings';
import { ConditioningEditor } from './ConditioningEditor';

export const InspectorPanel = () => {
    const {
        project, selectedShotId, updateShot, deleteShot,
        addShot, addConditioningToShot
    } = useTimelineStore(useShallow(state => ({
        project: state.project,
        selectedShotId: state.selectedShotId,
        updateShot: state.updateShot,
        deleteShot: state.deleteShot,
        addShot: state.addShot,
        addConditioningToShot: state.addConditioningToShot
    })));

    const shot = project.shots.find(s => s.id === selectedShotId);

    // Auto-resume polling on mount if generating
    useEffect(() => {
        if (shot?.isGenerating && shot?.lastJobId) {
            pollJobStatus(shot.lastJobId, shot.id);
        }
    }, [shot?.id, shot?.isGenerating, shot?.lastJobId]);

    if (!shot) {
        return (
            <div className="w-80 h-full bg-[#0a0a0a] border-l border-white/5 flex flex-col items-center justify-center text-white/20 p-8 text-center">
                <Settings size={48} className="mb-4 opacity-20" />
                <p className="text-sm">Select a shot to edit parameters</p>
                <div className="mt-8 pt-8 border-t border-white/5 w-full">
                    <h4 className="text-xs uppercase tracking-widest font-semibold mb-4 text-white/40">Project Settings</h4>
                    <div className="flex justify-between text-xs text-white/50 mb-2">
                        <span>Resolution</span>
                        <span>{project.resolutionW}x{project.resolutionH}</span>
                    </div>
                    <div className="flex justify-between text-xs text-white/50">
                        <span>FPS</span>
                        <span>{project.fps}</span>
                    </div>
                </div>
            </div>
        );
    }

    const handleGenerate = async () => {
        if (!shot) return;
        updateShot(shot.id, { isGenerating: true, progress: 0 });

        try {
            const response = await fetch('http://localhost:8000/generate/advanced', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    project_id: project.id,
                    shot_config: {
                        // Explicitly map fields to match backend ShotConfig(BaseModel)
                        // Do NOT spread ...shot to avoid sending junk(UI state, etc) causing 422
                        id: shot.id,
                        prompt: shot.prompt,
                        negative_prompt: shot.negativePrompt,
                        seed: shot.seed,
                        width: shot.width,
                        height: shot.height,
                        fps: shot.fps || project.fps,
                        num_frames: shot.numFrames,
                        num_inference_steps: 40,
                        cfg_scale: shot.cfgScale,
                        enhance_prompt: shot.enhancePrompt,
                        upscale: shot.upscale,
                        pipeline_override: shot.pipelineOverride,
                        auto_continue: !!shot.autoContinue,

                        timeline: shot.timeline.map(t => ({
                            type: t.type,
                            path: t.path,
                            frame_index: t.frameIndex,
                            strength: t.strength
                        }))
                    }
                })
            });

            const data = await response.json();
            if (data.job_id) {
                updateShot(shot.id, { lastJobId: data.job_id });
                pollJobStatus(data.job_id, shot.id);
            } else {
                updateShot(shot.id, { isGenerating: false });
                alert("Failed to start generation");
            }

        } catch (e) {
            console.error(e);
            updateShot(shot.id, { isGenerating: false });
        }
    };

    const handleCancel = async () => {
        if (!shot?.lastJobId) return;
        try {
            await fetch(`http://localhost:8000/jobs/${shot.lastJobId}/cancel`, { method: 'POST' });
            updateShot(shot.id, { isGenerating: false, progress: 0 });
        } catch (e) { console.error(e); }
    };

    // Extend Shot Logic
    const handleExtend = async () => {
        let jobId = shot?.lastJobId;

        // Fallback: extract from videoUrl if valid
        if (!jobId && shot?.videoUrl) {
            const match = shot.videoUrl.match(/generated\/(job_[a-f0-9]+)/);
            if (match) jobId = match[1];
        }

        if (!jobId) {
            console.error("No job ID found for this shot");
            return;
        }

        try {
            // 1. Get Last Frame
            const res = await fetch(`http://localhost:8000/shot/${jobId}/last-frame`, {
                method: 'POST'
            });
            if (!res.ok) {
                alert("Could not extract last frame.");
                return;
            }
            const data = await res.json();

            // 2. Create New Shot
            addShot(); // Use hook action

            // 3. Get the ID of the newly created shot
            const newShotId = useTimelineStore.getState().selectedShotId;

            if (!newShotId) return;

            // 4. Update New Shot Prompt with Previous Context
            const store = useTimelineStore.getState();
            store.updateShot(newShotId, {
                prompt: shot.prompt, // Carry over prompt
                negativePrompt: shot.negativePrompt
            });

            // 5. Add Conditioning to New Shot
            const item: any = {
                id: crypto.randomUUID(),
                type: 'image',
                path: data.url,
                frameIndex: 0,
                strength: 1.0
            };
            addConditioningToShot(newShotId, item);

        } catch (e) {
            console.error(e);
            alert("Extend failed");
        }
    };

    return (
        <div className="w-80 h-full bg-[#0a0a0a] border-l border-white/5 flex flex-col overflow-y-auto">
            <div className="p-6 border-b border-white/5 flex justify-between items-center">
                <div className="flex items-center gap-4">
                    <h3 className="text-sm font-semibold text-white tracking-wider uppercase">Shot Inspector</h3>
                    {shot?.videoUrl && (
                        <button onClick={handleExtend} className="text-[10px] text-milimo-400 border border-milimo-500/30 px-2 py-1 rounded hover:bg-milimo-500/10 transition-colors uppercase font-bold">
                            Extend
                        </button>
                    )}
                </div>
                <button
                    onClick={() => deleteShot(shot.id)}
                    className="text-white/20 hover:text-red-500 transition-colors"
                >
                    <Trash2 size={16} />
                </button>
            </div>

            <div className="p-6 space-y-6 flex-1">
                {/* Live Evolving Prompt OR Final Enhanced Prompt (Collapsible) */}
                <NarrativeDirector shot={shot} />

                {/* Prompt */}
                <div className="space-y-2">
                    <label className="text-[10px] uppercase tracking-widest text-white/40 font-bold">Prompt</label>
                    <textarea
                        className="w-full bg-white/5 border border-white/10 rounded-lg p-3 text-sm text-white/90 focus:outline-none focus:border-milimo-500/50 min-h-[100px] resize-none"
                        value={shot.prompt}
                        onChange={(e) => updateShot(shot.id, { prompt: e.target.value })}
                        placeholder="Describe the shot..."
                    />
                </div>

                <div className="space-y-2">
                    <label className="text-[10px] uppercase tracking-widest text-white/40 font-bold">Negative Prompt</label>
                    <textarea
                        className="w-full bg-white/5 border border-white/10 rounded-lg p-3 text-xs text-white/70 focus:outline-none focus:border-milimo-500/50 min-h-[60px] resize-none"
                        value={shot.negativePrompt}
                        onChange={(e) => updateShot(shot.id, { negativePrompt: e.target.value })}
                    />
                </div>

                {/* Enhanced Prompt Result */}
                {shot.enhancedPromptResult && (
                    <div className="space-y-2">
                        <label className="text-[10px] uppercase tracking-widest text-milimo-500 font-bold flex items-center gap-2">
                            <span>✨ Enhanced Prompt</span>
                        </label>
                        <div className="w-full bg-milimo-500/10 border border-milimo-500/30 rounded-lg p-3 text-xs text-milimo-200/90 italic font-medium min-h-[60px] max-h-[120px] overflow-y-auto">
                            {shot.enhancedPromptResult}
                        </div>
                    </div>
                )}

                {/* Parameters Grid */}
                <ShotParameters shot={shot} fps={project.fps} />

                {/* Advanced Controls (Collapsible) */}
                <AdvancedSettings shot={shot} />
            </div>

            {/* Conditioning Visualizer */}
            <ConditioningEditor shot={shot} />


            {/* Footer Actions */}
            <div className="p-6 border-t border-white/5">
                {shot.isGenerating ? (
                    <div className="flex gap-2 w-full">
                        <button
                            disabled
                            className="flex-1 bg-white/5 text-white/50 rounded-xl flex flex-col items-center justify-center gap-1 font-bold uppercase tracking-wider text-xs border border-white/5 px-2"
                        >
                            <span>{(shot as any).statusMessage || "Generating..."}</span>
                            <span className="text-[10px] opacity-60">{(shot as any).progress || 0}%</span>
                        </button>
                        <button
                            onClick={handleCancel}
                            className="w-12 bg-red-500/10 border border-red-500/20 text-red-500 hover:bg-red-500/20 rounded-xl flex items-center justify-center transition-colors"
                            title="Cancel"
                        >
                            X
                        </button>
                    </div>
                ) : (
                    <button
                        onClick={handleGenerate}
                        className="w-full py-4 rounded-xl flex items-center justify-center gap-2 font-bold uppercase tracking-wider text-xs transition-all bg-milimo-500 hover:bg-milimo-400 text-black shadow-lg shadow-milimo-500/25"
                    >
                        <Play size={16} fill="currentColor" /> Generate Shot
                    </button>
                )}
            </div>
        </div >
    );
};
