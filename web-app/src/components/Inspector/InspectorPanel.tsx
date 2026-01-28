import { useTimelineStore } from '../../stores/timelineStore';
import { clsx } from 'clsx';
import { Settings, Play, Trash2 } from 'lucide-react';

export const InspectorPanel = () => {
    const {
        project, selectedShotId, updateShot, deleteShot,
        updateConditioning, removeConditioning
    } = useTimelineStore();

    const shot = project.shots.find(s => s.id === selectedShotId);

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

    const pollJobStatus = async (jobId: string, shotId: string) => {
        const poll = async () => {
            try {
                const res = await fetch(`http://localhost:8000/status/${jobId}`);
                const statusData = await res.json();

                if (statusData.status === 'completed') {
                    updateShot(shotId, {
                        isGenerating: false,
                        videoUrl: statusData.video_url, // This might be an image URL now, which is fine
                        progress: 100
                    });

                    // Trigger refresh of history
                    const { triggerAssetRefresh } = useTimelineStore.getState();
                    triggerAssetRefresh();
                } else if (statusData.status === 'failed') {
                    updateShot(shotId, { isGenerating: false, progress: 0 });
                    alert(`Generation Failed: ${statusData.error || 'Unknown error'}`);
                } else {
                    // Still processing
                    if (statusData.progress !== undefined) {
                        updateShot(shotId, { progress: statusData.progress });
                    }
                    setTimeout(poll, 1000);
                }
            } catch (e) {
                console.error("Polling error", e);
                setTimeout(poll, 2000);
            }
        };
        poll();
    };

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
                        ...shot,
                        timeline: shot.timeline.map(t => ({
                            type: t.type,
                            path: t.path,
                            frame_index: t.frameIndex,
                            strength: t.strength
                        })),
                        // Explicitly map camelCase to snake_case for backend
                        num_frames: shot.numFrames,
                        negative_prompt: shot.negativePrompt,
                        num_inference_steps: 40, // Default or add to store if needed

                        cfg_scale: shot.cfgScale,
                        enhance_prompt: shot.enhancePrompt,
                        upscale: shot.upscale,
                        pipeline_override: shot.pipelineOverride
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
            await fetch(`http://localhost:8000/cancel/${shot.lastJobId}`, { method: 'POST' });
            updateShot(shot.id, { isGenerating: false, progress: 0 });
        } catch (e) { console.error(e); }
    };

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();

        // Try getting data from dataTransfer (set by MediaLibrary)
        const assetData = e.dataTransfer.getData('application/json');
        if (assetData) {
            try {
                const asset = JSON.parse(assetData);
                // Add to timeline
                const { addConditioningToShot } = useTimelineStore.getState();
                addConditioningToShot(shot.id, {
                    type: asset.type === 'video' ? 'video' : 'image', // normalize
                    path: asset.url,
                    frameIndex: 0,
                    strength: 1.0
                });
            } catch (err) {
                console.error("Failed to parse dropped asset", err);
            }
        }
    };

    const handleDragOver = (e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
    };

    // Extend Shot Logic
    const handleExtend = async () => {
        if (!shot?.lastJobId) return;

        try {
            // 1. Get Last Frame
            const res = await fetch(`http://localhost:8000/shot/${shot.lastJobId}/last-frame`, {
                method: 'POST'
            });
            if (!res.ok) {
                alert("Could not extract last frame.");
                return;
            }
            const data = await res.json();

            // 2. Create New Shot
            const { useTimelineStore } = await import('../../stores/timelineStore');
            const store = useTimelineStore.getState();

            // Add new shot (takes no args, automatically selects it)
            store.addShot();

            // 3. Get the ID of the newly created shot
            const newShotId = useTimelineStore.getState().selectedShotId;

            if (!newShotId) return;

            // 4. Update New Shot Prompt with Previous Context
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
            store.addConditioningToShot(newShotId, item);

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

                {/* Parameters Grid */}
                <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-1">
                        <label className="text-[10px] uppercase tracking-widest text-white/40">Seed</label>
                        <input
                            type="number"
                            className="w-full bg-white/5 border border-white/10 rounded px-2 py-1 text-xs"
                            value={shot.seed}
                            onChange={(e) => updateShot(shot.id, { seed: parseInt(e.target.value) })}
                        />
                    </div>
                    <label className="text-[10px] uppercase tracking-widest text-white/40">Duration / Mode</label>
                    <div className="flex gap-2">
                        {/* Mode Toggle */}
                        <div className="flex-1 flex bg-white/5 rounded border border-white/10 p-0.5">
                            <button
                                onClick={() => updateShot(shot.id, { numFrames: 121 })}
                                className={clsx(
                                    "flex-1 text-[9px] uppercase font-bold rounded py-1 transition-all",
                                    shot.numFrames > 1 ? "bg-milimo-500 text-black shadow-sm" : "text-white/40 hover:text-white"
                                )}
                            >
                                Video
                            </button>
                            <button
                                onClick={() => updateShot(shot.id, { numFrames: 1 })}
                                className={clsx(
                                    "flex-1 text-[9px] uppercase font-bold rounded py-1 transition-all",
                                    shot.numFrames === 1 ? "bg-milimo-500 text-black shadow-sm" : "text-white/40 hover:text-white"
                                )}
                            >
                                Image
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            {/* Duration Slider (Only if Video) */}
            {shot.numFrames > 1 && (
                <div className="space-y-1">
                    <div className="flex justify-between text-[10px] text-white/60">
                        <span>Duration</span>
                        <span>{(shot.numFrames / (project.fps || 25)).toFixed(1)}s ({shot.numFrames}f)</span>
                    </div>
                    <input
                        type="range" min="25" max="250" step="1"
                        className="w-full h-1 bg-white/10 rounded appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-2 [&::-webkit-slider-thumb]:h-2 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-milimo-500"
                        value={shot.numFrames}
                        onChange={(e) => updateShot(shot.id, { numFrames: parseInt(e.target.value) })}
                    />
                </div>
            )}

            {/* Advanced Controls */}
            <div className="space-y-4 pt-4 border-t border-white/5">
                <div className="flex justify-between items-center text-[10px] uppercase tracking-widest text-white/40 font-bold">
                    <span>Advanced Settings</span>
                </div>

                {/* Pipeline Override */}
                <div className="space-y-1">
                    <label className="text-[10px] uppercase tracking-widest text-white/40">Pipeline / Mode</label>
                    <select
                        className="w-full bg-white/5 border border-white/10 rounded px-2 py-1.5 text-xs text-white/80 focus:outline-none focus:border-milimo-500"
                        value={shot.pipelineOverride || 'auto'}
                        onChange={(e) => updateShot(shot.id, { pipelineOverride: e.target.value as any })}
                    >
                        <option value="auto">Auto (Smart Detect)</option>
                        <option value="ti2vid">Text/Image to Video</option>
                        <option value="ic_lora">Video to Video (IC LoRA)</option>
                        <option value="keyframe">Keyframe Interpolation</option>
                    </select>
                </div>

                {/* CFG Scale */}
                <div className="space-y-1">
                    <div className="flex justify-between text-[10px] text-white/60">
                        <span>CFG Scale</span>
                        <span>{shot.cfgScale?.toFixed(1) || 3.0}</span>
                    </div>
                    <input
                        type="range" min="1" max="20" step="0.5"
                        className="w-full h-1 bg-white/10 rounded appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-2 [&::-webkit-slider-thumb]:h-2 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-milimo-500"
                        value={shot.cfgScale || 3.0}
                        onChange={(e) => updateShot(shot.id, { cfgScale: parseFloat(e.target.value) })}
                    />
                </div>

                {/* Toggles */}
                <div className="flex gap-4">
                    <label className="flex items-center gap-2 cursor-pointer group">
                        <div className={clsx("w-3 h-3 rounded-sm border transition-colors flex items-center justify-center", shot.enhancePrompt ? "bg-milimo-500 border-milimo-500" : "border-white/20")} >
                            {shot.enhancePrompt && <div className="w-1.5 h-1.5 bg-black rounded-sm" />}
                            <input type="checkbox" className="hidden" checked={shot.enhancePrompt} onChange={(e) => updateShot(shot.id, { enhancePrompt: e.target.checked })} />
                        </div>
                        <span className="text-[10px] uppercase text-white/60 group-hover:text-white/80">Enhance</span>
                    </label>
                    <label className="flex items-center gap-2 cursor-pointer group">
                        <div className={clsx("w-3 h-3 rounded-sm border transition-colors flex items-center justify-center", shot.upscale ? "bg-milimo-500 border-milimo-500" : "border-white/20")} >
                            {shot.upscale && <div className="w-1.5 h-1.5 bg-black rounded-sm" />}
                            <input type="checkbox" className="hidden" checked={shot.upscale} onChange={(e) => updateShot(shot.id, { upscale: e.target.checked })} />
                        </div>
                        <span className="text-[10px] uppercase text-white/60 group-hover:text-white/80">Upscale</span>
                    </label>
                </div>
            </div>

            {/* Conditioning Visualizer */}
            <div className="space-y-3 pt-4 border-t border-white/5">
                <div className="flex justify-between items-center">
                    <label className="text-[10px] uppercase tracking-widest text-white/40 font-bold">Conditioning ({shot.timeline.length})</label>
                    {/* Placeholder Add Button - functional only via DnD currently */}
                </div>

                <div
                    onDrop={handleDrop}
                    onDragOver={handleDragOver}
                    className={clsx(
                        "min-h-[60px] rounded transition-colors border-dashed border border-white/10",
                        shot.timeline.length === 0 ? "flex items-center justify-center" : "space-y-2 p-2"
                    )}
                >
                    {shot.timeline.length === 0 ? (
                        <div className="text-xs text-white/20 italic text-center pointer-events-none">
                            Drag images from library
                        </div>
                    ) : (
                        <>
                            {shot.timeline.map((item) => (
                                <div key={item.id} className="flex gap-2 items-center bg-white/5 p-2 rounded border border-white/5">
                                    <div className="w-8 h-8 bg-black rounded overflow-hidden relative">
                                        {/* Show Image preview if possible, otherwise generic */}
                                        {/* Show Image preview if possible, otherwise generic */}
                                        <img
                                            src={item.path.startsWith('http') ? item.path : `http://localhost:8000${item.path}`}
                                            className="w-full h-full object-cover"
                                            onError={(e) => e.currentTarget.style.display = 'none'}
                                        />
                                    </div>
                                    <div className="flex-1 min-w-0">
                                        <div className="flex gap-2 mb-1">
                                            <div className="flex-1">
                                                <label className="text-[9px] uppercase text-white/40 block mb-0.5">Frame</label>
                                                <input
                                                    type="number"
                                                    className="w-full bg-black/20 border border-white/10 rounded px-1 text-[10px] text-white/80"
                                                    value={item.frameIndex || 0}
                                                    onChange={(e) => updateConditioning(shot.id, item.id, { frameIndex: parseInt(e.target.value) })}
                                                />
                                            </div>
                                            <div className="flex-[2]">
                                                <label className="text-[9px] uppercase text-white/40 block mb-0.5">Strength: <span className="text-milimo-400">{item.strength.toFixed(1)}</span></label>
                                                <input
                                                    type="range" min="0" max="1" step="0.1"
                                                    className="w-full h-1 bg-white/10 rounded appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-2 [&::-webkit-slider-thumb]:h-2 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-milimo-500"
                                                    value={item.strength}
                                                    onChange={(e) => updateConditioning(shot.id, item.id, { strength: parseFloat(e.target.value) })}
                                                />
                                            </div>
                                        </div>
                                        <div className="flex justify-between text-[10px] uppercase">
                                            <span className="text-white/60">{item.type}</span>
                                        </div>
                                    </div>
                                    <button
                                        onClick={() => removeConditioning(shot.id, item.id)}
                                        className="text-white/20 hover:text-red-400"
                                    >
                                        <Trash2 size={12} />
                                    </button>
                                </div>
                            ))}
                        </>
                    )}
                </div>
            </div>


            {/* Footer Actions */}
            <div className="p-6 border-t border-white/5">
                {shot.isGenerating ? (
                    <div className="flex gap-2 w-full">
                        <button
                            disabled
                            className="flex-1 bg-white/5 text-white/50 rounded-xl flex items-center justify-center gap-2 font-bold uppercase tracking-wider text-xs border border-white/5"
                        >
                            Generating {(shot as any).progress || 0}%
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
