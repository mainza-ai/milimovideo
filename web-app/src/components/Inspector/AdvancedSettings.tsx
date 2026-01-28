import { clsx } from 'clsx';
import { ChevronDown, ChevronRight } from 'lucide-react';
import { useState } from 'react';
import { useTimelineStore, type Shot } from '../../stores/timelineStore';

interface AdvancedSettingsProps {
    shot: Shot;
}

export const AdvancedSettings = ({ shot }: AdvancedSettingsProps) => {
    const [isCollapsed, setIsCollapsed] = useState(false); // Default open in original? No, user had state for it.
    // Original: const [isAdvancedCollapsed, setIsAdvancedCollapsed] = useState(false);
    // So default NOT collapsed? 
    // Wait, original code: useState(false) -> not collapsed means visible?
    // "onClick={() => setIsAdvancedCollapsed(!isAdvancedCollapsed)}"
    // "{!isAdvancedCollapsed && ( ... )}"
    // So false means VISIBLE.
    // I'll default to true (collapsed) to cleaner look, or match original?
    // I will match original: false (visible).

    const updateShot = useTimelineStore(state => state.updateShot);

    return (
        <div className="space-y-4 pt-4 border-t border-white/5">
            <button
                className="flex justify-between items-center w-full group focus:outline-none"
                onClick={() => setIsCollapsed(!isCollapsed)}
            >
                <div className="flex items-center gap-2 text-[10px] uppercase tracking-widest text-white/40 font-bold group-hover:text-white/60">
                    {isCollapsed ? <ChevronRight size={12} /> : <ChevronDown size={12} />}
                    <span>Advanced Settings</span>
                </div>
            </button>

            {!isCollapsed && (
                <div className="space-y-4 animate-in fade-in slide-in-from-top-2 duration-300">
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
                                <input type="checkbox" className="hidden" checked={!!shot.enhancePrompt} onChange={(e) => updateShot(shot.id, { enhancePrompt: e.target.checked })} />
                            </div>
                            <span className="text-[10px] uppercase text-white/60 group-hover:text-white/80">Enhance</span>
                        </label>
                        <label className="flex items-center gap-2 cursor-pointer group">
                            <div className={clsx("w-3 h-3 rounded-sm border transition-colors flex items-center justify-center", shot.upscale ? "bg-milimo-500 border-milimo-500" : "border-white/20")} >
                                {shot.upscale && <div className="w-1.5 h-1.5 bg-black rounded-sm" />}
                                <input type="checkbox" className="hidden" checked={!!shot.upscale} onChange={(e) => updateShot(shot.id, { upscale: e.target.checked })} />
                            </div>
                            <span className="text-[10px] uppercase text-white/60 group-hover:text-white/80">Upscale</span>
                        </label>
                    </div>
                </div>
            )}
        </div>
    );
};
