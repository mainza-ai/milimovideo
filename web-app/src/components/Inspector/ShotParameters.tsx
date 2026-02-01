import { clsx } from 'clsx';
import { useTimelineStore, type Shot } from '../../stores/timelineStore';

interface ShotParametersProps {
    shot: Shot;
    fps: number;
}

export const ShotParameters = ({ shot, fps }: ShotParametersProps) => {
    const updateShot = useTimelineStore(state => state.updateShot);

    return (
        <>
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
                <label className="text-[10px] uppercase tracking-widest text-white/40">FPS / Duration</label>
                <div className="flex gap-2">
                    <select
                        className="bg-white/5 border border-white/10 rounded px-1 py-1 text-xs w-16"
                        value={shot.fps || fps || 25}
                        onChange={(e) => updateShot(shot.id, { fps: parseInt(e.target.value) })}
                    >
                        {[24, 25, 30, 50, 60].map(f => (
                            <option key={f} value={f}>{f} FPS</option>
                        ))}
                    </select>

                    <div className="flex-1 flex bg-white/5 rounded border border-white/10 p-0.5 gap-1">
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

            {/* Duration Slider (Only if Video) */}
            {shot.numFrames > 1 && (
                <div className="space-y-1">
                    <div className="flex justify-between text-[10px] text-white/60">
                        <span>Duration</span>
                        <span>{(shot.numFrames / (shot.fps || fps || 25)).toFixed(1)}s ({shot.numFrames}f)</span>
                    </div>
                    <input
                        type="range" min="25" max="1200" step="1"
                        className="w-full h-1 bg-white/10 rounded appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-2 [&::-webkit-slider-thumb]:h-2 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-milimo-500"
                        value={shot.numFrames}
                        onChange={(e) => updateShot(shot.id, { numFrames: parseInt(e.target.value) })}
                    />
                </div>
            )}
        </>
    );
};
