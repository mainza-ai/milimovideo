import { clsx } from 'clsx';
import { Trash2 } from 'lucide-react';
import { useTimelineStore, type Shot } from '../../stores/timelineStore';

interface ConditioningEditorProps {
    shot: Shot;
}

export const ConditioningEditor = ({ shot }: ConditioningEditorProps) => {
    const { updateConditioning, removeConditioning, addConditioningToShot } = useTimelineStore();

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();

        const assetData = e.dataTransfer.getData('application/json');
        if (assetData) {
            try {
                const asset = JSON.parse(assetData);
                addConditioningToShot(shot.id, {
                    type: asset.type === 'video' ? 'video' : 'image',
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

    return (
        <div className="space-y-3 pt-4 border-t border-white/5">
            <div className="flex justify-between items-center">
                <label className="text-[10px] uppercase tracking-widest text-white/40 font-bold">Conditioning ({shot.timeline.length})</label>
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
                                    <img
                                        src={item.path.startsWith('http') ? item.path : `http://localhost:8000${item.path}`}
                                        className="w-full h-full object-cover"
                                        onError={(e) => e.currentTarget.style.display = 'none'}
                                        alt="conditioning"
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
    );
};
