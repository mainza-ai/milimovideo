import { useTimelineStore } from '../../stores/timelineStore';
import { Plus, GripVertical, Image as ImageIcon, Film, Play, Sparkles } from 'lucide-react';
import { ScriptInput } from './ScriptInput';
import React from 'react';

export const StoryboardView = () => {
    const { project, selectShot, selectedShotId, generateShot, loadProject } = useTimelineStore();

    // Polling for active jobs
    React.useEffect(() => {
        const hasActiveJobs = project.shots.some(s => s.statusMessage === 'Generating...' || s.statusMessage === 'Queued...');

        let interval: ReturnType<typeof setInterval>;
        if (hasActiveJobs) {
            interval = setInterval(async () => {
                // Ideally we'd poll specific jobs, but reloading project is safer for now to get full state
                await loadProject(project.id);
            }, 3000);
        }

        return () => clearInterval(interval);
    }, [project.shots, project.id, loadProject]);

    const handleGenerate = async (e: React.MouseEvent, shotId: string) => {
        e.stopPropagation();
        await generateShot(shotId);
    };

    return (
        <div className="h-full bg-[#111] p-8 overflow-y-auto custom-scrollbar">
            <h2 className="text-2xl font-bold text-white mb-6 tracking-tight">Storyboard Engine</h2>

            {/* 1. Script Inputs */}
            <ScriptInput />

            <div className="border-t border-white/10 pt-8">
                <h3 className="text-lg font-bold text-white mb-4">Shot List ({project.shots.length})</h3>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                    {/* Add New Shot Card (Manual) */}
                    <button
                        className="aspect-video bg-white/5 hover:bg-white/10 border-2 border-dashed border-white/10 hover:border-milimo-500/50 rounded-xl flex flex-col items-center justify-center text-white/30 hover:text-white transition-all group"
                        onClick={() => console.log("Add shot via storyboard")}
                    >
                        <div className="w-12 h-12 rounded-full bg-white/5 group-hover:bg-milimo-500 group-hover:text-black flex items-center justify-center mb-2 transition-colors">
                            <Plus size={24} />
                        </div>
                        <span className="text-sm font-medium">Add Manual Shot</span>
                    </button>

                    {project.shots.map((shot, index) => (
                        <div
                            key={shot.id}
                            onClick={() => selectShot(shot.id)}
                            className={`relative group bg-[#1a1a1a] rounded-xl overflow-hidden border-2 transition-all cursor-pointer flex flex-col ${shot.id === selectedShotId ? 'border-milimo-500 shadow-xl shadow-milimo-500/10' : 'border-transparent hover:border-white/20'}`}
                        >
                            {/* Header / ID */}
                            <div className="px-3 py-2 flex justify-between items-center bg-black/40 border-b border-white/5">
                                <span className="text-xs font-mono text-white/40">SHOT {index + 1}</span>
                                <div className="flex items-center gap-2">
                                    {/* Status Indicators */}
                                    {shot.statusMessage && (
                                        <span className="text-[10px] uppercase font-bold text-milimo-400 animate-pulse">{shot.statusMessage}</span>
                                    )}
                                    <GripVertical size={14} className="text-white/20 hover:text-white cursor-grab active:cursor-grabbing" />
                                </div>
                            </div>

                            {/* Thumbnail / Content */}
                            <div className="aspect-video bg-black relative w-full group">
                                {shot.videoUrl ? (
                                    shot.videoUrl.match(/\.(jpg|jpeg|png|webp)$/i) ?
                                        <img src={shot.videoUrl} className="w-full h-full object-cover" />
                                        : <video src={shot.videoUrl} className="w-full h-full object-cover" controls={false} />
                                ) : (
                                    <div className="absolute inset-0 flex flex-col items-center justify-center text-white/10 p-4 text-center">
                                        <ImageIcon size={32} className="mb-2 opacity-50" />
                                        <p className="text-[10px] line-clamp-3 opacity-50">{shot.action || shot.prompt}</p>
                                    </div>
                                )}

                                {/* Overlay Gradient */}
                                <div className="absolute inset-0 bg-gradient-to-t from-black/90 via-transparent to-transparent opacity-60 pointer-events-none" />

                                {/* Prompt Text Overlay */}
                                <div className="absolute bottom-2 left-3 right-3 text-xs text-white/90 line-clamp-3 font-medium pointer-events-none drop-shadow-md">
                                    {shot.character && (
                                        <span className="block text-[10px] font-bold text-milimo-300 uppercase mb-0.5 tracking-wider">{shot.character}</span>
                                    )}
                                    {shot.dialogue ? (
                                        <span className="italic text-white">"{shot.dialogue}"</span>
                                    ) : (
                                        shot.action || shot.prompt
                                    )}
                                </div>

                                {/* Hover Action: Generate */}
                                <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center backdrop-blur-sm">
                                    {!shot.videoUrl && (
                                        <button
                                            onClick={(e) => handleGenerate(e, shot.id)}
                                            className="px-4 py-2 bg-milimo-500 text-black font-bold rounded-lg transform scale-95 group-hover:scale-100 transition-transform flex items-center gap-2"
                                        >
                                            <Sparkles size={16} />
                                            Generate
                                        </button>
                                    )}
                                    {shot.videoUrl && (
                                        <div className="flex gap-2">
                                            <button className="p-2 bg-white/10 hover:bg-white/20 rounded-full text-white"><Play size={20} fill="currentColor" /></button>
                                            <button className="p-2 bg-white/10 hover:bg-white/20 rounded-full text-white" onClick={(e) => handleGenerate(e, shot.id)} title="Regenerate"><Sparkles size={20} /></button>
                                        </div>
                                    )}
                                </div>
                            </div>

                            {/* Footer Stats */}
                            <div className="px-3 py-2 bg-[#151515] flex justify-between items-center text-[10px] text-white/30 font-mono mt-auto">
                                <div className="flex gap-2">
                                    <span className="flex items-center gap-1"><Film size={10} /> {shot.numFrames}f</span>
                                </div>
                                <span className="uppercase tracking-wider">{shot.width}x{shot.height}</span>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
};
