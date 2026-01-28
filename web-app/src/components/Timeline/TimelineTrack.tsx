import { useRef } from 'react';
import { useTimelineStore } from '../../stores/timelineStore';
import type { Shot } from '../../stores/timelineStore';
import { motion, AnimatePresence } from 'framer-motion';
import { clsx } from 'clsx';
import { Plus, Image as ImageIcon, Film } from 'lucide-react';

const ShotBlock = ({ shot, isSelected, onSelect }: { shot: Shot, isSelected: boolean, onSelect: () => void }) => {
    return (
        <motion.div
            layout
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            className={clsx(
                "relative h-32 rounded-xl border-2 overflow-hidden cursor-pointer flex-shrink-0 group transition-colors",
                isSelected ? "border-milimo-500 shadow-milimo-500/20 shadow-lg" : "border-white/10 hover:border-white/30 bg-white/5"
            )}
            style={{ width: shot.numFrames * 2 }} // 2px per frame
            onClick={onSelect}
        >
            {/* Background / Status */}
            <div className="absolute inset-0 bg-gradient-to-br from-white/5 to-transparent pointer-events-none" />

            {/* Generating State */}
            {shot.isGenerating && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/50 backdrop-blur-sm z-20">
                    <div className="w-6 h-6 border-2 border-milimo-500 border-t-transparent rounded-full animate-spin" />
                </div>
            )}

            {/* Content Preview (Thumbnail if available, or placeholder) */}
            {shot.videoUrl ? (
                <video src={shot.videoUrl} className="w-full h-full object-cover opacity-60" muted />
            ) : (
                <div className="w-full h-full flex items-center justify-center text-xs text-white/30 font-mono p-2 text-center">
                    {shot.prompt.slice(0, 30)}...
                </div>
            )}

            {/* Conditioning Markers Overlay */}
            <div className="absolute bottom-2 left-2 right-2 flex justify-between pointer-events-none">
                {shot.timeline.map((item) => (
                    <div
                        key={item.id}
                        className="w-6 h-6 rounded bg-black/60 border border-white/20 flex items-center justify-center text-white/80"
                        style={{
                            left: `${(item.frameIndex / (shot.numFrames - 1)) * 100}%`,
                            position: 'absolute'
                        }}
                    >
                        {item.type === 'video' ? <Film size={12} /> : <ImageIcon size={12} />}
                    </div>
                ))}
            </div>

            {/* Shot Label */}
            <div className="absolute top-2 left-2 px-2 py-0.5 rounded bg-black/40 text-[10px] font-mono text-white/70 backdrop-blur-md">
                {shot.numFrames}f ({(shot.numFrames / 25).toFixed(1)}s)
            </div>
        </motion.div>
    );
};

export const TimelineTrack = () => {
    const { project, selectedShotId, selectShot, addShot } = useTimelineStore();
    const scrollRef = useRef<HTMLDivElement>(null);

    return (
        <div className="w-full h-full bg-[#0a0a0a] border-t border-white/5 flex flex-col">
            {/* Ruler / Time marker (simplified) */}
            <div className="h-6 w-full bg-white/5 border-b border-white/5 flex items-end px-4">
                <div className="text-[10px] text-white/30 font-mono">00:00</div>
                {/* Imagine ticks here */}
            </div>

            {/* Tracks Area */}
            <div
                ref={scrollRef}
                className="flex-1 overflow-x-auto overflow-y-hidden p-8 flex items-center gap-1 scrollbar-thin scrollbar-thumb-white/10 scrollbar-track-transparent"
            >
                <AnimatePresence mode='popLayout'>
                    {project.shots.map((shot) => (
                        <ShotBlock
                            key={shot.id}
                            shot={shot}
                            isSelected={selectedShotId === shot.id}
                            onSelect={() => selectShot(shot.id)}
                        />
                    ))}
                </AnimatePresence>

                {/* Add Shot Button */}
                <motion.button
                    layout
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={addShot}
                    className="h-32 w-24 rounded-xl border-2 border-dashed border-white/10 flex flex-col items-center justify-center text-white/20 hover:text-white/50 hover:border-white/30 hover:bg-white/5 transition-colors gap-2 ml-4 flex-shrink-0"
                >
                    <Plus size={24} />
                    <span className="text-xs font-medium">New Shot</span>
                </motion.button>
            </div>
        </div>
    );
};
