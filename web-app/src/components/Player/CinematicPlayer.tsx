import { useRef, useEffect } from 'react';
import { useTimelineStore } from '../../stores/timelineStore';
import { Play, Pause, SkipBack, SkipForward, Maximize, RefreshCw } from 'lucide-react';
import { motion } from 'framer-motion';

export const CinematicPlayer = () => {
    const {
        project, selectedShotId,
        isPlaying, setIsPlaying,
        setCurrentTime
    } = useTimelineStore();

    const videoRef = useRef<HTMLVideoElement>(null);

    const selectedShot = project.shots.find(s => s.id === selectedShotId);

    // Auto-load video when shot changes
    useEffect(() => {
        if (videoRef.current && selectedShot?.videoUrl) {
            videoRef.current.src = selectedShot.videoUrl.startsWith('http')
                ? selectedShot.videoUrl
                : `http://localhost:8000${selectedShot.videoUrl}`;
            videoRef.current.load();
            if (isPlaying) videoRef.current.play();
        }
    }, [selectedShot?.videoUrl]);

    // Sync play state
    useEffect(() => {
        if (!videoRef.current) return;
        if (isPlaying) videoRef.current.play();
        else videoRef.current.pause();
    }, [isPlaying]);

    return (
        <div className="flex-1 bg-black relative flex flex-col items-center justify-center overflow-hidden group">
            {/* Main Video Surface */}
            <div className="relative aspect-video max-h-full max-w-full shadow-2xl bg-[#050505]">
                {selectedShot ? (
                    selectedShot.videoUrl ? (
                        (() => {
                            const isImage = selectedShot.videoUrl.match(/\.(jpg|jpeg|png|webp)$/i);
                            return isImage ? (
                                <img
                                    src={selectedShot.videoUrl.startsWith('http')
                                        ? selectedShot.videoUrl
                                        : `http://localhost:8000${selectedShot.videoUrl}`}
                                    className="w-full h-full object-contain"
                                    alt="Generated Shot"
                                />
                            ) : (
                                <video
                                    ref={videoRef}
                                    className="w-full h-full object-contain"
                                    playsInline
                                    onTimeUpdate={(e) => {
                                        const startTime = useTimelineStore.getState().getShotStartTime(selectedShot.id);
                                        setCurrentTime(startTime + e.currentTarget.currentTime);
                                    }}
                                    onEnded={() => {
                                        const { project, selectShot, setIsPlaying } = useTimelineStore.getState();
                                        const currentIndex = project.shots.findIndex(s => s.id === selectedShot.id);
                                        if (currentIndex < project.shots.length - 1) {
                                            // Play next
                                            selectShot(project.shots[currentIndex + 1].id);
                                        } else {
                                            setIsPlaying(false);
                                        }
                                    }}
                                    onClick={() => setIsPlaying(!isPlaying)}
                                />
                            );
                        })()
                    ) : (
                        <div className="absolute inset-0 flex flex-col items-center justify-center text-white/20 gap-4">
                            {selectedShot.isGenerating ? (
                                <div className="animate-pulse flex flex-col items-center">
                                    <RefreshCw className="animate-spin mb-2" size={32} />
                                    <span className="text-sm font-mono tracking-widest uppercase">
                                        {selectedShot.numFrames === 1 ? "Rendering Milimo Image..." : "Rendering Milimo Video..."}
                                    </span>
                                </div>
                            ) : (
                                <>
                                    <div className="w-16 h-16 rounded-full border border-white/10 flex items-center justify-center">
                                        <Play size={24} className="ml-1 opacity-50" />
                                    </div>
                                    <span className="text-xs font-mono">Ready to Generate</span>
                                </>
                            )}
                        </div>
                    )
                ) : (
                    <div className="absolute inset-0 flex items-center justify-center text-white/10 text-sm font-mono uppercase tracking-widest">
                        No Shot Selected
                    </div>
                )}

                {/* HUD Overlay (Cinema Mode) */}
                <div className="absolute top-4 left-4 flex gap-4 text-[10px] font-mono text-white/50 bg-black/50 px-3 py-1 rounded backdrop-blur-md opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
                    <span>{project.resolutionW}x{project.resolutionH}</span>
                    <span>{project.fps} FPS</span>
                    <span>SEED: {selectedShot?.seed || project.seed}</span>
                </div>
            </div>

            {/* Controls Bar (Floating) */}
            <motion.div
                initial={{ y: 50, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                className="absolute bottom-6 left-1/2 -translate-x-1/2 bg-black/60 backdrop-blur-md border border-white/10 rounded-full px-6 py-3 flex items-center gap-6 shadow-2xl z-20"
            >
                <button className="text-white/50 hover:text-white transition-colors">
                    <SkipBack size={20} />
                </button>

                <button
                    onClick={() => setIsPlaying(!isPlaying)}
                    className="w-12 h-12 rounded-full bg-white text-black flex items-center justify-center hover:bg-milimo-400 transition-colors shadow-lg shadow-white/10"
                >
                    {isPlaying ? <Pause size={20} fill="currentColor" /> : <Play size={20} fill="currentColor" className="ml-1" />}
                </button>

                <button className="text-white/50 hover:text-white transition-colors">
                    <SkipForward size={20} />
                </button>

                <div className="w-px h-6 bg-white/10 mx-2" />

                <button className="text-white/50 hover:text-white transition-colors">
                    <Maximize size={18} />
                </button>
            </motion.div>
        </div>
    );
};
