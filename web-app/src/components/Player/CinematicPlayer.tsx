import { useRef, useEffect } from 'react';
import { useTimelineStore } from '../../stores/timelineStore';
import { Play, Pause, SkipBack, SkipForward, Maximize } from 'lucide-react';
import { motion } from 'framer-motion';

export const CinematicPlayer = () => {
    const project = useTimelineStore(state => state.project);
    const selectedShotId = useTimelineStore(state => state.selectedShotId);
    const isPlaying = useTimelineStore(state => state.isPlaying);
    const setIsPlaying = useTimelineStore(state => state.setIsPlaying);
    const setCurrentTime = useTimelineStore(state => state.setCurrentTime);

    const videoRef = useRef<HTMLVideoElement>(null);

    const selectedShot = project.shots.find(s => s.id === selectedShotId);
    const isGenerating = selectedShot?.isGenerating;
    const progress = selectedShot?.progress || 0;

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
            <div className="relative aspect-video max-h-full max-w-full shadow-2xl bg-[#050505] w-full flex items-center justify-center">

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
                            <div className="w-16 h-16 rounded-full border border-white/10 flex items-center justify-center">
                                <Play size={24} className="ml-1 opacity-50" />
                            </div>
                            <span className="text-xs font-mono">Ready to Generate</span>
                        </div>
                    )
                ) : (
                    <div className="absolute inset-0 flex items-center justify-center text-white/10 text-sm font-mono uppercase tracking-widest">
                        No Shot Selected
                    </div>
                )}

                {/* HUD Overlay (Cinema Mode) */}
                <div className="absolute top-4 left-4 flex gap-4 text-[10px] font-mono text-white/50 bg-black/50 px-3 py-1 rounded backdrop-blur-md opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
                    <span>{project.resolutionW}x{project.resolutionH}</span>
                    <span>{project.fps} FPS</span>
                    <span>SEED: {selectedShot?.seed || project.seed}</span>
                </div>

                {/* Overlay: Generation Animation - Shows ON TOP of everything if generating */}
                {isGenerating && (
                    <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/90 z-50">
                        {/* Background Ambient Glow */}
                        <div className="absolute w-full h-full opacity-20">
                            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[500px] bg-milimo-500/30 rounded-full blur-[100px] animate-pulse" />
                        </div>

                        {/* Central Cinematic Loader */}
                        <div className="relative w-24 h-24 mb-8">
                            {/* Rotating Rings */}
                            <motion.div
                                animate={{ rotate: 360 }}
                                transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
                                className="absolute inset-0 border-2 border-t-milimo-400 border-r-transparent border-b-milimo-600/50 border-l-transparent rounded-full"
                            />
                            <motion.div
                                animate={{ rotate: -360 }}
                                transition={{ duration: 5, repeat: Infinity, ease: "linear" }}
                                className="absolute inset-2 border-2 border-t-transparent border-r-milimo-500/50 border-b-transparent border-l-milimo-300 rounded-full"
                            />

                            {/* Core Core */}
                            <motion.div
                                animate={{ scale: [0.8, 1.2, 0.8], opacity: [0.5, 1, 0.5] }}
                                transition={{ duration: 2, repeat: Infinity }}
                                className="absolute inset-8 bg-milimo-500 rounded-full blur-md"
                            />
                            <div className="absolute inset-8 bg-white rounded-full mix-blend-overlay" />
                        </div>

                        {/* Text Animation */}
                        <div className="text-center z-10">
                            <h3 className="text-xl font-bold tracking-[0.2em] text-white uppercase mb-2 animate-pulse">
                                Generating
                            </h3>
                            <div className="flex items-center justify-center gap-2 text-[10px] text-milimo-400 uppercase tracking-widest font-mono">
                                <span className="opacity-80">Director Mode</span>
                                <span className="opacity-30">|</span>
                                <span>{progress}%</span>
                            </div>
                        </div>

                        {/* Scanline Effect */}
                        <div className="absolute inset-0 pointer-events-none bg-[linear-gradient(rgba(18,16,16,0)_50%,rgba(0,0,0,0.25)_50%),linear-gradient(90deg,rgba(255,0,0,0.06),rgba(0,255,0,0.02),rgba(0,0,255,0.06))] z-20 bg-[length:100%_2px,3px_100%] opacity-20" />
                    </div>
                )}
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
