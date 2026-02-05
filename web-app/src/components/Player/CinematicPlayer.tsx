import { useRef, useEffect } from 'react';
import { useTimelineStore } from '../../stores/timelineStore';
import { useShallow } from 'zustand/react/shallow';
import { Play, Pause, SkipBack, SkipForward, Maximize, Brush } from 'lucide-react';
import { motion } from 'framer-motion';
import { MaskingCanvas } from '../Editor/MaskingCanvas';

export const CinematicPlayer = () => {
    const {
        isPlaying, setIsPlaying, currentTime,
        resolutionW, resolutionH, fps, seed
    } = useTimelineStore(useShallow(state => ({
        isPlaying: state.isPlaying,
        setIsPlaying: state.setIsPlaying,
        currentTime: state.currentTime,
        resolutionW: state.project.resolutionW,
        resolutionH: state.project.resolutionH,
        fps: state.project.fps,
        seed: state.project.seed
    })));

    // Program Monitor Logic: Find the top-most visible shot at currentTime
    const activeShot = useTimelineStore(useShallow(state => {
        const time = state.currentTime;
        const shots = state.project.shots;
        const fps = state.project.fps || 25;

        // 1. Check V2 (Overlay) - Priority
        const v2 = shots.find(s => {
            const track = s.trackIndex ?? 0;
            if (Number(track) !== 1) return false;
            const start = (s.startFrame || 0) / fps;
            // Use trimmed duration
            const rawDuration = (s.numFrames - (s.trimIn || 0) - (s.trimOut || 0));
            const duration = Math.max(1, rawDuration) / fps;
            return time >= start && time < (start + duration);
        });

        if (v2) {
            return v2;
        }

        // 2. Check V1 (Main) - Magnetic Logic using Dynamic Calculation
        let v1StartTime = 0;
        for (const shot of shots) {
            const track = shot.trackIndex ?? 0;
            if (Number(track) === 0) {
                const rawDuration = (shot.numFrames - (shot.trimIn || 0) - (shot.trimOut || 0));
                const duration = Math.max(1, rawDuration) / fps;

                if (time >= v1StartTime && time < (v1StartTime + duration)) {
                    // Inject calculated startFrame so seeking works correctly
                    return {
                        ...shot,
                        startFrame: v1StartTime * fps
                    };
                }
                v1StartTime += duration;
            }
        }

        return undefined;
    }));

    const videoRef = useRef<HTMLVideoElement>(null);

    const isGenerating = activeShot?.isGenerating;
    const progress = activeShot?.progress || 0;

    // Auto-load video when shot changes
    useEffect(() => {
        if (videoRef.current && activeShot?.videoUrl) {
            const currentSrc = videoRef.current.src;
            const newSrc = activeShot.videoUrl.startsWith('http')
                ? activeShot.videoUrl
                : `http://localhost:8000${activeShot.videoUrl}`;

            // Only reload if src changed to avoid stutter
            if (currentSrc !== newSrc && !currentSrc.endsWith(activeShot.videoUrl)) {
                videoRef.current.src = newSrc;
                videoRef.current.load();
                if (isPlaying) videoRef.current.play();
            }
        }
    }, [activeShot?.id, activeShot?.videoUrl]);

    // Sync play state
    useEffect(() => {
        if (!videoRef.current) return;
        if (isPlaying) {
            videoRef.current.play().catch(() => { });
        } else {
            videoRef.current.pause();
        }
    }, [isPlaying]);

    // Sync time (Seek when playhead moves)
    useEffect(() => {
        if (!videoRef.current || !activeShot) return;

        // Calculate local time for the shot
        // Local = Global - ShotStart + TrimIn
        const startTime = (activeShot.startFrame || 0) / (fps || 25);
        const trimIn = (activeShot.trimIn || 0) / (fps || 25);

        const localTime = (currentTime - startTime) + trimIn;

        // Only seek if the difference is significant
        if (Math.abs(videoRef.current.currentTime - localTime) > 0.25) {
            // Force seek even if duration represents infinity or is not ready yet
            // Browsers handle this safely (buffering if needed)
            if (localTime >= 0) {
                videoRef.current.currentTime = localTime;
            } else {
                videoRef.current.currentTime = 0;
            }
        }
    }, [currentTime, activeShot]); // activeShot included to handle cuts

    return (
        <div className="flex-1 bg-black relative flex flex-col items-center justify-center overflow-hidden group">
            {/* Main Video Surface */}
            <div className="relative aspect-video max-h-full max-w-full shadow-2xl bg-[#050505] w-full flex items-center justify-center">

                {activeShot ? (
                    activeShot.videoUrl ? (
                        (() => {
                            const isImage = activeShot.videoUrl.match(/\.(jpg|jpeg|png|webp)$/i);
                            return isImage ? (
                                <img
                                    src={activeShot.videoUrl.startsWith('http')
                                        ? activeShot.videoUrl
                                        : `http://localhost:8000${activeShot.videoUrl}`}
                                    className="w-full h-full object-contain"
                                    alt="Generated Shot"
                                />
                            ) : (
                                <video
                                    key={activeShot.id}
                                    ref={videoRef}
                                    className="w-full h-full object-contain"
                                    playsInline
                                    loop={false}
                                    muted={false} // Ensure it's not muted unless track is muted (handled by volume usually)
                                    onClick={() => setIsPlaying(!isPlaying)}
                                    onLoadedMetadata={(e) => {
                                        // Ensure we seek to correct start immediately on load
                                        const startTime = (activeShot.startFrame || 0) / (fps || 25);
                                        const trimIn = (activeShot.trimIn || 0) / (fps || 25);
                                        const t = (useTimelineStore.getState().currentTime - startTime) + trimIn;
                                        if (t > 0) e.currentTarget.currentTime = t;
                                    }}
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
                        Project Preview
                    </div>
                )}

                {/* HUD Overlay (Cinema Mode) */}
                <div className="absolute top-4 left-4 flex gap-4 text-[10px] font-mono text-white/50 bg-black/50 px-3 py-1 rounded backdrop-blur-md opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
                    <span>{resolutionW}x{resolutionH}</span>
                    <span>{fps} FPS</span>
                    <span>SEED: {activeShot?.seed || seed}</span>
                </div>

                {/* Overlay: Generation Animation - Shows ON TOP of everything if generating */}
                {isGenerating && (
                    <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/90 z-40">
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
                            <div className="flex flex-col items-center justify-center gap-2">
                                <div className="flex items-center justify-center gap-2 text-[10px] text-milimo-400 uppercase tracking-widest font-mono">
                                    <span className="opacity-80">Director Mode</span>
                                    <span className="opacity-30">|</span>
                                    <span>{progress}%</span>
                                </div>
                                {activeShot?.etaSeconds && activeShot.etaSeconds > 0 && (
                                    <div className="text-[10px] text-milimo-500/80 font-mono animate-pulse">
                                        EST. {Math.floor(activeShot.etaSeconds / 60)}:{(activeShot.etaSeconds % 60).toString().padStart(2, '0')}
                                    </div>
                                )}
                            </div>

                            {/* Job Parameters Stats */}
                            <div className="grid grid-cols-2 gap-x-8 gap-y-1 mt-6 text-[9px] font-mono text-white/30 uppercase tracking-widest text-left border-t border-white/5 pt-4">
                                <div className="flex justify-between gap-4"><span>Res</span> <span className="text-white/60">{activeShot.width}x{activeShot.height}</span></div>
                                <div className="flex justify-between gap-4"><span>FPS</span> <span className="text-white/60">{activeShot.fps || fps}</span></div>
                                <div className="flex justify-between gap-4"><span>Frames</span> <span className="text-white/60">{activeShot.numFrames}</span></div>
                                <div className="flex justify-between gap-4"><span>Seed</span> <span className="text-white/60">{activeShot.seed}</span></div>
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

                {/* Edit Button */}
                <button
                    onClick={() => {
                        const { isEditing, setEditing } = useTimelineStore.getState();
                        setEditing(!isEditing);
                    }}
                    className={`transition-colors ${useTimelineStore(state => state.isEditing) ? 'text-milimo-500' : 'text-white/50 hover:text-white'}`}
                    title="Toggle Edit Mode"
                >
                    <Brush size={18} />
                </button>

                <button className="text-white/50 hover:text-white transition-colors">
                    <Maximize size={18} />
                </button>
            </motion.div>

            {/* Masking Overlay */}
            {useTimelineStore(state => state.isEditing) && videoRef.current && activeShot && (
                <MaskingCanvas
                    width={videoRef.current.videoWidth || 1280}
                    height={videoRef.current.videoHeight || 720}
                    onCancel={() => useTimelineStore.getState().setEditing(false)}
                    onSave={async (maskDataUrl) => {
                        if (!activeShot || !videoRef.current) return;

                        // 1. Capture current frame
                        const canvas = document.createElement('canvas');
                        canvas.width = videoRef.current.videoWidth || 1280;
                        canvas.height = videoRef.current.videoHeight || 720;
                        const ctx = canvas.getContext('2d');
                        if (ctx) {
                            ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
                        }
                        const frameDataUrl = canvas.toDataURL('image/jpeg', 0.95);

                        // 2. Get Prompt - Use active shot logic
                        const prompt = window.prompt("Describe what to generate in the masked area:", activeShot.prompt);
                        if (!prompt) return;

                        // 3. Trigger Store Action
                        await useTimelineStore.getState().inpaintShot(activeShot.id, frameDataUrl, maskDataUrl, prompt);

                        useTimelineStore.getState().setEditing(false);
                    }}
                />
            )}
        </div>
    );
};
