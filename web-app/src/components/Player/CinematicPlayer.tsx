import { useRef, useEffect, useMemo, useState, memo, forwardRef } from 'react';
import { useTimelineStore } from '../../stores/timelineStore';
import { useShallow } from 'zustand/react/shallow';
import { Play, Pause, SkipBack, SkipForward, Maximize, Brush, Crosshair } from 'lucide-react';
import { motion } from 'framer-motion';
import { MaskingCanvas } from '../Editor/MaskingCanvas';
import { TrackingPanel } from '../Editor/TrackingPanel';
import { computeTimelineLayout } from '../../utils/timelineUtils';

// --- Sub-Components ---

// 1. Video Surface - Only re-renders when the generic "Shot" identity or URL changes.
// It manages its own play/pause state and drift correction via direct store subscription.
const VideoSurface = memo(forwardRef<HTMLVideoElement, { shot: any; isPlaying: boolean; fps: number }>(({ shot, isPlaying, fps }, ref) => {
    // Internal ref for local logic
    const internalRef = useRef<HTMLVideoElement>(null);
    const lastDriftCheckRef = useRef<number>(0);
    const userInteractedRef = useRef<boolean>(false);

    // Merge forwarded ref with internal ref
    const setRef = (el: HTMLVideoElement | null) => {
        internalRef.current = el;
        if (typeof ref === 'function') ref(el);
        else if (ref) (ref as any).current = el;
    };

    // Safari Autoplay Policy: Start muted, unmute on first user interaction
    useEffect(() => {
        const handleUserGesture = () => {
            userInteractedRef.current = true;
            if (internalRef.current && internalRef.current.muted) {
                internalRef.current.muted = false;
            }
            // Clean up after first interaction
            document.removeEventListener('click', handleUserGesture);
            document.removeEventListener('keydown', handleUserGesture);
            document.removeEventListener('touchstart', handleUserGesture);
        };

        document.addEventListener('click', handleUserGesture);
        document.addEventListener('keydown', handleUserGesture);
        document.addEventListener('touchstart', handleUserGesture);

        return () => {
            document.removeEventListener('click', handleUserGesture);
            document.removeEventListener('keydown', handleUserGesture);
            document.removeEventListener('touchstart', handleUserGesture);
        };
    }, []);

    // Auto-load
    useEffect(() => {
        if (internalRef.current && shot?.videoUrl) {
            const currentSrc = internalRef.current.src;
            const newSrc = shot.videoUrl.startsWith('http') ? shot.videoUrl : `http://localhost:8000${shot.videoUrl}`;
            if (currentSrc !== newSrc && !currentSrc.endsWith(shot.videoUrl)) {
                internalRef.current.src = newSrc;
                internalRef.current.load();
                if (isPlaying) {
                    internalRef.current.play().catch((e) => {
                        // Safari: If autoplay blocked, ensure muted and retry
                        if (e.name === 'NotAllowedError' && internalRef.current) {
                            internalRef.current.muted = true;
                            internalRef.current.play().catch(() => { });
                        }
                    });
                }
            }
        }
    }, [shot?.id, shot?.videoUrl]);

    // Sync Play State
    useEffect(() => {
        if (!internalRef.current) return;
        if (isPlaying) {
            internalRef.current.play().catch((e) => {
                // Safari: If autoplay is blocked, retry muted
                if (e.name === 'NotAllowedError' && internalRef.current) {
                    internalRef.current.muted = true;
                    internalRef.current.play().catch(() => { });
                }
            });
        } else {
            internalRef.current.pause();
        }
    }, [isPlaying]);

    // Drift Correction (Headless Subscription) — Throttled for Safari performance
    useEffect(() => {
        const unsub = useTimelineStore.subscribe((state) => {
            if (!internalRef.current || !shot) return;

            // Throttle: Only check drift every 250ms to avoid overwhelming Safari's decoder
            const now = performance.now();
            if (now - lastDriftCheckRef.current < 250) return;
            lastDriftCheckRef.current = now;

            const currentTime = state.currentTime;

            // Calculate local time
            const startTime = (shot.startFrame || 0) / fps;
            const trimIn = (shot.trimIn || 0) / fps;
            const localTime = (currentTime - startTime) + trimIn;

            // Drift Check (0.3s tolerance — relaxed for Safari)
            const drift = Math.abs(internalRef.current.currentTime - localTime);
            if (drift > 0.3) {
                const seekTo = Math.max(0, localTime);
                // Use fastSeek when available (Safari supports it, cheaper than .currentTime)
                if ('fastSeek' in internalRef.current && typeof internalRef.current.fastSeek === 'function') {
                    internalRef.current.fastSeek(seekTo);
                } else {
                    internalRef.current.currentTime = seekTo;
                }
            }
        });
        return unsub;
    }, [shot, fps]); // Re-subscribe if shot changes

    if (!shot || !shot.videoUrl) return null;

    const isImage = shot.videoUrl.match(/\.(jpg|jpeg|png|webp)$/i);

    if (isImage) {
        return (
            <img
                src={shot.videoUrl.startsWith('http') ? shot.videoUrl : `http://localhost:8000${shot.videoUrl}`}
                className="w-full h-full object-contain"
                crossOrigin="anonymous"
                alt="Generated Shot"
            />
        );
    }

    return (
        <video
            ref={setRef}
            className="w-full h-full object-contain"
            crossOrigin="anonymous"
            playsInline
            loop={false}
            muted={true}
            preload="metadata"
            onClick={() => useTimelineStore.getState().setIsPlaying(!isPlaying)}
        />
    );
}));

// 2. HUD - Re-renders only when metadata changes
const PlayerHUD = memo(({ resolutionW, resolutionH, fps, seed }: any) => (
    <div className="absolute top-4 left-4 flex gap-4 text-[10px] font-mono text-white/50 bg-black/50 px-3 py-1 rounded backdrop-blur-md opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
        <span>{resolutionW}x{resolutionH}</span>
        <span>{fps} FPS</span>
        <span>SEED: {seed}</span>
    </div>
));

// 3. Loading Overlay - Re-renders when generating state changes
const LoadingOverlay = memo(({ isGenerating, progress, etaSeconds, width, height, fps, numFrames, seed }: any) => {
    if (!isGenerating) return null;
    return (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/90 z-40">
            {/* Background Ambient Glow */}
            <div className="absolute w-full h-full opacity-20">
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[500px] bg-milimo-500/30 rounded-full blur-[100px] animate-pulse" />
            </div>

            {/* Central Cinematic Loader */}
            <div className="relative w-24 h-24 mb-8">
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
                    {etaSeconds && etaSeconds > 0 && (
                        <div className="text-[10px] text-milimo-500/80 font-mono animate-pulse">
                            EST. {Math.floor(etaSeconds / 60)}:{(etaSeconds % 60).toString().padStart(2, '0')}
                        </div>
                    )}
                </div>

                {/* Job Parameters Stats */}
                <div className="grid grid-cols-2 gap-x-8 gap-y-1 mt-6 text-[9px] font-mono text-white/30 uppercase tracking-widest text-left border-t border-white/5 pt-4">
                    <div className="flex justify-between gap-4"><span>Res</span> <span className="text-white/60">{width}x{height}</span></div>
                    <div className="flex justify-between gap-4"><span>FPS</span> <span className="text-white/60">{fps}</span></div>
                    <div className="flex justify-between gap-4"><span>Frames</span> <span className="text-white/60">{numFrames}</span></div>
                    <div className="flex justify-between gap-4"><span>Seed</span> <span className="text-white/60">{seed}</span></div>
                </div>
            </div>
            {/* Scanline Effect */}
            <div className="absolute inset-0 pointer-events-none bg-[linear-gradient(rgba(18,16,16,0)_50%,rgba(0,0,0,0.25)_50%),linear-gradient(90deg,rgba(255,0,0,0.06),rgba(0,255,0,0.02),rgba(0,0,255,0.06))] z-20 bg-[length:100%_2px,3px_100%] opacity-20" />
        </div>
    );
});

// 4. Controls Bar
const ControlsBar = memo(({ isPlaying, isEditing, isTracking, onTogglePlay, onToggleEdit, onToggleTracking }: any) => (
    <motion.div
        initial={{ y: 50, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        className="absolute bottom-6 left-1/2 -translate-x-1/2 bg-black/60 backdrop-blur-md border border-white/10 rounded-full px-6 py-3 flex items-center gap-6 shadow-2xl z-30"
    >
        <button className="text-white/50 hover:text-white transition-colors">
            <SkipBack size={20} />
        </button>

        <button
            onClick={onTogglePlay}
            className="w-12 h-12 rounded-full bg-white text-black flex items-center justify-center hover:bg-milimo-400 transition-colors shadow-lg shadow-white/10"
        >
            {isPlaying ? <Pause size={20} fill="currentColor" /> : <Play size={20} fill="currentColor" className="ml-1" />}
        </button>

        <button className="text-white/50 hover:text-white transition-colors">
            <SkipForward size={20} />
        </button>

        <div className="w-px h-6 bg-white/10 mx-2" />

        <button
            onClick={onToggleEdit}
            className={`transition-colors ${isEditing ? 'text-milimo-500' : 'text-white/50 hover:text-white'}`}
            title="Toggle Edit Mode (Inpainting)"
        >
            <Brush size={18} />
        </button>

        <button
            onClick={onToggleTracking}
            className={`transition-colors ${isTracking ? 'text-cyan-400' : 'text-white/50 hover:text-white'}`}
            title="Toggle Object Tracking"
        >
            <Crosshair size={18} />
        </button>

        <button className="text-white/50 hover:text-white transition-colors">
            <Maximize size={18} />
        </button>
    </motion.div>
));

export const CinematicPlayer = () => {
    // 1. Stable Store Selections
    const { isPlaying, setIsPlaying, isEditing, setEditing, resolutionW, resolutionH, fps, seed, shots } = useTimelineStore(useShallow(state => ({
        isPlaying: state.isPlaying,
        setIsPlaying: state.setIsPlaying,
        isEditing: state.isEditing,
        setEditing: state.setEditing,
        resolutionW: state.project.resolutionW,
        resolutionH: state.project.resolutionH,
        fps: state.project.fps || 25,
        seed: state.project.seed,
        shots: state.project.shots
    })));

    // Tracking mode state (local — not in store since it's view-only)
    const [isTracking, setIsTracking] = useState(false);

    // 2. Compute Derived Schedule (Memoized)
    // This creates a flat list of visible regions for all tracks, prioritized.
    // Ideally matches VisualTimeline logic.
    const schedule = useMemo(() => {
        const clips = computeTimelineLayout({ shots, fps });
        return clips.map(c => ({
            start: c.start,
            end: c.end,
            priority: c.track,
            shot: { ...c.shot, startFrame: c.start * fps } // Injection for VideoSurface
        })).sort((a, b) => a.start - b.start);
    }, [shots, fps]);

    // 3. Active Shot State - Updates via subscription to avoid parent re-renders loop?
    // Actually, we need the active shot to render the video.
    // If we use `useTimelineStore.subscribe` to update a local state `activeShot`, 
    // it will still trigger re-render of this component. 
    // BUT checking logic is cheaper now.

    // However, we want to AVOID re-rendering the whole tree if `activeShot` hasn't changed ID.
    const [activeShot, setActiveShot] = useState<any>(null);

    useEffect(() => {
        // Subscribe to currentTime to update activeShot efficiently
        const unsub = useTimelineStore.subscribe((state) => {
            const time = state.currentTime;

            // Find matching clip in schedule
            // Priority: V2 (1) > V1 (0)
            let found = null;

            // Check V2 first
            const v2 = schedule.find(item => item.priority === 1 && time >= item.start && time < item.end);
            if (v2) found = v2.shot;
            else {
                const v1 = schedule.find(item => item.priority === 0 && time >= item.start && time < item.end);
                if (v1) found = v1.shot;
            }

            // Update state only if changed
            setActiveShot((prev: any) => {
                if (!prev && !found) return null;
                if (prev && found
                    && prev.id === found.id
                    && prev.videoUrl === found.videoUrl
                    && prev.thumbnailUrl === found.thumbnailUrl
                    && prev.isGenerating === found.isGenerating
                ) return prev; // Stable reference — only skip if content matches too
                return found;
            });
        });
        return unsub;
    }, [schedule]);

    // --- Render ---

    // Derived values for HUD/Overlay
    const isGenerating = activeShot?.isGenerating;

    // We need a ref to the video element for masking.
    const videoRef = useRef<HTMLVideoElement>(null);

    return (
        <div className="flex-1 bg-black relative flex flex-col items-center justify-center overflow-hidden group">
            {/* Main Video Surface */}
            <div className="relative aspect-video max-h-full max-w-full shadow-2xl bg-[#050505] w-full flex items-center justify-center">
                {activeShot ? (
                    activeShot.videoUrl ? (
                        <VideoSurface
                            ref={videoRef}
                            shot={activeShot}
                            isPlaying={isPlaying}
                            fps={fps}
                        />
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

                <PlayerHUD
                    resolutionW={resolutionW}
                    resolutionH={resolutionH}
                    fps={fps}
                    seed={activeShot?.seed || seed}
                />

                {activeShot && (
                    <LoadingOverlay
                        isGenerating={isGenerating}
                        progress={activeShot.progress || 0}
                        etaSeconds={activeShot.etaSeconds}
                        width={activeShot.width || resolutionW}
                        height={activeShot.height || resolutionH}
                        fps={activeShot.fps || fps}
                        numFrames={activeShot.numFrames}
                        seed={activeShot.seed}
                    />
                )}
                {/* Masking Overlay */}
                {isEditing && activeShot && activeShot.videoUrl && (
                    <MaskingCanvas
                        width={resolutionW}
                        height={resolutionH}
                        videoRef={videoRef}
                        onCancel={() => setEditing(false)}
                        onSave={async (maskDataUrl) => {
                            if (!videoRef.current) return;
                            const videoEl = videoRef.current;

                            const canvas = document.createElement('canvas');
                            canvas.width = videoEl.videoWidth;
                            canvas.height = videoEl.videoHeight;
                            const ctx = canvas.getContext('2d');
                            if (ctx) ctx.drawImage(videoEl, 0, 0);
                            const frameDataUrl = canvas.toDataURL('image/jpeg', 0.95);

                            const prompt = window.prompt("Describe what to generate in the masked area:", activeShot.prompt);
                            if (!prompt) return;

                            await useTimelineStore.getState().inpaintShot(activeShot.id, frameDataUrl, maskDataUrl, prompt);
                            setEditing(false);
                        }}
                    />
                )}

                {/* Tracking Overlay */}
                {isTracking && activeShot && activeShot.videoUrl && (
                    <TrackingPanel
                        videoPath={activeShot.videoUrl}
                        videoRef={videoRef}
                        containerWidth={resolutionW}
                        containerHeight={resolutionH}
                        onClose={() => setIsTracking(false)}
                    />
                )}
            </div>

            <ControlsBar
                isPlaying={isPlaying}
                isEditing={isEditing}
                isTracking={isTracking}
                onTogglePlay={() => setIsPlaying(!isPlaying)}
                onToggleEdit={() => { setEditing(!isEditing); if (!isEditing) setIsTracking(false); }}
                onToggleTracking={() => { setIsTracking(!isTracking); if (!isTracking) setEditing(false); }}
            />
        </div>
    );
};
