import { useState, useCallback, useRef, useEffect, useMemo, memo } from 'react';
import { motion } from 'framer-motion';
import { X, Crosshair, Play, Square, RotateCcw, ChevronLeft, ChevronRight, Loader2, Download, Trash2, Eye, EyeOff, ChevronsLeft, ChevronsRight } from 'lucide-react';
import { API_BASE_URL } from '../../config';

// ── Types ──

interface TrackingObject {
    id: number;
    label: string;
    color: string;
    visible: boolean;
}

interface FrameResult {
    frameIndex: number;
    masks: Record<string, string>; // obj_id -> base64 mask PNG
    scores: Record<string, number>;
}

interface TrackingPanelProps {
    videoPath: string;          // Backend path to the video file
    videoRef: React.RefObject<HTMLVideoElement | null>;
    containerWidth: number;
    containerHeight: number;
    onClose: () => void;
}

// ── Color Palette ──

const TRACK_COLORS = [
    '#6366f1', '#ec4899', '#22c55e', '#f59e0b',
    '#06b6d4', '#a855f7', '#ef4444', '#14b8a6',
];

// ── Main Component ──

export const TrackingPanel = memo(({
    videoPath,
    videoRef: _videoRef,
    containerWidth,
    containerHeight,
    onClose,
}: TrackingPanelProps) => {
    // ── Session State ──
    const [sessionId, setSessionId] = useState<string | null>(null);
    const [phase, setPhase] = useState<'idle' | 'starting' | 'prompting' | 'propagating' | 'done' | 'error'>('idle');
    const [isPropagating, setIsPropagating] = useState(false);
    const [isExporting, setIsExporting] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [statusMsg, setStatusMsg] = useState('');

    // ── Prompt State ──
    const [promptMode, setPromptMode] = useState<'text' | 'click'>('text');
    const [textPrompt, setTextPrompt] = useState('');
    const [frameIndex, setFrameIndex] = useState(0);
    const [direction, setDirection] = useState<'forward' | 'backward' | 'both'>('both');

    // ── Results ──
    const [objects, setObjects] = useState<TrackingObject[]>([]);
    const [frameResults, setFrameResults] = useState<FrameResult[]>([]);
    const [viewingFrame, setViewingFrame] = useState(0);
    const [fps, setFps] = useState(30.0);
    const [totalFrames, setTotalFrames] = useState(0);
    const [keyframeIndices, setKeyframeIndices] = useState<Set<number>>(new Set());

    // ── Mask Overlay Canvas ──
    const overlayRef = useRef<HTMLCanvasElement>(null);
    const rafRef = useRef<number | null>(null);

    // ── API Helpers ──
    const apiCall = useCallback(async (endpoint: string, body: any) => {
        const res = await fetch(`${API_BASE_URL}${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        if (!res.ok) {
            const errText = await res.text();
            throw new Error(errText || `API error ${res.status}`);
        }
        return res.json();
    }, []);

    // ── Load Existing Session (Persistence) ──
    const loadSession = useCallback(async (sessId: string) => {
        try {
            setStatusMsg('Loading saved tracking data...');
            const result = await apiCall('/edit/track/load', { session_id: sessId, video_path: videoPath });

            if (result.status === 'loaded' && result.frames && result.frames.length > 0) {
                // Map backend field names (frame_idx) to frontend (frameIndex)
                const loadedFrames: FrameResult[] = result.frames.map((f: any) => ({
                    frameIndex: f.frameIndex ?? f.frame_idx ?? f.frame_index ?? 0,
                    masks: f.masks || {},
                    scores: f.scores || {},
                }));
                setFrameResults(loadedFrames);

                // Restore Objects from metadata (if available) or infer
                const metadata = result.objects || {};
                const uniqueObjects = new Map<number, TrackingObject>();

                // 1. Populate from metadata first
                Object.keys(metadata).forEach(objIdStr => {
                    const id = parseInt(objIdStr);
                    const meta = metadata[objIdStr];
                    uniqueObjects.set(id, {
                        id,
                        label: meta.label || `Object ${id}`,
                        color: meta.color || TRACK_COLORS[id % TRACK_COLORS.length],
                        visible: true
                    });
                });

                // 2. Discover any additional objects from frames that weren't in metadata
                loadedFrames.forEach(f => {
                    Object.keys(f.masks).forEach((objIdStr) => {
                        const id = parseInt(objIdStr);
                        if (!uniqueObjects.has(id)) {
                            uniqueObjects.set(id, {
                                id,
                                label: `Object ${id}`,
                                color: TRACK_COLORS[id % TRACK_COLORS.length],
                                visible: true
                            });
                        }
                    });
                });

                setObjects(Array.from(uniqueObjects.values()));
                setPhase('done');
                setStatusMsg(`Loaded ${loadedFrames.length} frames.`);
            } else {
                setStatusMsg('Ready (no saved data found).');
                setTimeout(() => setStatusMsg(''), 2000);
            }
        } catch (e) {
            console.warn('Failed to load session:', e);
            setStatusMsg('');
        }
    }, [apiCall, videoPath]);

    // ── Start Session ──
    const startSession = useCallback(async () => {
        setPhase('starting');
        setError(null);
        setStatusMsg('Starting tracking session...');
        try {
            const result = await apiCall('/track/start', { video_path: videoPath });
            if (result.error) throw new Error(result.error);
            setSessionId(result.session_id);
            if (result.fps) setFps(result.fps);
            if (result.num_frames) setTotalFrames(result.num_frames);
            setPhase('prompting');
            setStatusMsg('Session started. Add a prompt to begin tracking.');

            // Attempt to load existing data for this session
            if (result.session_id) {
                loadSession(result.session_id);
            }
        } catch (e: any) {
            setError(e.message);
            setPhase('error');
            setStatusMsg('');
        }
    }, [videoPath, apiCall, loadSession]);

    // ── Add Prompt (Text) ──
    const addTextPrompt = useCallback(async () => {
        if (!sessionId || !textPrompt.trim()) return;
        setPhase('prompting');
        setStatusMsg(`Detecting "${textPrompt}"...`);
        try {
            const result = await apiCall('/track/prompt', {
                session_id: sessionId,
                frame_idx: frameIndex,
                text: textPrompt.trim(),
            });
            if (result.error) throw new Error(result.error);

            // Parse detected objects from response
            const newObjects: TrackingObject[] = (result.object_ids || []).map((id: number, idx: number) => ({
                id,
                label: textPrompt.trim(),
                color: TRACK_COLORS[idx % TRACK_COLORS.length],
                visible: true,
            }));

            setObjects(prev => {
                // Filter out duplicates if any
                const existingIds = new Set(prev.map(o => o.id));
                const uniqueNew = newObjects.filter(o => !existingIds.has(o.id));
                return [...prev, ...uniqueNew];
            });

            // Mark this frame as a keyframe
            setKeyframeIndices(prev => new Set(prev).add(frameIndex));

            // Store initial frame result
            if (result.masks) {
                setFrameResults(prev => {
                    const exists = prev.findIndex(f => f.frameIndex === frameIndex);
                    const newItem = {
                        frameIndex,
                        masks: result.masks,
                        scores: result.scores || {},
                    };
                    if (exists >= 0) {
                        const copy = [...prev];
                        copy[exists] = newItem;
                        return copy;
                    }
                    return [...prev, newItem].sort((a, b) => a.frameIndex - b.frameIndex);
                });
            }

            setStatusMsg(`Detected ${newObjects.length} object(s). Ready to propagate.`);
        } catch (e: any) {
            setError(e.message);
            setStatusMsg('Prompt failed');
        }
    }, [sessionId, textPrompt, frameIndex, apiCall]);

    // ── Add Point Click Prompt ──
    const handleCanvasClick = useCallback(async (e: React.MouseEvent<HTMLCanvasElement>) => {
        if (!sessionId || promptMode !== 'click' || phase !== 'prompting') return;

        const canvas = overlayRef.current;
        if (!canvas) return;
        const rect = canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left) / rect.width;
        const y = (e.clientY - rect.top) / rect.height;

        setStatusMsg('Adding point prompt...');
        try {
            // When adding a point, we might be refining an existing object OR adding a new one.
            // For now, let's assume adding a new object unless user selected one (selection logic not here yet).
            // Pass obj_id=null to let SAM create a new ID, or we could track selectedObjectId.

            const result = await apiCall('/track/prompt', {
                session_id: sessionId,
                frame_idx: frameIndex,
                points: [[x, y]],
                point_labels: [1], // positive click
                // obj_id: selectedObjectId // Todo: Add object selection to refine
            });
            if (result.error) throw new Error(result.error);

            // SAM might return multiple IDs if it detected multiple things or we sent multiple points
            const newObjectIds: number[] = result.object_ids || [];

            setObjects(prev => {
                const existingIds = new Set(prev.map(o => o.id));
                const newObjs = newObjectIds
                    .filter(id => !existingIds.has(id))
                    .map((id, idx) => ({
                        id,
                        label: `Object ${id}`,
                        color: TRACK_COLORS[(prev.length + idx) % TRACK_COLORS.length],
                        visible: true,
                    }));
                return [...prev, ...newObjs];
            });

            // Mark this frame as a keyframe
            setKeyframeIndices(prev => new Set(prev).add(frameIndex));

            // Update frame results for immediate feedback
            if (result.masks) {
                setFrameResults(prev => {
                    const exists = prev.findIndex(f => f.frameIndex === frameIndex);
                    // Merge new masks with existing ones for this frame if any
                    let mergedMasks = result.masks;
                    let mergedScores = result.scores || {};

                    if (exists >= 0) {
                        mergedMasks = { ...prev[exists].masks, ...result.masks };
                        mergedScores = { ...prev[exists].scores, ...(result.scores || {}) };
                    }

                    const newItem = {
                        frameIndex,
                        masks: mergedMasks,
                        scores: mergedScores,
                    };

                    if (exists >= 0) {
                        const copy = [...prev];
                        copy[exists] = newItem;
                        return copy;
                    }
                    return [...prev, newItem].sort((a, b) => a.frameIndex - b.frameIndex);
                });
            }

            setStatusMsg(`Point detected. Object(s): ${newObjectIds.join(', ')}`);
        } catch (e: any) {
            setError(e.message);
        }
    }, [sessionId, promptMode, phase, frameIndex, apiCall]);

    // ── Propagate ──
    const propagate = useCallback(async () => {
        if (!sessionId) return;
        setPhase('propagating');
        setIsPropagating(true);
        setStatusMsg('Propagating masks across frames...');
        try {
            const result = await apiCall('/track/propagate', {
                session_id: sessionId,
                direction,
                start_frame: frameIndex,
                max_frames: -1,
            });
            if (result.error) throw new Error(result.error);

            // Parse frame results — map backend field names to frontend
            const frames: FrameResult[] = (result.frames || []).map((f: any) => ({
                frameIndex: f.frameIndex ?? f.frame_idx ?? f.frame_index ?? 0,
                masks: f.masks || {},
                scores: f.scores || {},
            }));
            setFrameResults(frames);
            if (frames.length > 0) {
                const maxFrame = Math.max(...frames.map(f => f.frameIndex));
                if (maxFrame + 1 > totalFrames) setTotalFrames(maxFrame + 1);
            }
            setPhase('done');
            setStatusMsg(`Tracking complete — ${frames.length} frames processed.`);
        } catch (e: any) {
            setError(e.message);
            setPhase('error');
            setStatusMsg('Propagation failed');
        } finally {
            setIsPropagating(false);
        }
    }, [sessionId, direction, frameIndex, apiCall]);

    // ── Export Masks ──
    const exportMasks = useCallback(async () => {
        if (!sessionId || frameResults.length === 0) return;
        setIsExporting(true);
        setStatusMsg('Saving masks to disk...');
        try {
            // Build Objects Metadata
            const objectsMeta: Record<string, { label: string; color: string }> = {};
            objects.forEach(obj => {
                objectsMeta[obj.id] = { label: obj.label, color: obj.color };
            });

            // Send entire frameResults to backend to save
            const result = await apiCall('/edit/track/save', {
                session_id: sessionId,
                video_path: videoPath,
                frames: frameResults.map(f => ({
                    frame_idx: f.frameIndex,
                    masks: f.masks,
                    scores: f.scores,
                })),
                objects: objectsMeta
            });
            if (result.error) throw new Error(result.error);
            setStatusMsg(`Saved ${result.count} masks to ${result.path}`);
            setTimeout(() => setStatusMsg('Export/Save complete.'), 3000);
        } catch (e: any) {
            setError(e.message);
            setStatusMsg('Export failed');
        } finally {
            setIsExporting(false);
        }
    }, [sessionId, frameResults, objects, apiCall, videoPath]);

    // ── Remove Object ──
    const removeObject = useCallback(async (objId: number) => {
        if (!sessionId) return;
        try {
            setStatusMsg(`Removing object ${objId}...`);
            const result = await apiCall('/track/remove', {
                session_id: sessionId,
                obj_id: objId
            });
            if (result.error) throw new Error(result.error);

            // Update local state
            setObjects(prev => prev.filter(o => o.id !== objId));
            setFrameResults(prev => prev.map(f => {
                const newMasks = { ...f.masks };
                delete newMasks[objId];
                const newScores = { ...f.scores };
                delete newScores[objId];
                return {
                    ...f,
                    masks: newMasks,
                    scores: newScores
                };
            }));

            setStatusMsg(`Object ${objId} removed.`);
            setTimeout(() => setStatusMsg(''), 2000);
        } catch (e: any) {
            setError(e.message);
            setStatusMsg('Failed to remove object');
        }
    }, [sessionId, apiCall]);

    // ── Toggle Visibility ──
    const toggleVisibility = useCallback((objId: number) => {
        setObjects(prev => prev.map(o =>
            o.id === objId ? { ...o, visible: !o.visible } : o
        ));
    }, []);

    // ── Keyframe Navigation ──
    const jumpToKeyframe = useCallback((d: 'prev' | 'next') => {
        const sorted = Array.from(keyframeIndices).sort((a, b) => a - b);
        if (sorted.length === 0) return;

        let target = -1;
        if (d === 'prev') {
            // Find largest keyframe < current
            target = sorted.reverse().find(k => k < viewingFrame) ?? -1;
        } else {
            // Find smallest keyframe > current
            target = sorted.find(k => k > viewingFrame) ?? -1;
        }

        if (target !== -1) {
            setViewingFrame(target);
            setFrameIndex(target);
        }
    }, [keyframeIndices, viewingFrame]);

    // ── Stop Session ──
    const stopSession = useCallback(async () => {
        if (sessionId) {
            try {
                // We don't strictly *need* to stop the session on backend if we want persistence,
                // but freeing GPU memory is good. Maybe we rely on LRU eviction or explicit close?
                // For now, let's NOT call stop when just closing panel if we want to resume?
                // Actually user said "when I close and reopen tracking panel everything is gone".
                // So we want to Reload data.
                // Stopping releases GPU resources, which is safer. We'll rely on /edit/track/load to restore state.
                await apiCall('/track/stop', { session_id: sessionId });
            } catch { /* best effort */ }
        }
        setSessionId(null);
        setPhase('idle');
        setObjects([]);
        setFrameResults([]);
        setError(null);
        setStatusMsg('');
    }, [sessionId, apiCall]);

    // ── Close handler ──
    const handleClose = useCallback(() => {
        stopSession();
        onClose();
    }, [stopSession, onClose]);

    // ── Sync with Video Playback ──
    useEffect(() => {
        // Poll video time to update viewingFrame
        const updateFrameFromVideo = () => {
            if (_videoRef.current && !_videoRef.current.paused) {
                const currentTime = _videoRef.current.currentTime;
                // Assuming 24 or 30 fps? 
                // Backend usually knows, but let's estimate or define a constant.
                // Ideally this comes from metadata. Defaulting to 25 or 30 is risky.
                // Let's assume 25 for now as a common denominator or try to get it.
                // Better: use ratio if we know total duration and total frames? 
                // We know frameResults max index.

                // Use the fps returned from backend, default to 30 if not set yet
                const currentFps = fps || 30.0;
                const frame = Math.floor(currentTime * currentFps);
                setViewingFrame(frame);
            }
            rafRef.current = requestAnimationFrame(updateFrameFromVideo);
        };

        rafRef.current = requestAnimationFrame(updateFrameFromVideo);

        return () => {
            if (rafRef.current) cancelAnimationFrame(rafRef.current);
        };
    }, [_videoRef]);

    // Also sync when dragging slider/seeking (handled by video controls usually, 
    // but we can listen to 'timeupdate' on video ref if we attached it properly).
    useEffect(() => {
        const vid = _videoRef.current;
        if (!vid) return;

        const onTimeUpdate = () => {
            const currentFps = fps || 30.0;
            const frame = Math.floor(vid.currentTime * currentFps);
            setViewingFrame(frame);
        };

        vid.addEventListener('timeupdate', onTimeUpdate);
        return () => vid.removeEventListener('timeupdate', onTimeUpdate);
    }, [_videoRef, fps]);

    // Keep internal frameIndex synced too if user scrubs?
    // Maybe separation of "viewingFrame" (playback) vs "frameIndex" (prompting) is good.

    // ── O(1) Frame Lookup Map ──
    const frameMap = useMemo(() => {
        const map = new Map<number, FrameResult>();
        frameResults.forEach(f => map.set(f.frameIndex, f));
        return map;
    }, [frameResults]);

    // ── Draw Mask Overlay ──
    useEffect(() => {
        const canvas = overlayRef.current;
        if (!canvas) return;
        canvas.width = containerWidth;
        canvas.height = containerHeight;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const currentFrame = frameMap.get(viewingFrame);
        if (!currentFrame) return;

        // Collect visible objects that have masks for this frame
        const visibleObjects = objects.filter(obj => obj.visible && currentFrame.masks[obj.id]);
        if (visibleObjects.length === 0) return;

        // Load all mask images in parallel, then draw them
        const loadPromises = visibleObjects.map(obj => {
            return new Promise<{ obj: TrackingObject; img: HTMLImageElement }>((resolve) => {
                const img = new Image();
                img.onload = () => resolve({ obj, img });
                img.onerror = () => resolve({ obj, img }); // Skip broken masks gracefully
                img.src = `data:image/png;base64,${currentFrame.masks[obj.id]}`;
            });
        });

        let cancelled = false;
        Promise.all(loadPromises).then(results => {
            if (cancelled) return;
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            results.forEach(({ obj, img }) => {
                if (!img.naturalWidth) return; // Skip failed loads

                // Use an offscreen canvas to colorize each mask independently
                const offscreen = document.createElement('canvas');
                offscreen.width = canvas.width;
                offscreen.height = canvas.height;
                const offCtx = offscreen.getContext('2d');
                if (!offCtx) return;

                // Draw the grayscale mask
                offCtx.drawImage(img, 0, 0, canvas.width, canvas.height);

                // Colorize: fill with the object color, but only where mask pixels exist
                offCtx.globalCompositeOperation = 'source-in';
                offCtx.fillStyle = obj.color;
                offCtx.fillRect(0, 0, canvas.width, canvas.height);

                // Draw the colored mask onto the main canvas with transparency
                ctx.globalAlpha = 0.4;
                ctx.drawImage(offscreen, 0, 0);
            });

            ctx.globalAlpha = 1.0;
        });

        return () => { cancelled = true; };
    }, [frameMap, viewingFrame, objects, containerWidth, containerHeight]);

    // ── Auto-start session on mount ──
    useEffect(() => {
        if (phase === 'idle' && videoPath) {
            startSession();
        }
    }, [videoPath]); // Dependency on videoPath to restart if it changes

    // ── Frame navigation ──
    const navigateFrame = useCallback((delta: number) => {
        setViewingFrame(prev => {
            const next = prev + delta;
            if (next < 0) return 0;
            // Allow going beyond processed frames for prompting?
            return next;
        });
        // Also update the prompting frame index
        setFrameIndex(prev => Math.max(0, prev + delta));
    }, []);

    // ── Render ──
    return (
        <div className="absolute inset-0 z-40 flex flex-col">
            {/* Semi-transparent backdrop */}
            <div className="absolute inset-0 bg-black/30 backdrop-blur-[2px]" />

            {/* Mask Overlay Canvas */}
            <canvas
                ref={overlayRef}
                className={`absolute inset-0 w-full h-full z-10 ${promptMode === 'click' && phase === 'prompting' ? 'cursor-crosshair' : 'pointer-events-none'}`}
                onClick={handleCanvasClick}
            />

            {/* Control Panel */}
            <motion.div
                initial={{ x: -300, opacity: 0 }}
                animate={{ x: 0, opacity: 1 }}
                exit={{ x: -300, opacity: 0 }}
                transition={{ type: 'spring', damping: 25 }}
                className="absolute left-4 top-4 bottom-4 w-72 bg-[#0a0a0a]/95 backdrop-blur-xl border border-white/10 rounded-2xl shadow-2xl z-20 flex flex-col overflow-hidden"
            >
                {/* Header */}
                <div className="flex items-center justify-between px-4 py-3 border-b border-white/5">
                    <div className="flex items-center gap-2">
                        <Crosshair size={16} className="text-cyan-400" />
                        <span className="text-sm font-bold text-white tracking-wide">Object Tracking</span>
                    </div>
                    <button onClick={handleClose} className="text-white/30 hover:text-white transition-colors">
                        <X size={16} />
                    </button>
                </div>

                {/* Status */}
                <div className="px-4 py-2 border-b border-white/5">
                    <div className="flex items-center gap-2">
                        {(phase === 'starting' || phase === 'propagating') && (
                            <Loader2 size={12} className="text-cyan-400 animate-spin" />
                        )}
                        <span className={`text-[11px] font-mono ${error ? 'text-red-400' : 'text-white/50'}`}>
                            {error || statusMsg || 'Ready'}
                        </span>
                    </div>
                </div>

                {/* Prompt Input Section */}
                {phase === 'prompting' && (
                    <div className="px-4 py-3 space-y-3 border-b border-white/5">
                        {/* Prompt mode toggle */}
                        <div className="flex gap-1 bg-white/5 rounded-lg p-0.5">
                            <button
                                onClick={() => setPromptMode('text')}
                                className={`flex-1 text-[10px] font-mono py-1.5 rounded-md transition-all ${promptMode === 'text' ? 'bg-cyan-500/20 text-cyan-400' : 'text-white/40 hover:text-white/60'
                                    }`}
                            >
                                Text Prompt
                            </button>
                            <button
                                onClick={() => setPromptMode('click')}
                                className={`flex-1 text-[10px] font-mono py-1.5 rounded-md transition-all ${promptMode === 'click' ? 'bg-cyan-500/20 text-cyan-400' : 'text-white/40 hover:text-white/60'
                                    }`}
                            >
                                Click Points
                            </button>
                        </div>

                        {/* Text input */}
                        {promptMode === 'text' && (
                            <div className="space-y-2">
                                <input
                                    type="text"
                                    value={textPrompt}
                                    onChange={e => setTextPrompt(e.target.value)}
                                    onKeyDown={e => { if (e.key === 'Enter') addTextPrompt(); }}
                                    placeholder="Be precise, e.g. 'red car'"
                                    title="Describe the exact object to track. Be specific — too broad tracks everything, too narrow misses parts."
                                    className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white/90 font-mono focus:outline-none focus:border-cyan-500/50 placeholder:text-white/20"
                                    autoFocus
                                />
                                <button
                                    onClick={addTextPrompt}
                                    disabled={!textPrompt.trim()}
                                    className="w-full py-2 bg-cyan-500/20 text-cyan-400 font-bold text-xs rounded-lg hover:bg-cyan-500/30 disabled:opacity-30 disabled:cursor-not-allowed transition-all"
                                >
                                    Detect Object
                                </button>
                            </div>
                        )}

                        {/* Click mode instructions */}
                        {promptMode === 'click' && (
                            <div className="text-[10px] text-white/30 font-mono text-center py-2">
                                Click on the object to track.<br />
                                Left-click = include · Right-click = exclude
                            </div>
                        )}

                        {/* Frame selector */}
                        <div className="flex items-center justify-between">
                            <span className="text-[10px] text-white/40 font-mono">Prompt Frame</span>
                            <input
                                type="number"
                                value={frameIndex}
                                onChange={e => setFrameIndex(Math.max(0, parseInt(e.target.value) || 0))}
                                className="w-16 bg-white/5 border border-white/10 rounded px-2 py-1 text-xs text-white/80 font-mono text-right focus:outline-none focus:border-cyan-500/50"
                            />
                        </div>
                    </div>
                )}

                {/* Propagation Controls */}
                {(phase === 'prompting' || phase === 'done') && objects.length > 0 && (
                    <div className="px-4 py-3 space-y-3 border-b border-white/5">
                        <div className="flex items-center justify-between">
                            <span className="text-[10px] text-white/40 font-mono uppercase">Propagation</span>
                        </div>
                        {/* Direction selector */}
                        <div className="flex gap-1 bg-white/5 rounded-lg p-0.5">
                            {(['forward', 'backward', 'both'] as const).map(dir => (
                                <button
                                    key={dir}
                                    onClick={() => setDirection(dir)}
                                    className={`flex-1 text-[10px] font-mono py-1.5 rounded-md transition-all capitalize ${direction === dir ? 'bg-cyan-500/20 text-cyan-400' : 'text-white/40 hover:text-white/60'
                                        }`}
                                >
                                    {dir}
                                </button>
                            ))}
                        </div>
                        <button
                            onClick={propagate}
                            disabled={isPropagating}
                            className="w-full py-2 bg-green-500/20 text-green-400 font-bold text-xs rounded-lg hover:bg-green-500/30 disabled:opacity-30 transition-all flex items-center justify-center gap-2"
                        >
                            {isPropagating ? (
                                <><Loader2 size={12} className="animate-spin" /> Propagating...</>
                            ) : (
                                <><Play size={12} /> Propagate Tracking</>
                            )}
                        </button>
                    </div>
                )}

                {/* Tracked Objects List */}
                {objects.length > 0 && (
                    <div className="flex-1 overflow-y-auto px-4 py-3 space-y-2">
                        <span className="text-[10px] text-white/40 font-mono uppercase">Tracked Objects</span>
                        {objects.map(obj => (
                            <div
                                key={obj.id}
                                className="flex items-center gap-2 px-2 py-1.5 bg-white/5 rounded-lg"
                            >
                                <button
                                    onClick={() => toggleVisibility(obj.id)}
                                    className="text-white/40 hover:text-white transition-colors"
                                >
                                    {obj.visible ? <Eye size={12} /> : <EyeOff size={12} />}
                                </button>
                                <div
                                    className="w-3 h-3 rounded-full flex-shrink-0"
                                    style={{ backgroundColor: obj.color }}
                                />
                                <span className="text-xs text-white/70 font-mono flex-1 truncate">
                                    {obj.label}
                                </span>
                                <span className="text-[9px] text-white/30 font-mono">
                                    #{obj.id}
                                </span>
                                <button
                                    onClick={() => removeObject(obj.id)}
                                    className="text-white/20 hover:text-red-400 transition-colors ml-1"
                                    title="Remove Object"
                                >
                                    <Trash2 size={12} />
                                </button>
                            </div>
                        ))}
                    </div>
                )}

                {/* Frame Navigation (when results exist) */}
                {frameResults.length > 0 && (
                    <div className="px-4 py-3 border-t border-white/5">
                        <div className="flex items-center justify-between">
                            <button
                                onClick={() => navigateFrame(-1)}
                                disabled={viewingFrame === 0}
                                className="text-white/40 hover:text-white disabled:opacity-20 transition-colors"
                            >
                                <ChevronLeft size={16} />
                            </button>
                            <span className="text-[10px] text-white/50 font-mono flex items-center gap-2">
                                <span className="flex gap-0.5">
                                    <button onClick={() => jumpToKeyframe('prev')} disabled={keyframeIndices.size === 0} className="hover:text-white disabled:opacity-30"><ChevronsLeft size={12} /></button>
                                    <button onClick={() => jumpToKeyframe('next')} disabled={keyframeIndices.size === 0} className="hover:text-white disabled:opacity-30"><ChevronsRight size={12} /></button>
                                </span>
                                Frame {viewingFrame + 1}{totalFrames > 0 ? ` / ${totalFrames}` : (frameResults.length > 0 ? ` / ${frameResults.length}` : '')}
                                {keyframeIndices.has(viewingFrame) && (
                                    <div className="w-1.5 h-1.5 rounded-full bg-yellow-400" title="Keyframe (User Prompt)" />
                                )}
                            </span>
                            <button
                                onClick={() => navigateFrame(1)}
                                disabled={viewingFrame >= frameResults.length - 1}
                                className="text-white/40 hover:text-white disabled:opacity-20 transition-colors"
                            >
                                <ChevronRight size={16} />
                            </button>
                        </div>
                    </div>
                )}

                {/* Footer Actions */}
                <div className="px-4 py-3 border-t border-white/5 flex gap-2">
                    <button
                        onClick={stopSession}
                        className="flex-1 py-2 bg-white/5 text-white/40 font-bold text-xs rounded-lg hover:bg-white/10 hover:text-white/60 transition-all flex items-center justify-center gap-1"
                    >
                        <RotateCcw size={12} /> Reset
                    </button>
                    <button
                        onClick={handleClose}
                        className="flex-1 py-2 bg-red-500/15 text-red-400 font-bold text-xs rounded-lg hover:bg-red-500/25 transition-all flex items-center justify-center gap-1"
                    >
                        <Square size={12} /> Close
                    </button>
                    {frameResults.length > 0 && (
                        <button
                            onClick={exportMasks}
                            disabled={isExporting}
                            className="flex-1 py-2 bg-blue-500/20 text-blue-400 font-bold text-xs rounded-lg hover:bg-blue-500/30 disabled:opacity-30 transition-all flex items-center justify-center gap-1"
                        >
                            {isExporting ? <Loader2 size={12} className="animate-spin" /> : <Download size={12} />} Export
                        </button>
                    )}
                </div>
            </motion.div>
        </div>
    );
});

TrackingPanel.displayName = 'TrackingPanel';
