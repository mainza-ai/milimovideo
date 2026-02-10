import { useState, useCallback, useRef, useEffect, memo } from 'react';
import { motion } from 'framer-motion';
import { X, Crosshair, Play, Square, RotateCcw, ChevronLeft, ChevronRight, Loader2 } from 'lucide-react';
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
    masks: Record<number, string>; // obj_id -> base64 mask PNG
    scores: Record<number, number>;
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

    // ── Mask Overlay Canvas ──
    const overlayRef = useRef<HTMLCanvasElement>(null);

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

    // ── Start Session ──
    const startSession = useCallback(async () => {
        setPhase('starting');
        setError(null);
        setStatusMsg('Starting tracking session...');
        try {
            const result = await apiCall('/track/start', { video_path: videoPath });
            if (result.error) throw new Error(result.error);
            setSessionId(result.session_id);
            setPhase('prompting');
            setStatusMsg('Session started. Add a prompt to begin tracking.');
        } catch (e: any) {
            setError(e.message);
            setPhase('error');
            setStatusMsg('');
        }
    }, [videoPath, apiCall]);

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
            setObjects(prev => [...prev, ...newObjects]);

            // Store initial frame result
            if (result.masks) {
                setFrameResults([{
                    frameIndex,
                    masks: result.masks,
                    scores: result.scores || {},
                }]);
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
            const result = await apiCall('/track/prompt', {
                session_id: sessionId,
                frame_idx: frameIndex,
                points: [[x, y]],
                point_labels: [1], // positive click
            });
            if (result.error) throw new Error(result.error);

            const newObjects: TrackingObject[] = (result.object_ids || []).map((id: number, idx: number) => ({
                id,
                label: `Object ${id}`,
                color: TRACK_COLORS[(objects.length + idx) % TRACK_COLORS.length],
                visible: true,
            }));
            setObjects(prev => [...prev, ...newObjects]);
            setStatusMsg(`Point detected ${newObjects.length} object(s).`);
        } catch (e: any) {
            setError(e.message);
        }
    }, [sessionId, promptMode, phase, frameIndex, objects.length, apiCall]);

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

            // Parse frame results
            const frames: FrameResult[] = (result.frames || []).map((f: any) => ({
                frameIndex: f.frame_index,
                masks: f.masks || {},
                scores: f.scores || {},
            }));
            setFrameResults(frames);
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

    // ── Stop Session ──
    const stopSession = useCallback(async () => {
        if (sessionId) {
            try {
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

    // ── Draw Mask Overlay ──
    useEffect(() => {
        const canvas = overlayRef.current;
        if (!canvas) return;
        canvas.width = containerWidth;
        canvas.height = containerHeight;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const currentFrame = frameResults.find(f => f.frameIndex === viewingFrame);
        if (!currentFrame) return;

        objects.forEach(obj => {
            if (!obj.visible) return;
            const maskB64 = currentFrame.masks[obj.id];
            if (!maskB64) return;

            const img = new Image();
            img.onload = () => {
                ctx.globalAlpha = 0.35;
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                ctx.globalCompositeOperation = 'source-in';
                ctx.fillStyle = obj.color;
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                ctx.globalCompositeOperation = 'source-over';
                ctx.globalAlpha = 1.0;
            };
            img.src = `data:image/png;base64,${maskB64}`;
        });
    }, [frameResults, viewingFrame, objects, containerWidth, containerHeight]);

    // ── Auto-start session on mount ──
    useEffect(() => {
        if (phase === 'idle' && videoPath) {
            startSession();
        }
    }, []);

    // ── Frame navigation ──
    const navigateFrame = useCallback((delta: number) => {
        setViewingFrame(prev => {
            const next = prev + delta;
            if (next < 0 || next >= frameResults.length) return prev;
            return next;
        });
    }, [frameResults.length]);

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
                            <span className="text-[10px] text-white/50 font-mono">
                                Frame {viewingFrame + 1} / {frameResults.length}
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
                </div>
            </motion.div>
        </div>
    );
});

TrackingPanel.displayName = 'TrackingPanel';
