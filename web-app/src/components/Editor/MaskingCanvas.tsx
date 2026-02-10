import { useRef, useEffect, useState, useCallback } from 'react';
import { Eraser, Paintbrush, RotateCcw, Check, MousePointer, Type, Loader2 } from 'lucide-react';
import { SegmentationOverlay, type DetectedObject } from './SegmentationOverlay';

interface MaskingCanvasProps {
    width: number;
    height: number;
    videoRef?: React.RefObject<HTMLVideoElement | null>;
    onSave: (maskDataUrl: string) => void;
    onCancel: () => void;
}

const API_BASE = 'http://localhost:8000';

export const MaskingCanvas = ({ width, height, videoRef, onSave, onCancel }: MaskingCanvasProps) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);
    const [isDrawing, setIsDrawing] = useState(false);
    const [tool, setTool] = useState<'brush' | 'eraser' | 'click' | 'text'>('brush');
    const [brushSize, setBrushSize] = useState(20);
    const [textPrompt, setTextPrompt] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [errorMsg, setErrorMsg] = useState<string | null>(null);
    const [detectedObjects, setDetectedObjects] = useState<DetectedObject[]>([]);
    const [samClickPoints, setSamClickPoints] = useState<{ x: number; y: number; label: number }[]>([]);

    // Init canvas
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        if (ctx) {
            ctx.lineCap = 'round';
            ctx.lineJoin = 'round';
            ctx.clearRect(0, 0, width, height);
        }
    }, [width, height]);

    const getPos = (e: React.MouseEvent | React.TouchEvent) => {
        const canvas = canvasRef.current;
        if (!canvas) return { x: 0, y: 0 };
        const rect = canvas.getBoundingClientRect();

        let clientX, clientY;
        if ('touches' in e) {
            clientX = e.touches[0].clientX;
            clientY = e.touches[0].clientY;
        } else {
            clientX = (e as React.MouseEvent).clientX;
            clientY = (e as React.MouseEvent).clientY;
        }

        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;

        return {
            x: (clientX - rect.left) * scaleX,
            y: (clientY - rect.top) * scaleY
        };
    };

    const captureFrameBlob = useCallback(async (): Promise<{ blob: Blob; w: number; h: number } | null> => {
        // Use videoRef prop first, then fall back to DOM search
        const videoEl = videoRef?.current
            || canvasRef.current?.parentElement?.parentElement?.querySelector('video') as HTMLVideoElement | null;
        const imgEl = canvasRef.current?.parentElement?.parentElement?.querySelector('img') as HTMLImageElement | null;

        if (videoEl && videoEl.videoWidth > 0) {
            const w = videoEl.videoWidth;
            const h = videoEl.videoHeight;
            const tmpCanvas = document.createElement('canvas');
            tmpCanvas.width = w; tmpCanvas.height = h;
            const ctx = tmpCanvas.getContext('2d');
            if (ctx) ctx.drawImage(videoEl, 0, 0);
            const blob = await new Promise<Blob | null>(resolve => tmpCanvas.toBlob(resolve, 'image/jpeg', 0.95));
            if (blob) return { blob, w, h };
        } else if (imgEl && imgEl.naturalWidth > 0) {
            const w = imgEl.naturalWidth;
            const h = imgEl.naturalHeight;
            const tmpCanvas = document.createElement('canvas');
            tmpCanvas.width = w; tmpCanvas.height = h;
            const ctx = tmpCanvas.getContext('2d');
            if (ctx) ctx.drawImage(imgEl, 0, 0);
            const blob = await new Promise<Blob | null>(resolve => tmpCanvas.toBlob(resolve, 'image/jpeg', 0.95));
            if (blob) return { blob, w, h };
        }
        return null;
    }, [videoRef]);

    // ─── Click-to-Segment ──────────────────────────────────────────

    const handleClickSegment = useCallback(async (e: React.MouseEvent) => {
        if (tool !== 'click') return;
        e.preventDefault();

        const canvas = canvasRef.current;
        if (!canvas) return;

        const { x, y } = getPos(e);
        const label = 1;
        const newPoints = [...samClickPoints, { x, y, label }];
        setSamClickPoints(newPoints);
        setErrorMsg(null);

        setIsLoading(true);
        try {
            const frame = await captureFrameBlob();
            if (!frame) {
                setErrorMsg('No video frame available — select a generated shot first');
                setIsLoading(false);
                return;
            }

            const formData = new FormData();
            formData.append('image', frame.blob, 'frame.jpg');
            formData.append('points', JSON.stringify(newPoints.map(p => [p.x, p.y])));
            formData.append('labels', JSON.stringify(newPoints.map(p => p.label)));

            const res = await fetch(`${API_BASE}/edit/segment`, {
                method: 'POST',
                body: formData,
            });

            if (res.ok) {
                const maskBlob = await res.blob();
                const maskUrl = URL.createObjectURL(maskBlob);

                const maskImg = new Image();
                maskImg.onload = () => {
                    const ctx = canvas.getContext('2d');
                    if (ctx) {
                        ctx.clearRect(0, 0, width, height);
                        ctx.globalAlpha = 0.8;
                        ctx.drawImage(maskImg, 0, 0, width, height);
                        ctx.globalAlpha = 1.0;
                    }
                    URL.revokeObjectURL(maskUrl);
                };
                maskImg.src = maskUrl;
            } else {
                const errText = await res.text();
                setErrorMsg(`SAM error: ${errText.slice(0, 100)}`);
            }
        } catch (err) {
            console.error('SAM click-to-segment failed:', err);
            setErrorMsg('SAM service unavailable — is run_sam.sh running?');
        } finally {
            setIsLoading(false);
        }
    }, [tool, samClickPoints, width, height, captureFrameBlob]);

    // ─── Text-Prompted Detection ───────────────────────────────────

    const handleTextDetect = useCallback(async () => {
        if (!textPrompt.trim()) return;

        const canvas = canvasRef.current;
        if (!canvas) return;

        setIsLoading(true);
        setErrorMsg(null);
        try {
            const frame = await captureFrameBlob();
            if (!frame) {
                setErrorMsg('No video frame available — select a generated shot first');
                setIsLoading(false);
                return;
            }

            const formData = new FormData();
            formData.append('image', frame.blob, 'frame.jpg');
            formData.append('text', textPrompt);
            formData.append('confidence', '0.5');

            const res = await fetch(`${API_BASE}/edit/detect`, {
                method: 'POST',
                body: formData,
            });

            if (res.ok) {
                const data = await res.json();
                const objects: DetectedObject[] = (data.objects || []).map((obj: any) => ({
                    ...obj,
                    selected: true,
                }));
                setDetectedObjects(objects);

                if (objects.length > 0) {
                    applyDetectedMasksToCanvas(objects.filter(o => o.selected));
                } else {
                    setErrorMsg(`No objects matching "${textPrompt}" found — try a different description`);
                }
            } else {
                const errText = await res.text();
                setErrorMsg(`SAM error: ${errText.slice(0, 100)}`);
            }
        } catch (err) {
            console.error('SAM text detection failed:', err);
            setErrorMsg('SAM service unavailable — is run_sam.sh running?');
        } finally {
            setIsLoading(false);
        }
    }, [textPrompt, captureFrameBlob]);

    const applyDetectedMasksToCanvas = useCallback((objects: DetectedObject[]) => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        ctx.clearRect(0, 0, width, height);

        // Merge all selected masks onto the canvas
        let loadedCount = 0;
        objects.forEach((obj) => {
            const img = new Image();
            img.onload = () => {
                ctx.globalCompositeOperation = 'source-over';
                ctx.globalAlpha = 0.8;
                ctx.drawImage(img, 0, 0, width, height);
                ctx.globalAlpha = 1.0;
                loadedCount++;
            };
            img.src = `data:image/png;base64,${obj.mask}`;
        });
    }, [width, height]);

    const handleToggleObject = useCallback((id: number) => {
        setDetectedObjects(prev => {
            const updated = prev.map(obj =>
                obj.id === id ? { ...obj, selected: !obj.selected } : obj
            );
            applyDetectedMasksToCanvas(updated.filter(o => o.selected));
            return updated;
        });
    }, [applyDetectedMasksToCanvas]);

    // ─── Brush/Eraser Drawing ──────────────────────────────────────

    const startDraw = (e: React.MouseEvent | React.TouchEvent) => {
        if (tool !== 'brush' && tool !== 'eraser') return;
        e.preventDefault();
        setIsDrawing(true);
        const { x, y } = getPos(e);
        const ctx = canvasRef.current?.getContext('2d');
        if (!ctx) return;
        ctx.beginPath();
        ctx.moveTo(x, y);
    };

    const draw = (e: React.MouseEvent | React.TouchEvent) => {
        if (!isDrawing || (tool !== 'brush' && tool !== 'eraser')) return;
        e.preventDefault();
        const { x, y } = getPos(e);
        const ctx = canvasRef.current?.getContext('2d');
        if (!ctx) return;

        ctx.lineWidth = brushSize;
        if (tool === 'brush') {
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)';
            ctx.globalCompositeOperation = 'source-over';
        } else {
            ctx.globalCompositeOperation = 'destination-out';
        }
        ctx.lineTo(x, y);
        ctx.stroke();
    };

    const stopDraw = () => setIsDrawing(false);

    const handleClear = () => {
        const ctx = canvasRef.current?.getContext('2d');
        if (ctx) ctx.clearRect(0, 0, width, height);
        setDetectedObjects([]);
        setSamClickPoints([]);
    };

    const handleSave = () => {
        if (!canvasRef.current) return;

        // Export B&W mask: black bg + white painted areas
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = width;
        tempCanvas.height = height;
        const tCtx = tempCanvas.getContext('2d');
        if (tCtx) {
            tCtx.fillStyle = '#000000';
            tCtx.fillRect(0, 0, width, height);
            tCtx.drawImage(canvasRef.current, 0, 0);
        }

        const dataUrl = tempCanvas.toDataURL('image/png');
        onSave(dataUrl);
    };

    const cursorStyle = tool === 'click' ? 'crosshair' : tool === 'text' ? 'default' : 'crosshair';

    return (
        <div ref={containerRef} className="absolute inset-0 z-50 flex flex-col items-center justify-center">
            {/* Canvas */}
            <canvas
                ref={canvasRef}
                className="absolute inset-0 w-full h-full touch-none"
                style={{ cursor: cursorStyle }}
                onMouseDown={tool === 'click' ? handleClickSegment : startDraw}
                onMouseMove={draw}
                onMouseUp={stopDraw}
                onMouseLeave={stopDraw}
                onTouchStart={startDraw}
                onTouchMove={draw}
                onTouchEnd={stopDraw}
            />

            {/* Detection Overlay */}
            {detectedObjects.length > 0 && (
                <SegmentationOverlay
                    objects={detectedObjects}
                    containerWidth={containerRef.current?.clientWidth || width}
                    containerHeight={containerRef.current?.clientHeight || height}
                    imageWidth={width}
                    imageHeight={height}
                    onToggleObject={handleToggleObject}
                    onClearAll={() => setDetectedObjects([])}
                />
            )}

            {/* Loading Indicator */}
            {isLoading && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/40 z-40">
                    <div className="flex items-center gap-2 bg-[#111] border border-white/10 px-4 py-2 rounded-xl text-white/80 text-sm font-mono">
                        <Loader2 size={16} className="animate-spin" />
                        SAM 3 Processing...
                    </div>
                </div>
            )}

            {/* Error Message */}
            {errorMsg && !isLoading && (
                <div className="absolute bottom-20 left-1/2 -translate-x-1/2 z-50 animate-in fade-in">
                    <div
                        className="flex items-center gap-2 bg-red-900/80 border border-red-500/30 px-4 py-2 rounded-xl text-red-200 text-xs font-mono max-w-md cursor-pointer"
                        onClick={() => setErrorMsg(null)}
                    >
                        {errorMsg}
                    </div>
                </div>
            )}

            {/* Toolbar */}
            <div className="absolute top-4 flex gap-1 bg-[#111] border border-white/10 p-1.5 rounded-xl shadow-2xl backdrop-blur-md z-50" onClick={e => e.stopPropagation()}>
                {/* Manual tools */}
                <button
                    onClick={() => setTool('brush')}
                    className={`p-2 rounded-lg transition-colors ${tool === 'brush' ? 'bg-milimo-500 text-black' : 'text-white/50 hover:bg-white/10'}`}
                    title="Brush"
                >
                    <Paintbrush size={16} />
                </button>
                <button
                    onClick={() => setTool('eraser')}
                    className={`p-2 rounded-lg transition-colors ${tool === 'eraser' ? 'bg-milimo-500 text-black' : 'text-white/50 hover:bg-white/10'}`}
                    title="Eraser"
                >
                    <Eraser size={16} />
                </button>

                <div className="w-px h-8 bg-white/10 mx-0.5 self-center" />

                {/* SAM tools */}
                <button
                    onClick={() => { setTool('click'); setSamClickPoints([]); }}
                    className={`p-2 rounded-lg transition-colors ${tool === 'click' ? 'bg-cyan-500 text-black' : 'text-white/50 hover:bg-white/10'}`}
                    title="Click to Segment (SAM 3)"
                >
                    <MousePointer size={16} />
                </button>
                <button
                    onClick={() => setTool('text')}
                    className={`p-2 rounded-lg transition-colors ${tool === 'text' ? 'bg-cyan-500 text-black' : 'text-white/50 hover:bg-white/10'}`}
                    title="Text Prompt (SAM 3)"
                >
                    <Type size={16} />
                </button>

                {/* Brush size (only for brush/eraser) */}
                {(tool === 'brush' || tool === 'eraser') && (
                    <>
                        <div className="w-px h-8 bg-white/10 mx-0.5 self-center" />
                        <input
                            type="range"
                            min="5" max="100"
                            value={brushSize}
                            onChange={e => setBrushSize(parseInt(e.target.value))}
                            className="w-20 accent-milimo-500"
                            title="Brush Size"
                        />
                    </>
                )}

                <div className="w-px h-8 bg-white/10 mx-0.5 self-center" />

                <button
                    onClick={handleClear}
                    className="p-2 text-white/50 hover:text-red-400 hover:bg-white/10 rounded-lg transition-colors"
                    title="Clear All"
                >
                    <RotateCcw size={16} />
                </button>
            </div>

            {/* Text Prompt Input (when text tool is active) */}
            {tool === 'text' && (
                <div className="absolute top-16 flex items-center gap-2 bg-[#111] border border-white/10 p-2 rounded-xl shadow-2xl backdrop-blur-md z-50" onClick={e => e.stopPropagation()}>
                    <input
                        type="text"
                        value={textPrompt}
                        onChange={e => setTextPrompt(e.target.value)}
                        onKeyDown={e => { if (e.key === 'Enter') handleTextDetect(); }}
                        placeholder="Be precise, e.g. 'red car' not 'car'"
                        title="Describe the exact object to segment. Be specific about which object you mean — too broad captures everything, too narrow misses parts."
                        className="bg-transparent border border-white/10 rounded-lg px-3 py-1.5 text-sm text-white/90 font-mono w-64 focus:outline-none focus:border-cyan-500/50 placeholder:text-white/20"
                        autoFocus
                    />
                    <button
                        onClick={handleTextDetect}
                        disabled={isLoading || !textPrompt.trim()}
                        className="px-3 py-1.5 bg-cyan-500 text-black font-bold text-sm rounded-lg hover:bg-cyan-400 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                    >
                        Detect
                    </button>
                </div>
            )}

            {/* Click mode instructions */}
            {tool === 'click' && (
                <div className="absolute top-16 text-cyan-400/70 text-[10px] font-mono pointer-events-none bg-black/60 px-3 py-1.5 rounded-lg z-50">
                    CLICK ON OBJECTS TO SEGMENT • {samClickPoints.length} point{samClickPoints.length !== 1 ? 's' : ''} placed
                </div>
            )}

            {/* Action Bar */}
            <div className="absolute bottom-6 flex gap-4 z-50" onClick={e => e.stopPropagation()}>
                <button
                    onClick={onCancel}
                    className="px-6 py-2 bg-black/60 backdrop-blur border border-white/10 rounded-full text-white/70 hover:text-white hover:bg-white/10 transition-colors font-medium text-sm"
                >
                    Cancel
                </button>
                <button
                    onClick={handleSave}
                    className="px-6 py-2 bg-milimo-500 text-black rounded-full font-bold text-sm shadow-lg shadow-milimo-500/20 hover:bg-milimo-400 transition-colors flex items-center gap-2"
                >
                    <Check size={16} /> Apply Mask
                </button>
            </div>

            {/* Helper Text */}
            {tool === 'brush' && (
                <div className="absolute top-16 text-white/50 text-[10px] font-mono pointer-events-none bg-black/50 px-2 py-1 rounded z-50">
                    DRAW TO MASK AREA (WHITE = EDIT)
                </div>
            )}
        </div>
    );
};
