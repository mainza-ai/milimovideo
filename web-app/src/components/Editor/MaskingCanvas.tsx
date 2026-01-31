import { useRef, useEffect, useState } from 'react';
import { Eraser, Paintbrush, RotateCcw, Check } from 'lucide-react';

interface MaskingCanvasProps {
    width: number;
    height: number;
    onSave: (maskDataUrl: string) => void;
    onCancel: () => void;
}

export const MaskingCanvas = ({ width, height, onSave, onCancel }: MaskingCanvasProps) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [isDrawing, setIsDrawing] = useState(false);
    const [tool, setTool] = useState<'brush' | 'eraser'>('brush');
    const [brushSize, setBrushSize] = useState(20);

    // Init canvas
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        // Set resolution (accounting for high DPI displays if needed, but for mask generation 1:1 is usually best to avoid scaling artifacts)
        canvas.width = width;
        canvas.height = height;

        const ctx = canvas.getContext('2d');
        if (ctx) {
            ctx.lineCap = 'round';
            ctx.lineJoin = 'round';
            // Clear to transparent
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

    const startDraw = (e: React.MouseEvent | React.TouchEvent) => {
        e.preventDefault(); // Prevent scroll on touch
        setIsDrawing(true);
        const { x, y } = getPos(e);
        const ctx = canvasRef.current?.getContext('2d');
        if (!ctx) return;

        ctx.beginPath();
        ctx.moveTo(x, y);
    };

    const draw = (e: React.MouseEvent | React.TouchEvent) => {
        if (!isDrawing) return;
        e.preventDefault();
        const { x, y } = getPos(e);
        const ctx = canvasRef.current?.getContext('2d');
        if (!ctx) return;

        ctx.lineWidth = brushSize;
        if (tool === 'brush') {
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)'; // Semi-transparent white for visibility
            ctx.globalCompositeOperation = 'source-over';
        } else {
            ctx.globalCompositeOperation = 'destination-out'; // Eraser
        }

        ctx.lineTo(x, y);
        ctx.stroke();
    };

    const stopDraw = () => {
        setIsDrawing(false);
    };

    const handleClear = () => {
        const ctx = canvasRef.current?.getContext('2d');
        if (ctx) ctx.clearRect(0, 0, width, height);
    };

    const handleSave = () => {
        if (!canvasRef.current) return;

        // Export mask. 
        // IMPORTANT: We need a black and white mask (white = inpaint, black = ignored).
        // Our canvas currently has transparent background and white brush.
        // We might want to composit it onto a black background before exporting if the backend expects RGB grayscale.
        // Or if backend handles transparency. Let's send a standard B&W image to be safe.

        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = width;
        tempCanvas.height = height;
        const tCtx = tempCanvas.getContext('2d');
        if (tCtx) {
            // Fill black
            tCtx.fillStyle = '#000000';
            tCtx.fillRect(0, 0, width, height);
            // Draw original canvas (which is white strokes on transparent)
            tCtx.drawImage(canvasRef.current, 0, 0);
        }

        const dataUrl = tempCanvas.toDataURL('image/png');
        onSave(dataUrl);
    };

    return (
        <div className="absolute inset-0 z-50 flex flex-col items-center justify-center">
            {/* Canvas */}
            <canvas
                ref={canvasRef}
                className="absolute inset-0 w-full h-full cursor-crosshair touch-none"
                onMouseDown={startDraw}
                onMouseMove={draw}
                onMouseUp={stopDraw}
                onMouseLeave={stopDraw}
                onTouchStart={startDraw}
                onTouchMove={draw}
                onTouchEnd={stopDraw}
            />

            {/* Toolbar */}
            <div className="absolute top-4 flex gap-2 bg-[#111] border border-white/10 p-2 rounded-xl shadow-2xl backdrop-blur-md" onClick={e => e.stopPropagation()}>
                <button
                    onClick={() => setTool('brush')}
                    className={`p-2 rounded-lg transition-colors ${tool === 'brush' ? 'bg-milimo-500 text-black' : 'text-white/50 hover:bg-white/10'}`}
                    title="Brush"
                >
                    <Paintbrush size={18} />
                </button>
                <button
                    onClick={() => setTool('eraser')}
                    className={`p-2 rounded-lg transition-colors ${tool === 'eraser' ? 'bg-milimo-500 text-black' : 'text-white/50 hover:bg-white/10'}`}
                    title="Eraser"
                >
                    <Eraser size={18} />
                </button>

                <div className="w-px h-8 bg-white/10 mx-1" />

                <input
                    type="range"
                    min="5" max="100"
                    value={brushSize}
                    onChange={e => setBrushSize(parseInt(e.target.value))}
                    className="w-24 accent-milimo-500"
                    title="Brush Size"
                />

                <div className="w-px h-8 bg-white/10 mx-1" />

                <button
                    onClick={handleClear}
                    className="p-2 text-white/50 hover:text-red-400 hover:bg-white/10 rounded-lg transition-colors"
                    title="Clear All"
                >
                    <RotateCcw size={18} />
                </button>
            </div>

            {/* Action Bar */}
            <div className="absolute bottom-6 flex gap-4" onClick={e => e.stopPropagation()}>
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
            <div className="absolute top-20 text-white/50 text-[10px] font-mono pointer-events-none bg-black/50 px-2 py-1 rounded">
                DRAW TO MASK AREA (WHITE = EDIT)
            </div>
        </div>
    );
};
