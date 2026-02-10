import { useState, useEffect } from 'react';

interface DetectedObject {
    id: number;
    mask: string;   // base64 PNG
    bbox: number[]; // [x0, y0, x1, y1]
    score: number;
    label: string;
    selected: boolean;
}

interface SegmentationOverlayProps {
    objects: DetectedObject[];
    containerWidth: number;
    containerHeight: number;
    imageWidth: number;
    imageHeight: number;
    onToggleObject: (id: number) => void;
    onClearAll: () => void;
}

const COLORS = [
    'rgba(99, 102, 241, 0.4)',   // Indigo
    'rgba(236, 72, 153, 0.4)',   // Pink
    'rgba(34, 197, 94, 0.4)',    // Green
    'rgba(245, 158, 11, 0.4)',   // Amber
    'rgba(6, 182, 212, 0.4)',    // Cyan
    'rgba(168, 85, 247, 0.4)',   // Purple
    'rgba(239, 68, 68, 0.4)',    // Red
    'rgba(20, 184, 166, 0.4)',   // Teal
];

const BORDER_COLORS = [
    'rgba(99, 102, 241, 0.9)',
    'rgba(236, 72, 153, 0.9)',
    'rgba(34, 197, 94, 0.9)',
    'rgba(245, 158, 11, 0.9)',
    'rgba(6, 182, 212, 0.9)',
    'rgba(168, 85, 247, 0.9)',
    'rgba(239, 68, 68, 0.9)',
    'rgba(20, 184, 166, 0.9)',
];

export const SegmentationOverlay = ({
    objects,
    containerWidth,
    containerHeight,
    imageWidth,
    imageHeight,
    onToggleObject,
}: SegmentationOverlayProps) => {
    const [maskImages, setMaskImages] = useState<Record<number, HTMLImageElement>>({});

    // Pre-load mask images
    useEffect(() => {
        const loaded: Record<number, HTMLImageElement> = {};
        let pending = objects.length;
        if (pending === 0) {
            setMaskImages({});
            return;
        }

        objects.forEach((obj) => {
            const img = new Image();
            img.onload = () => {
                loaded[obj.id] = img;
                pending--;
                if (pending === 0) setMaskImages({ ...loaded });
            };
            img.onerror = () => {
                pending--;
                if (pending === 0) setMaskImages({ ...loaded });
            };
            img.src = `data:image/png;base64,${obj.mask}`;
        });
    }, [objects]);

    // Scale factors
    const scaleX = containerWidth / imageWidth;
    const scaleY = containerHeight / imageHeight;

    return (
        <div className="absolute inset-0 pointer-events-none z-30">
            {objects.map((obj, idx) => {
                const colorIdx = idx % COLORS.length;
                const [x0, y0, x1, y1] = obj.bbox;

                // Scale bbox to container
                const left = x0 * scaleX;
                const top = y0 * scaleY;
                const width = (x1 - x0) * scaleX;
                const height = (y1 - y0) * scaleY;

                return (
                    <div key={obj.id}>
                        {/* Mask overlay */}
                        {obj.selected && maskImages[obj.id] && (
                            <canvas
                                className="absolute inset-0 w-full h-full"
                                ref={(canvas) => {
                                    if (!canvas || !maskImages[obj.id]) return;
                                    canvas.width = containerWidth;
                                    canvas.height = containerHeight;
                                    const ctx = canvas.getContext('2d');
                                    if (!ctx) return;
                                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                                    ctx.globalAlpha = 0.4;
                                    // Tint the mask
                                    ctx.drawImage(maskImages[obj.id], 0, 0, containerWidth, containerHeight);
                                    ctx.globalCompositeOperation = 'source-in';
                                    ctx.fillStyle = COLORS[colorIdx].replace('0.4', '0.6');
                                    ctx.fillRect(0, 0, canvas.width, canvas.height);
                                    ctx.globalCompositeOperation = 'source-over';
                                    ctx.globalAlpha = 1.0;
                                }}
                            />
                        )}

                        {/* Bounding box + label */}
                        <div
                            className="absolute pointer-events-auto cursor-pointer transition-opacity"
                            style={{
                                left: `${left}px`,
                                top: `${top}px`,
                                width: `${width}px`,
                                height: `${height}px`,
                                border: `2px ${obj.selected ? 'solid' : 'dashed'} ${BORDER_COLORS[colorIdx]}`,
                                opacity: obj.selected ? 1 : 0.5,
                            }}
                            onClick={() => onToggleObject(obj.id)}
                        >
                            {/* Label tag */}
                            <div
                                className="absolute -top-5 left-0 px-1.5 py-0.5 text-[9px] font-mono font-bold text-white rounded-t"
                                style={{ backgroundColor: BORDER_COLORS[colorIdx] }}
                            >
                                {obj.label} {(obj.score * 100).toFixed(0)}%
                            </div>
                        </div>
                    </div>
                );
            })}
        </div>
    );
};

export type { DetectedObject };
