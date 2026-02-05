import { memo } from 'react';
import { motion } from 'framer-motion';
import { clsx } from 'clsx';
import { Music } from 'lucide-react';
import { useTimelineStore } from '../../stores/timelineStore';
import { getSnapPoint } from '../../utils/snapEngine';
import { AudioClip } from './AudioClip';

interface Props {
    clip: any;
    trackType: 'video' | 'audio';
    trackIndex: number;
    zoom: number;
    isSelected: boolean;
    allClips: any[];
    onSnap: (lines: number[]) => void;
    onDragEnd: (id: string, newStart: number, newTrack: number, action?: any) => void;
}

export const TimelineClip = memo(({ clip, trackType, trackIndex, zoom, isSelected, allClips, onSnap, onDragEnd }: Props) => {
    const selectShot = useTimelineStore(state => state.selectShot);
    const fps = useTimelineStore(state => state.project.fps);

    const startPx = clip.start * zoom;
    const widthPx = clip.duration * zoom;

    const handleDrag = (_: any, info: any) => {
        const movePx = info.offset.x;
        const potentialNewPx = startPx + movePx;
        const potentialTime = Math.max(0, potentialNewPx / zoom);
        const myDuration = clip.numFrames / (fps || 25);
        const endTime = potentialTime + myDuration;
        useTimelineStore.getState().setTransientDuration(endTime + 5);

        const snap = getSnapPoint(potentialTime, zoom, clip.id, allClips, trackIndex);
        if (snap.isSnapped) onSnap([snap.snappedTime]);
        else onSnap([]);
    };

    const handleDragEndInternal = (_: any, info: any) => {
        onSnap([]);
        useTimelineStore.getState().setTransientDuration(null);
        const movePx = info.offset.x;
        if (Math.abs(movePx) < 5) return;

        const newPx = startPx + movePx;
        let newTime = Math.max(0, newPx / zoom);

        const snap = getSnapPoint(newTime, zoom, clip.id, allClips, trackIndex === 0 ? 0 : 0);
        if (snap.isSnapped) newTime = snap.snappedTime;

        const currentFps = fps || 25;
        const newStartFrame = Math.round(newTime * currentFps);
        onDragEnd(clip.id, newStartFrame, trackIndex);
    };

    const startResize = (e: React.MouseEvent, type: 'start' | 'end') => {
        e.preventDefault();
        e.stopPropagation();
        const startX = e.clientX;

        const onUp = (upEvent: MouseEvent) => {
            const deltaX = upEvent.clientX - startX;
            if (Math.abs(deltaX) > 2) {
                onDragEnd(clip.id, clip.start, trackIndex, { type: type === 'start' ? 'resize-start' : 'resize-end', deltaX });
            }
            window.removeEventListener('mouseup', onUp);
        };

        window.addEventListener('mouseup', onUp);
    };

    return (
        <motion.div
            drag="x"
            dragConstraints={{ left: -startPx }}
            dragMomentum={false}
            dragElastic={0}
            onDragStart={() => { }}
            onDrag={handleDrag}
            onDragEnd={handleDragEndInternal}
            onClick={(e) => { e.stopPropagation(); selectShot(clip.id); }}
            className={clsx(
                "absolute top-1 bottom-1 rounded-lg border-2 cursor-pointer group bg-[#151515] overflow-visible",
                isSelected
                    ? "border-milimo-500 shadow-lg shadow-milimo-500/20 z-20"
                    : "border-white/10 hover:border-white/30 z-10 opacity-90 hover:opacity-100"
            )}
            style={{
                left: startPx,
                width: widthPx,
            }}
            whileDrag={{ scale: 1.02, zIndex: 100, boxShadow: "0 10px 20px rgba(0,0,0,0.5)" }}
        >
            <div className="absolute inset-0 overflow-hidden rounded-md bg-zinc-900 pointer-events-none">
                {trackType === 'audio' ? (
                    <div className="w-full h-full bg-milimo-500/10 flex items-center justify-center border border-milimo-500/30 relative">
                        <div className="absolute inset-0 z-0">
                            <AudioClip clip={clip} zoom={zoom} />
                        </div>
                        <div className="z-10 flex items-center bg-black/40 px-2 py-1 rounded backdrop-blur-sm">
                            <Music size={12} className="text-milimo-400" />
                            <span className="ml-1 text-[9px] text-milimo-200 truncate max-w-[80px]">{clip.name}</span>
                        </div>
                    </div>
                ) : (
                    <>
                        {clip.thumbnail ? (
                            <img src={clip.thumbnail} className="w-full h-full object-cover opacity-80" alt="" />
                        ) : clip.shot?.videoUrl ? (
                            <video src={clip.shot.videoUrl} className="w-full h-full object-cover opacity-60" />
                        ) : (
                            <div className="w-full h-full bg-gradient-to-br from-white/5 to-transparent flex items-center justify-center">
                                <span className="text-[10px] text-white/30 truncate px-2">{clip.name}</span>
                            </div>
                        )}
                        {clip.shot?.statusMessage && (
                            <div className="absolute inset-0 flex items-center justify-center bg-black/60">
                                <span className="text-[9px] text-milimo-300 font-mono animate-pulse">{clip.shot.statusMessage}</span>
                            </div>
                        )}
                    </>
                )}
            </div>

            <div className="absolute top-1 left-1 px-1.5 py-0.5 bg-black/50 backdrop-blur rounded text-[9px] text-white/80 font-mono truncate max-w-full z-20 pointer-events-none">
                {clip.name} ({clip.shot?.numFrames}f)
            </div>

            {isSelected && (
                <>
                    <div
                        className="absolute left-0 top-0 bottom-0 w-3 -ml-1.5 cursor-ew-resize hover:bg-white/50 z-30 transition-colors rounded-l"
                        onMouseDown={(e) => startResize(e, 'start')}
                    />
                    <div
                        className="absolute right-0 top-0 bottom-0 w-3 -mr-1.5 cursor-ew-resize hover:bg-white/50 z-30 transition-colors rounded-r"
                        onMouseDown={(e) => startResize(e, 'end')}
                    />
                </>
            )}
        </motion.div>
    );
});
