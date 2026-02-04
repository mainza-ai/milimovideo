import { useState, useMemo, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Play, Pause, Plus, SkipBack } from 'lucide-react';
import { clsx } from 'clsx';
import { useTimelineStore } from '../../stores/timelineStore';

// Types for Timeline (Internal view models)
interface Track {
    id: number;
    type: 'video' | 'audio';
    name: string;
}

// Auto-Save Hook
const useAutoSave = (project: any, saveProject: () => Promise<void>) => {
    const timeoutRef = useRef<any>(null);
    const lastSavedState = useRef<string>(JSON.stringify(project));

    useEffect(() => {
        const currentStr = JSON.stringify(project);
        if (currentStr === lastSavedState.current) return;

        if (timeoutRef.current) clearTimeout(timeoutRef.current);

        timeoutRef.current = setTimeout(async () => {
            console.log("Auto-Saving Project...");
            await saveProject();
            lastSavedState.current = JSON.stringify(project);
        }, 2000); // 2s debounce

        return () => {
            if (timeoutRef.current) clearTimeout(timeoutRef.current);
        };
    }, [project, saveProject]);
};


export const VisualTimeline = () => {
    const {
        project, selectedShotId, selectShot,
        isPlaying, setIsPlaying, reorderShots, addShot,
        currentTime, setCurrentTime,
        saveProject
    } = useTimelineStore();

    useAutoSave(project, saveProject);

    const [zoom, setZoom] = useState(20); // pixels per second
    const scrollContainerRef = useRef<HTMLDivElement>(null);

    // Convert linear shots list to timeline clips on Track 0
    // We assume shots are sequential for now.
    const clips = useMemo(() => {
        let currentTime = 0;
        return project.shots.map(shot => {
            const duration = shot.numFrames / (project.fps || 25);
            const clip = {
                id: shot.id,
                start: currentTime,
                duration: duration,
                track: 0,
                name: shot.prompt || 'Untitled Shot',
                thumbnail: shot.thumbnailUrl, // Use static thumbnail if available
                shot: shot // Keep ref
            };
            currentTime += duration;
            return clip;
        });
    }, [project.shots, project.fps]);

    const totalDuration = clips.reduce((acc, c) => Math.max(acc, c.start + c.duration), 0);
    const tracks: Track[] = [
        { id: 0, type: 'video', name: 'V1 (Main)' },
        // { id: 1, type: 'video', name: 'V2 (Overlay)' }, // Future
    ];

    // Helpers
    const formatTime = (seconds: number) => {
        const m = Math.floor(seconds / 60);
        const s = Math.floor(seconds % 60);
        const f = Math.floor((seconds % 1) * (project.fps || 25));
        return `${m}:${s.toString().padStart(2, '0')}:${f.toString().padStart(2, '0')}`;
    };

    const handleSeek = (e: React.MouseEvent) => {
        const rect = e.currentTarget.getBoundingClientRect();
        const scrollLeft = e.currentTarget.scrollLeft;
        const x = e.clientX - rect.left + scrollLeft;
        const time = Math.max(0, x / zoom);
        setCurrentTime(time);
    };

    const handleDragEnd = (clip: any, info: any) => {
        const movePx = info.offset.x;
        // Ignore clicks/small moves
        if (Math.abs(movePx) < 5) return;

        const myIndex = clips.findIndex(c => c.id === clip.id);
        if (myIndex === -1) return;

        const newX = clip.start * zoom + movePx;
        const myNewCenter = newX + (clip.duration * zoom) / 2;

        // Calculate the new index by comparing against other clips
        const otherClips = clips.filter(c => c.id !== clip.id);
        let newIndex = 0;

        for (const c of otherClips) {
            const cCenter = (c.start + c.duration / 2) * zoom;
            if (myNewCenter > cCenter) {
                newIndex++;
            }
        }

        if (newIndex !== myIndex) {
            reorderShots(myIndex, newIndex);
        }
    };

    const handleDragOver = (e: React.DragEvent) => {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'copy';
    };

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();

        let data = e.dataTransfer.getData('application/milimo-element');
        let asset: any = null;

        if (data) {
            asset = JSON.parse(data);
            // Element logic
            if (asset.image_path) {
                asset.url = asset.image_path;
                asset.type = 'image';
                asset.filename = asset.name; // Use name as filename for prompt
            }
        } else {
            data = e.dataTransfer.getData('application/json');
            if (data) asset = JSON.parse(data);
        }

        if (!asset) return;

        try {
            // Create new shot
            addShot();

            // Get the new shot (it's auto-selected by addShot)
            const { selectedShotId, updateShot, addConditioningToShot } = useTimelineStore.getState();
            if (!selectedShotId) return;

            // Configure shot
            updateShot(selectedShotId, {
                prompt: `Shot using ${asset.filename}`,
                numFrames: asset.type === 'video' ? 121 : 49, // Default duration
                videoUrl: asset.url, // Preview the asset immediately
                thumbnailUrl: asset.thumbnail || (asset.type === 'image' ? asset.url : undefined) // Set thumbnail
            });

            // Add conditioning
            const item: any = {
                type: asset.type === 'video' ? 'video' : 'image',
                path: asset.url,
                frameIndex: 0,
                strength: 1.0
            };
            addConditioningToShot(selectedShotId, item);

        } catch (err) {
            console.error("Drop failed", err);
        }
    };

    return (
        <div
            className="flex flex-col h-full bg-[#0a0a0a] border-t border-white/5 select-none"
            onDragOver={handleDragOver}
            onDrop={handleDrop}
        >
            {/* Toolbar */}
            <div className="h-10 flex items-center px-4 border-b border-white/5 bg-white/5 justify-between shrink-0">
                <div className="flex items-center gap-4">
                    <button
                        onClick={() => {
                            setCurrentTime(0);
                            if (scrollContainerRef.current) {
                                scrollContainerRef.current.scrollTo({ left: 0, behavior: 'smooth' });
                            }
                        }}
                        className="text-white hover:text-milimo-400 focus:outline-none"
                        title="Reset Playhead"
                    >
                        <SkipBack size={16} fill="currentColor" />
                    </button>
                    <button
                        onClick={() => setIsPlaying(!isPlaying)}
                        className="text-white hover:text-milimo-400 focus:outline-none"
                    >
                        {isPlaying ? <Pause size={16} fill="currentColor" /> : <Play size={16} fill="currentColor" />}
                    </button>

                    <button
                        onClick={addShot}
                        className="flex items-center gap-1.5 px-2 py-1 bg-white/10 hover:bg-white/20 rounded text-[10px] font-medium transition-colors"
                    >
                        <Plus size={12} />
                        Add Shot
                    </button>

                    <div className="text-xs font-mono text-milimo-300">
                        {formatTime(currentTime)} / {formatTime(totalDuration)}
                    </div>
                </div>

                <div className="flex items-center gap-2">
                    <span className="text-[10px] text-white/40">Zoom</span>
                    <input
                        type="range" min="5" max="100"
                        value={zoom} onChange={(e) => setZoom(parseInt(e.target.value))}
                        className="w-20 accent-milimo-500 h-1 bg-white/10 rounded-lg appearance-none cursor-pointer"
                    />
                </div>
            </div>

            {/* Timeline Area */}
            <div className="flex-1 flex overflow-hidden relative">
                {/* Track Headers */}
                <div className="w-32 bg-[#111] border-r border-white/5 min-w-[128px] z-10 flex flex-col">
                    {tracks.map(track => (
                        <div key={track.id} className="h-24 border-b border-white/5 flex items-center px-3 text-xs text-white/50 font-mono tracking-wider">
                            {track.name}
                        </div>
                    ))}
                </div>

                {/* Tracks Container */}
                <div
                    ref={scrollContainerRef}
                    className="flex-1 overflow-x-auto overflow-y-hidden relative bg-[#0f0f0f] custom-scrollbar cursor-pointer"
                    onClick={handleSeek}
                >
                    {/* Time/Grid Ruler Background */}
                    <div
                        className="absolute inset-0 pointer-events-none z-0"
                        style={{
                            backgroundImage: 'linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px)',
                            backgroundSize: `${zoom}px 100%`,
                            width: `${Math.max(totalDuration * zoom + 500, 2000)}px`
                        }}
                    />

                    {/* Track Lanes */}
                    <div className="relative z-10" style={{ width: `${Math.max(totalDuration * zoom + 500, 2000)}px` }}>
                        {tracks.map(track => (
                            <div key={track.id} className="h-24 border-b border-white/5 relative group">
                                {clips.filter(c => c.track === track.id).map(clip => {
                                    const isSelected = selectedShotId === clip.id;
                                    return (
                                        <motion.div
                                            key={clip.id}
                                            drag="x"
                                            dragMomentum={false}
                                            dragElastic={0}
                                            dragConstraints={{ left: -(clip.start * zoom) }}
                                            onDragEnd={(_, info) => handleDragEnd(clip, info)}
                                            onClick={(e) => { e.stopPropagation(); selectShot(clip.id); }}
                                            className={clsx(
                                                "absolute top-1 bottom-1 rounded-lg border-2 cursor-pointer group bg-[#151515]",
                                                isSelected
                                                    ? "border-milimo-500 shadow-lg shadow-milimo-500/20 z-20"
                                                    : "border-white/10 hover:border-white/30 z-10 opacity-90 hover:opacity-100"
                                            )}
                                            style={{
                                                left: clip.start * zoom,
                                                width: clip.duration * zoom,
                                            }}
                                            onDragOver={(e) => {
                                                e.preventDefault();
                                                e.stopPropagation();
                                                e.dataTransfer.dropEffect = 'copy';
                                                // Highlight effect? (handled by hover classes possibly, or add specific state)
                                                e.currentTarget.style.borderColor = '#22c55e'; // Green highlight
                                            }}
                                            onDragLeave={(e) => {
                                                e.preventDefault();
                                                e.stopPropagation();
                                                e.currentTarget.style.borderColor = isSelected ? '#22c55e' : 'rgba(255,255,255,0.1)';
                                            }}
                                            onDrop={(e) => {
                                                e.preventDefault();
                                                e.stopPropagation();
                                                e.currentTarget.style.borderColor = isSelected ? '#22c55e' : 'rgba(255,255,255,0.1)';

                                                const data = e.dataTransfer.getData('application/milimo-element'); // Try element first
                                                let asset: any = null;

                                                if (data) {
                                                    asset = JSON.parse(data);
                                                    // Element needs image_path to be conditioning
                                                    if (!asset.image_path) {
                                                        console.warn("Element has no visual");
                                                        return;
                                                    }
                                                    asset.url = asset.image_path;
                                                    asset.type = 'image'; // Elements are images for conditioning
                                                } else {
                                                    // Try generic JSON (from Images tab?)
                                                    const generic = e.dataTransfer.getData('application/json');
                                                    if (generic) asset = JSON.parse(generic);
                                                }

                                                if (!asset) return;

                                                // Calculate Frame Index
                                                const rect = e.currentTarget.getBoundingClientRect();
                                                const offsetX = e.clientX - rect.left;
                                                const pct = offsetX / rect.width;
                                                const frameIndex = Math.floor(pct * (clip.shot.numFrames - 1));

                                                console.log(`Dropped asset ${asset.filename || asset.name} @ frame ${frameIndex}`);

                                                const { addConditioningToShot, patchShot } = useTimelineStore.getState();

                                                const item: any = {
                                                    type: asset.type === 'video' ? 'video' : 'image',
                                                    path: asset.url,
                                                    frameIndex: frameIndex,
                                                    strength: 1.0
                                                };

                                                // 1. Local Update
                                                addConditioningToShot(clip.id, item);

                                                // 2. Persist (Patch)
                                                // We need the UPDATED timeline to send to backend? 
                                                // Or backend patch merges? No, it usually replaces the field.
                                                // So we construct the new timeline array.
                                                const newTimeline = [...clip.shot.timeline, item]; // Approximation (doesn't have ID yet, but store added one)
                                                // Actually store's addConditioningToShot generated an ID. 
                                                // We should ideally get the updated shot from store.
                                                // But `patchShot` sends JSON.
                                                // Let's rely on the fact that `item` is what we added.
                                                // Backend expects list of dicts.
                                                patchShot(clip.id, { timeline: newTimeline });
                                            }}
                                            whileDrag={{ scale: 1.02, zIndex: 100, boxShadow: "0 10px 20px rgba(0,0,0,0.5)" }}
                                        >
                                            {/* Clip Content (Masked) */}
                                            <div className="absolute inset-0 overflow-hidden rounded-md bg-zinc-900">
                                                {/* Clip Content */}
                                                {clip.thumbnail ? (
                                                    <img
                                                        src={clip.thumbnail}
                                                        className="w-full h-full object-cover opacity-80 pointer-events-none"
                                                        alt=""
                                                        onError={(e) => {
                                                            console.warn("Thumbnail load failed:", clip.thumbnail);
                                                            // e.currentTarget.style.display = 'none'; // Don't hide completely, maybe show placeholder?
                                                            // Fallback to clear
                                                            e.currentTarget.style.opacity = '0';
                                                        }}
                                                        onLoad={() => console.log("Thumbnail loaded:", clip.thumbnail)}
                                                    />
                                                ) : clip.shot.videoUrl ? (
                                                    <video src={clip.shot.videoUrl} className="w-full h-full object-cover opacity-60 pointer-events-none" />
                                                ) : (
                                                    <div className="w-full h-full bg-gradient-to-br from-white/5 to-transparent flex items-center justify-center">
                                                        <span className="text-[10px] text-white/30 truncate px-2">{clip.name}</span>
                                                    </div>
                                                )}
                                            </div>

                                            {/* Label */}
                                            <div className="absolute top-1 left-1 px-1.5 py-0.5 bg-black/50 backdrop-blur rounded text-[9px] text-white/80 font-mono truncate max-w-full z-20 pointer-events-none">
                                                {clip.shot.numFrames}f
                                            </div>

                                            {/* Conditioning Markers */}
                                            <div className="absolute bottom-0 left-0 right-0 h-4 px-1 flex items-end pointer-events-none z-20">
                                                {clip.shot && clip.shot.timeline.map(item => {
                                                    const pct = (item.frameIndex / (clip.shot.numFrames - 1 || 1)) * 100;
                                                    return (
                                                        <div
                                                            key={item.id}
                                                            className="absolute bottom-1 w-3 h-3 rounded-full bg-milimo-500 border border-black shadow-sm transform -translate-x-1/2"
                                                            style={{ left: `${pct}%` }}
                                                            title={`${item.type} @ f${item.frameIndex}`}
                                                        />
                                                    );
                                                })}
                                            </div>
                                        </motion.div>
                                    );
                                })}
                            </div>
                        ))}
                    </div>

                    {/* Playhead */}
                    <div
                        className="absolute top-0 bottom-0 w-px bg-red-500 z-30 pointer-events-none transition-all duration-100 ease-linear"
                        style={{ left: currentTime * zoom }}
                    >
                        <div className="w-3 h-3 bg-red-500 -ml-1.5 transform rotate-45 -mt-1.5 shadow-sm" />
                    </div>
                </div>
            </div>
        </div>
    );
};
