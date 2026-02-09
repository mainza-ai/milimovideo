import { useState, useMemo, useRef, useEffect, useCallback } from 'react';
import { Play, Pause, SkipBack, Scissors } from 'lucide-react';
import { useTimelineStore } from '../../stores/timelineStore';
import { TimelineTrack } from './TimelineTrack';
import { TimeDisplay } from './TimeDisplay';
import { PlaybackEngine } from '../Player/PlaybackEngine';
import { Playhead } from './Playhead';
import { computeTimelineLayout } from '../../utils/timelineUtils';

// Auto-Save Hook
const useAutoSave = (project: any, saveProject: () => Promise<void>) => {
    const timeoutRef = useRef<any>(null);


    useEffect(() => {
        // Debounce auto-save on any project change.
        // We skip the expensive JSON.stringify(project) check here.
        // If the project reference changed, we assume it's worth resetting the timer.
        // The actual saveProject implementation can handle "no changes" if it wants,
        // or we just accept a cheap PUT every 2s during editing.

        if (timeoutRef.current) clearTimeout(timeoutRef.current);

        timeoutRef.current = setTimeout(async () => {
            console.log("Auto-Saving Project...");
            await saveProject();
        }, 2000); // 2s debounce

        return () => {
            if (timeoutRef.current) clearTimeout(timeoutRef.current);
        };
    }, [project, saveProject]);
};

const TRACKS = [
    { id: 0, type: 'video' as const, name: 'V1 (Main)' },
    { id: 1, type: 'video' as const, name: 'V2 (Overlay)' },
    { id: 2, type: 'audio' as const, name: 'A1 (Audio)' },
];

export const VisualTimeline = () => {
    const project = useTimelineStore(state => state.project);
    const selectedShotId = useTimelineStore(state => state.selectedShotId);
    const isPlaying = useTimelineStore(state => state.isPlaying);
    const setIsPlaying = useTimelineStore(state => state.setIsPlaying);
    const setCurrentTime = useTimelineStore(state => state.setCurrentTime);
    const saveProject = useTimelineStore(state => state.saveProject);
    const splitShot = useTimelineStore(state => state.splitShot);
    const transientDuration = useTimelineStore(state => state.transientDuration);

    useAutoSave(project, saveProject);

    const [zoom, setZoom] = useState(20); // pixels per second
    const scrollContainerRef = useRef<HTMLDivElement>(null);
    const [snapLines, setSnapLines] = useState<number[]>([]);

    // Convert linear shots list to timeline clips using centralized utility
    const clips = useMemo(() => {
        return computeTimelineLayout(project);
    }, [project]);

    const totalDuration = clips.reduce((acc, c) => Math.max(acc, c.start + c.duration), 0);

    // Effective Duration (Project or Temporary Drag expansion)
    const effectiveDuration = Math.max(totalDuration, transientDuration || 0);


    // TRACKS moved outside component

    // Helpers

    // Global handlers
    const handleDragClipEnd = useCallback((id: string, newStartFrame: number, trackIndex: number, action?: any) => {
        const { project } = useTimelineStore.getState();
        // clips is available in component scope and is a dependency

        const fps = project.fps || 25;
        // Wait, clips is computed in VisualTimeline and passed to Track.
        // If we use getState(), we get Raw Shots.
        // But the logic below uses `clips` (the computed view).

        // CORRECTION: We need to pass `clips` to the dependency array.
        // BUT `clips` changes when `project` changes.
        // If `clips` changes, the callback changes.
        // If callback changes, `TimelineTrack` re-renders.
        // If `TimelineTrack` re-renders, `TimelineClip` re-renders.

        // HOWEVER: During drag, `clips` SHOULD BE STABLE unless we are updating the store 60fps.
        // The issue is `TimelineClip` calls `useTimelineStore.getState().setTransientDuration()`.
        // This causes `VisualTimeline` to re-render.
        // `clips` uses `useMemo(() => ..., [project.shots])`.
        // `transientDuration` does NOT affect `clips` array reference.
        // SO `clips` is stable.

        // THEREFORE: If we wrap logic in useCallback with `[clips, project.fps]`, it should be stable during transient drag.

        // Let's verify:
        // `handleDragClipEnd` is called on DRAG END. Not during drag.
        // So it doesn't matter if it changes?
        // WAIT. `TimelineClip` receives `onDragEnd={handleDragClipEnd}`.
        // If `VisualTimeline` re-renders (due to transientDuration), and `handleDragClipEnd` is re-created,
        // then `TimelineClip` sees a new prop, so it re-renders.

        // So we MUST memoize `handleDragClipEnd`.
        // `clips` is stable during drag (transientDuration change doesn't change project.shots).
        // `zoom` changes? No.

        // Refactoring to use internal logic or standard refs if needed?
        // No, standard useCallback should work if `clips` is stable.



        if (action?.type === 'resize-start') {
            const clip = clips.find(c => c.id === id);
            if (!clip) return;
            const deltaSeconds = action.deltaX / zoom;
            const deltaFrames = Math.round(deltaSeconds * fps);

            // Limit: trimIn cannot exceed duration or be < 0
            // But trimIn + trimOut must be < numFrames
            const currentTrimIn = clip.shot.trimIn || 0;
            const maxTrimIn = (clip.shot.numFrames || 0) - (clip.shot.trimOut || 0) - 25; // Keep 1s min?
            const newTrimIn = Math.min(Math.max(0, currentTrimIn + deltaFrames), maxTrimIn);

            // Also need to adjust startFrame if not magnetic V1
            const updates: any = { trimIn: newTrimIn };
            if (trackIndex > 0) {
                // For Free Track, if we trim start IN (positive delta), visual start moves RIGHT
                // So startFrame increases
                // But wait, user dragged Handle. 
                // If dragging handle RIGHT, we are cutting the start. Content starts later.
                // Visual Block Start moves RIGHT.
                // So StartFrame += Delta.
                const currentStart = clip.shot.startFrame || 0;
                const newStart = currentStart + deltaFrames;
                updates.startFrame = Math.max(0, newStart);
            }

            useTimelineStore.getState().patchShot(id, updates);
            return;
        }

        if (action?.type === 'resize-end') {
            const clip = clips.find(c => c.id === id);
            if (!clip) return;
            const deltaSeconds = action.deltaX / zoom;
            const deltaFrames = Math.round(deltaSeconds * fps);

            // Dragging Right Handle RIGHT (positive) -> Less TrimOut (Longer)
            // Wait, Right handle controls duration.
            // visual ends = start + duration.
            // Dragging Right Handle changes duration.
            // If delta > 0, we are extending. TrimOut decreases.

            const currentTrimOut = clip.shot.trimOut || 0;
            // newTrimOut = currentTrimOut - deltaFrames
            // Ensure trimOut >= 0
            // Also user might want to LOOP? For now just limit to 0 trimOut (full length)

            const newTrimOut = Math.max(0, currentTrimOut - deltaFrames);

            // If we hit 0, we can't extend further unless we loop (feature for later)
            // Or if we generated more frames?

            useTimelineStore.getState().patchShot(id, { trimOut: newTrimOut });
            return;
        }

        if (trackIndex === 0) {
            // V1 Reorder Logic
            const v1Clips = clips.filter(c => c.track === 0).sort((a, b) => a.start - b.start);
            const myIndex = v1Clips.findIndex(c => c.id === id);
            if (myIndex === -1) return;

            // Calculate new center based on dragged position
            const newStartTime = newStartFrame / (project.fps || 25);
            const me = v1Clips[myIndex];
            const myDuration = me.duration;
            const myNewCenter = newStartTime + (myDuration / 2);

            let newIndex = 0;
            const otherClips = v1Clips.filter(c => c.id !== id);

            for (const c of otherClips) {
                const cCenter = c.start + (c.duration / 2);
                if (myNewCenter > cCenter) {
                    newIndex++;
                }
            }

            if (newIndex !== myIndex) {
                // Connected Clips Logic: Move attached V2/A1 clips
                const movedClip = v1Clips[myIndex];

                const v2a1Clips = clips.filter(c => c.track > 0);

                // Helper to find dominant V1 clip for a free clip
                const getParentV1 = (freeClip: any) => {
                    const center = freeClip.start + (freeClip.duration / 2);
                    return v1Clips.find(v1 => center >= v1.start && center < (v1.start + v1.duration));
                };

                // Calculate shifts
                // Ideally we simulate the new state to know exact time differences
                // But for simple "Magnetic" feel, we can just apply delta.

                // Simplify: Just reorder V1 first. 
                // The issue is if we reorder V1, the V2 clips don't move automatically.
                // We need to valid "Before" and "After" times for detecting the delta.
                // This is complex to do purely client-side without re-running the layout engine.

                // Alternative: Just reorder. User manually moves audio.
                // BUT user requested "Magnetic Behaviors".

                // Let's iterate and find clips attached to the MOVED clip.
                const connectedClips = v2a1Clips.filter(c => getParentV1(c)?.id === movedClip.id);

                // Calculate where the MOVED clip ends up.
                // Sum of durations of clips before newIndex
                let newStart = 0;
                // v1Clips is sorted by old position.
                // We construct the NEW order.
                const newOrder = [...v1Clips];
                newOrder.splice(myIndex, 1);
                newOrder.splice(newIndex, 0, movedClip);

                // Find new start of movedClip
                for (let i = 0; i < newIndex; i++) {
                    newStart += newOrder[i].duration;
                }

                const delta = newStart - movedClip.start;

                // Apply delta to connected clips
                connectedClips.forEach(c => {
                    const newFrame = Math.round((c.start + delta) * fps);
                    useTimelineStore.getState().moveShotToValues(c.id, c.track, newFrame);
                });

                useTimelineStore.getState().reorderShots(myIndex, newIndex);
            }

        } else {
            useTimelineStore.getState().moveShotToValues(id, trackIndex, Math.max(0, newStartFrame));
        }
    }, [clips, project.fps, zoom]); // project.fps is stable. zoom is needed for calc? Ah, deltaX uses zoom.

    const handleSeek = useCallback((e: React.MouseEvent) => {
        // Only seek if clicking on the ruler area or background, 
        // Component clicks usually propagate unless stopped.
        if ((e.target as HTMLElement).closest('.clip-content')) return;

        // Since we are clicking inside scrollContainerRef, offset X is e.nativeEvent.offsetX if using raw,
        // but let's be safe with rects.
        const rect = e.currentTarget.getBoundingClientRect();
        // We need to account that this handler is on the Container which scrolls.
        // e.clientX is viewport X.
        // rect.left is viewport left of container (0 usually for sidebar)
        // scrollLeft is scroll.
        // const x = e.clientX - rect.left + e.currentTarget.scrollLeft - 128; // 128 is offset for Sidebar width if it's included in scroll?
        // Wait, sidebar is sticky? 
        // Layout: Sidebar (128px) | Track Content (Flex 1)
        // The click listener should be on the Track Content area mostly? 

        // Let's refine seek logic:
        // Ruler click is best. Background click is okay.

        // Simplified:
        // We attach click to the "Tracks Container".
        // It has sticky sidebar. 
        // If click is on sidebar, ignore.
        if (e.clientX < (rect.left + 128)) return;

        const scrollLeft = e.currentTarget.scrollLeft;
        const relativeX = e.clientX - rect.left - 128 + scrollLeft;
        // -128 because the sidebar is visually there but 'sticky'.
        // Ideally we separate scroll container for timeline only.
        // Current CSS: Sidebar is "sticky left-0".

        const time = Math.max(0, relativeX / zoom);
        useTimelineStore.getState().setCurrentTime(time);
    }, [zoom]);

    // Zoom & Pan
    useEffect(() => {
        const container = scrollContainerRef.current;
        if (!container) return;

        const handleWheel = (e: WheelEvent) => {
            // Ctrl/Cmd + Wheel = Zoom
            if (e.ctrlKey || e.metaKey) {
                e.preventDefault();
                const delta = e.deltaY > 0 ? 0.9 : 1.1; // Out : In
                setZoom(z => Math.min(Math.max(z * delta, 5), 300));
            }
            // Default behavior handles Vertical/Horizontal scroll
        };
        container.addEventListener('wheel', handleWheel, { passive: false });
        return () => container.removeEventListener('wheel', handleWheel);
    }, []);

    const handleSplit = async () => {
        if (!selectedShotId) return;
        const clip = clips.find(c => c.id === selectedShotId);
        if (!clip) return;
        const relativeTime = useTimelineStore.getState().currentTime - clip.start;
        if (relativeTime <= 0 || relativeTime >= clip.duration) return;
        const fps = project.fps || 25;
        const splitFrame = Math.round(relativeTime * fps);
        await splitShot(selectedShotId, splitFrame);
    };

    return (
        <div className="flex flex-col h-full bg-[#0a0a0a] border-t border-white/5 select-none text-white">
            <PlaybackEngine />
            {/* Toolbar */}
            <div className="h-10 flex items-center px-4 border-b border-white/5 bg-white/5 justify-between shrink-0 z-30 relative">
                <div className="flex items-center gap-4">
                    <button onClick={() => { setCurrentTime(0); scrollContainerRef.current?.scrollTo({ left: 0 }); }} className="text-white hover:text-milimo-400">
                        <SkipBack size={16} fill="currentColor" />
                    </button>
                    <button onClick={() => setIsPlaying(!isPlaying)} className="text-white hover:text-milimo-400">
                        {isPlaying ? <Pause size={16} fill="currentColor" /> : <Play size={16} fill="currentColor" />}
                    </button>
                    <TimeDisplay projectFps={project.fps} totalDuration={totalDuration} />
                </div>
                <div className="flex items-center gap-2">
                    <button onClick={handleSplit} className="text-white/50 hover:text-white" disabled={!selectedShotId}>
                        <Scissors size={14} />
                    </button>
                    <input type="range" min="5" max="300" value={zoom} onChange={(e) => setZoom(parseInt(e.target.value))} className="w-20 accent-milimo-500 h-1 bg-white/10 rounded-lg appearance-none cursor-pointer" />
                </div>
            </div>

            {/* Main Area */}
            <div
                ref={scrollContainerRef}
                className="flex-1 overflow-x-auto overflow-y-auto relative bg-[#0a0a0a] custom-scrollbar"
                onClick={handleSeek}
            >
                {/* Tracks Container */}
                <div className="min-w-full inline-block pb-32 relative z-10">
                    {/* Timeline Header Spacer */}
                    <div className="flex sticky top-0 z-30 bg-[#111] border-b border-white/5 h-6">
                        <div className="w-32 min-w-[128px] border-r border-white/5 sticky left-0 bg-[#111] z-40"></div>
                        <div className="flex-1 relative">
                            {/* Ruler Content */}
                            {/* ... logic for ruler can stay or be moved. Ideally ruler is top. */}
                        </div>
                    </div>

                    {TRACKS.map(track => (
                        <TimelineTrack
                            key={track.id}
                            track={track}
                            clips={clips.filter(c => c.track === track.id)}
                            zoom={zoom}
                            projectFps={project.fps || 25}
                            allClips={clips}
                            onSnap={setSnapLines}
                            onDragClipEnd={handleDragClipEnd}
                        />
                    ))}
                </div>

                {/* Grid Overlay (Z-Index 20 - Above Tracks Background, Below Clips?) 
                    Actually, if Tracks have transparent BG, this can stay behind.
                    BUT User wants it "Always Visible". 
                    If we put it Z-20, it covers Tracks. 
                    We need pointer-events-none.
                */}
                <div
                    className="absolute top-0 bottom-0 left-[128px] pointer-events-none z-20 mix-blend-overlay"
                    style={{
                        backgroundImage: 'linear-gradient(90deg, rgba(255,255,255,0.08) 1px, transparent 1px)',
                        backgroundSize: `${zoom}px 100%`,
                        width: `${Math.max(effectiveDuration * zoom + 500, 2000)}px`,
                        height: '100%'
                    }}
                />

                {/* Snap Lines Overlay */}
                {snapLines.map(t => (
                    <div
                        key={t}
                        className="absolute top-0 bottom-0 w-px bg-yellow-400 z-50 pointer-events-none"
                        style={{ left: 128 + (t * zoom) }}
                    />
                ))}

                {/* Playhead */}
                <Playhead zoom={zoom} />
            </div>
        </div >
    );
};
