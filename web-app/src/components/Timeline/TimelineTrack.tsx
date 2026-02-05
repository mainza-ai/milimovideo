import { useShallow } from 'zustand/react/shallow';
import { clsx } from 'clsx';
import { useTimelineStore } from '../../stores/timelineStore';
import { TimelineClip } from './TimelineClip';
import { Eye, EyeOff, Lock, Unlock, Volume2, VolumeX, Plus } from 'lucide-react';

interface Props {
    track: { id: number; type: 'video' | 'audio'; name: string };
    clips: any[];
    zoom: number;
    projectFps: number;
    allClips: any[];
    onSnap: (lines: number[]) => void;
    onDragClipEnd: (id: string, newStartFrame: number, trackIndex: number, action?: any) => void;
}

import { memo } from 'react';

export const TimelineTrack = memo(({ track, clips, zoom, projectFps, allClips, onSnap, onDragClipEnd }: Props) => {
    const selectedShotId = useTimelineStore(state => state.selectedShotId);

    const { toggleTrackMute, toggleTrackLock, toggleTrackHidden, addShot } = useTimelineStore(useShallow(state => ({
        toggleTrackMute: state.toggleTrackMute,
        toggleTrackLock: state.toggleTrackLock,
        toggleTrackHidden: state.toggleTrackHidden,
        addShot: state.addShot
    })));

    // Select specific track state
    const trackState = useTimelineStore(useShallow(state => state.trackStates[track.id] || { muted: false, locked: false, hidden: false }));

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();

        if (trackState.locked) return;

        const data = e.dataTransfer.getData('application/milimo-element');
        let asset: any = null;
        if (data) {
            asset = JSON.parse(data);
            if (asset.image_path) {
                asset.url = asset.image_path;
                asset.type = 'image';
                asset.filename = asset.name;
            }
            if (track.type === 'audio' && !asset.type) {
                asset.type = 'audio';
            }
        }

        if (!asset) return;

        const rect = e.currentTarget.getBoundingClientRect();
        const offsetX = e.clientX - rect.left;
        const time = Math.max(0, offsetX / zoom);
        const startFrame = Math.round(time * projectFps);

        const shotConfig: any = {
            trackIndex: track.id,
        };
        if (track.id > 0) shotConfig.startFrame = startFrame;

        addShot(shotConfig);

        setTimeout(() => {
            const { selectedShotId, updateShot, addConditioningToShot, project } = useTimelineStore.getState();
            if (selectedShotId) {
                const shot = project.shots.find(s => s.id === selectedShotId);
                if (shot && shot.id !== 'shot-init') {
                    const isAudio = track.type === 'audio' || asset.type === 'audio';
                    const defaultFrames = isAudio ? (25 * 10) : (asset.type === 'video' ? 121 : 49);

                    updateShot(selectedShotId, {
                        prompt: `Shot using ${asset.filename || 'Visual'}`,
                        videoUrl: asset.url,
                        thumbnailUrl: asset.thumbnail || (asset.type === 'image' ? asset.url : undefined),
                        numFrames: defaultFrames
                    });

                    if (!isAudio) {
                        const item: any = {
                            type: asset.type === 'video' ? 'video' : 'image',
                            path: asset.url,
                            frameIndex: 0,
                            strength: 1.0
                        };
                        addConditioningToShot(selectedShotId, item);
                    }
                }
            }
        }, 50);
    };

    return (
        <div className="flex border-b border-white/5">
            <div className="w-32 min-w-[128px] border-r border-white/5 bg-[#111] p-2 flex flex-col justify-between sticky left-0 z-20">
                <div className="text-[10px] text-white/50 font-mono font-bold truncate" title={track.name}>
                    {track.name}
                </div>

                <div className="flex items-center gap-2">
                    <button
                        onClick={() => track.type === 'audio' ? toggleTrackMute(track.id) : toggleTrackHidden(track.id)}
                        className={clsx("p-1 rounded hover:bg-white/10", (trackState.muted || trackState.hidden) ? "text-red-400" : "text-white/40")}
                        title={track.type === 'audio' ? "Mute" : "Hide Track"}
                    >
                        {track.type === 'audio'
                            ? (trackState.muted ? <VolumeX size={12} /> : <Volume2 size={12} />)
                            : (trackState.hidden ? <EyeOff size={12} /> : <Eye size={12} />)
                        }
                    </button>

                    <button
                        onClick={() => toggleTrackLock(track.id)}
                        className={clsx("p-1 rounded hover:bg-white/10", trackState.locked ? "text-milimo-400" : "text-white/40")}
                        title={trackState.locked ? "Unlock Track" : "Lock Track"}
                    >
                        {trackState.locked ? <Lock size={12} /> : <Unlock size={12} />}
                    </button>
                </div>
            </div>

            <div
                className={clsx(
                    "relative flex-1 h-24 transition-colors",
                    trackState.locked ? "bg-red-900/5 pattern-diagonal-lines" : "hover:bg-white/[0.02]"
                )}
                onDragOver={(e) => { e.preventDefault(); e.dataTransfer.dropEffect = 'copy'; }}
                onDrop={handleDrop}
                style={{ opacity: trackState.hidden ? 0.3 : 1 }}
            >
                {clips.map(clip => (
                    <TimelineClip
                        key={clip.id}
                        clip={clip}
                        trackType={track.type}
                        trackIndex={track.id}
                        zoom={zoom}
                        isSelected={selectedShotId === clip.id}
                        allClips={allClips}
                        onSnap={onSnap}
                        onDragEnd={onDragClipEnd}
                    />
                ))}

                {/* Add Shot Button for Main Track */}
                {track.id === 0 && (
                    <div
                        className="absolute top-0 bottom-0 flex items-center justify-center border-l border-white/10 hover:bg-white/5 transition-colors group cursor-pointer"
                        style={{
                            left: clips.length > 0
                                ? (clips[clips.length - 1].start + clips[clips.length - 1].duration) * zoom
                                : 0,
                            width: 100, // Fixed width button
                        }}
                        onClick={(e) => {
                            e.stopPropagation();
                            addShot({ trackIndex: 0 });
                        }}
                        title="Add Empty Shot"
                    >
                        <div className="w-8 h-8 rounded-full bg-white/10 flex items-center justify-center group-hover:bg-milimo-500 group-hover:text-black transition-colors">
                            <Plus size={16} />
                        </div>
                        <span className="ml-2 text-xs font-bold text-white/50 group-hover:text-white transition-colors">Add Shot</span>
                    </div>
                )}
            </div>
        </div>
    );
});
