import { useEffect, useMemo, useRef, useState } from 'react';
import { Columns2, Maximize, Pause, Play, SkipBack, SkipForward, SquareSplitHorizontal, Volume2 } from 'lucide-react';
import { useShallow } from 'zustand/react/shallow';
import { useTimelineStore } from '../../stores/timelineStore';
import { computeTimelineLayout } from '../../utils/timelineUtils';
import { PlaybackEngine } from '../../components/Player/PlaybackEngine';
import { StudioTimeline } from '../components/StudioTimeline';
import { formatDuration, useAssetUrl, useStudioTakes } from '../data/useStudioData';

export function EditSurface() {
    const project = useTimelineStore((state) => state.project);
    const { isPlaying, setIsPlaying, setCurrentTime, selectedShotId } = useTimelineStore(useShallow((state) => ({
        isPlaying: state.isPlaying,
        setIsPlaying: state.setIsPlaying,
        setCurrentTime: state.setCurrentTime,
        selectedShotId: state.selectedShotId,
    })));
    const videoRef = useRef<HTMLVideoElement>(null);
    const compareVideoRef = useRef<HTMLVideoElement>(null);
    const [compareMode, setCompareMode] = useState<'off' | 'side' | 'wipe'>('off');
    const [wipe, setWipe] = useState(50);
    const [guides, setGuides] = useState({ grid: false, safe: false, thirds: false });
    const [activeId, setActiveId] = useState<string | null>(() => {
        const state = useTimelineStore.getState();
        return state.selectedShotId || null;
    });
    const activeIdRef = useRef(activeId);

    const layout = useMemo(() => computeTimelineLayout(project), [project]);
    const totalDuration = useMemo(
        () => Math.max(0, ...layout.map((clip) => clip.end)),
        [layout],
    );

    const resolveActive = (time: number) => {
        const selected = selectedShotId ? layout.find((clip) => clip.id === selectedShotId) : null;
        if (selected) return selected;
        return layout.find((clip) => time >= clip.start && time < clip.end && clip.track === 0)
            || layout.find((clip) => time >= clip.start && time < clip.end && clip.track === 1)
            || null;
    };

    const active = selectedShotId
        ? layout.find((clip) => clip.id === selectedShotId) || null
        : layout.find((clip) => clip.id === activeId) || null;
    const mediaUrl = useAssetUrl(active?.shot.videoUrl);
    const isImage = !!mediaUrl?.match(/\.(jpg|jpeg|png|webp)$/i);
    const { takes } = useStudioTakes(active?.shot.id);
    const compareTake = takes.find((take) => !take.selected && take.url) || null;
    const compareUrl = useAssetUrl(compareTake?.url);
    const compareIsImage = !!compareUrl?.match(/\.(jpg|jpeg|png|webp)$/i);
    const fps = project.fps || 25;

    // The clock writes currentTime every frame. Sync media and the active clip
    // through a subscription instead of re-rendering the whole edit surface.
    useEffect(() => {
        const sync = (time: number, playing: boolean) => {
            const clip = resolveActive(time);
            const nextId = clip?.id ?? null;
            if (nextId !== activeIdRef.current) {
                activeIdRef.current = nextId;
                setActiveId(nextId);
            }
            if (!clip) return;
            const local = Math.max(0, (time - clip.start) + (clip.shot.trimIn || 0) / fps);

            const video = videoRef.current;
            if (video && mediaUrl && !isImage) {
                if (Math.abs(video.currentTime - local) > 0.3) {
                    try { video.currentTime = local; } catch { /* metadata not ready */ }
                }
                if (playing) void video.play().catch(() => {});
                else video.pause();
            }

            const compare = compareVideoRef.current;
            if (compare && compareUrl && !compareIsImage) {
                if (Math.abs(compare.currentTime - local) > 0.3) {
                    try { compare.currentTime = local; } catch { /* metadata not ready */ }
                }
                if (playing) void compare.play().catch(() => {});
                else compare.pause();
            }
        };

        const state = useTimelineStore.getState();
        sync(state.currentTime, state.isPlaying);
        const unsubscribe = useTimelineStore.subscribe((next, previous) => {
            if (next.currentTime !== previous.currentTime || next.isPlaying !== previous.isPlaying) {
                sync(next.currentTime, next.isPlaying);
            }
        });
        return unsubscribe;
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [compareIsImage, compareUrl, fps, isImage, layout, mediaUrl, selectedShotId]);

    return (
        <div className="st-edit">
            <PlaybackEngine />
            <div className="st-stage">
                <div className={`st-stage-frame ${compareMode !== 'off' ? 'st-stage-compare' : ''}`}>
                    {mediaUrl ? (
                        <div className="st-stage-primary">
                            {isImage ? (
                                <img src={mediaUrl} alt="" />
                            ) : (
                                <video ref={videoRef} src={mediaUrl} muted playsInline preload="metadata" />
                            )}
                        </div>
                    ) : (
                        <div className="st-stage-empty">
                            <Play size={22} />
                            <strong>No media in the cut</strong>
                            <span>Generate a shot or drop media on the V1 track.</span>
                        </div>
                    )}

                    {compareMode !== 'off' && compareUrl && (
                        <div
                            className={compareMode === 'side' ? 'st-stage-compare-side' : 'st-stage-compare-wipe'}
                            style={compareMode === 'wipe' ? { clipPath: `inset(0 ${100 - wipe}% 0 0)` } : undefined}
                        >
                            {compareIsImage ? (
                                <img src={compareUrl} alt="" />
                            ) : (
                                <video ref={compareVideoRef} src={compareUrl} muted playsInline preload="metadata" />
                            )}
                            <span className="st-compare-badge">
                                {compareMode === 'side' ? 'B · alternate take' : 'B'}
                            </span>
                        </div>
                    )}

                    {compareMode === 'wipe' && compareUrl && (
                        <div className="st-wipe-control">
                            <span>A</span>
                            <input
                                type="range"
                                min={0}
                                max={100}
                                value={wipe}
                                aria-label="Wipe position"
                                onChange={(event) => setWipe(Number(event.target.value))}
                            />
                            <span>B</span>
                        </div>
                    )}

                    {guides.grid && <div className="st-guide st-guide-grid" aria-hidden="true" />}
                    {guides.safe && <div className="st-guide st-guide-safe" aria-hidden="true" />}
                    {guides.thirds && <div className="st-guide st-guide-thirds" aria-hidden="true" />}

                    <div className="st-stage-overlay">
                        <span className="st-status-pill">V{active?.track === 1 ? 2 : 1}</span>
                        <span className="st-status-pill">{project.resolutionW}×{project.resolutionH}</span>
                        <span className="st-status-pill">{fps} fps</span>
                        {(['grid', 'safe', 'thirds'] as const).map((guide) => (
                            <button
                                key={guide}
                                type="button"
                                className="st-status-pill"
                                data-tone={guides[guide] ? 'good' : undefined}
                                onClick={() => setGuides({ ...guides, [guide]: !guides[guide] })}
                            >
                                {guide}
                            </button>
                        ))}
                    </div>
                </div>
            </div>

            <div className="st-transport">
                <button type="button" className="st-icon-button" onClick={() => setCurrentTime(0)} aria-label="Go to start">
                    <SkipBack size={15} />
                </button>
                <button type="button" className="st-button" data-variant="primary" onClick={() => setIsPlaying(!isPlaying)} aria-label={isPlaying ? 'Pause' : 'Play'}>
                    {isPlaying ? <Pause size={15} /> : <Play size={15} />}
                </button>
                <button
                    type="button"
                    className="st-icon-button"
                    onClick={() => {
                        const state = useTimelineStore.getState();
                        state.setCurrentTime(state.currentTime + 1);
                    }}
                    aria-label="Forward one second"
                >
                    <SkipForward size={15} />
                </button>
                <TimecodeDisplay totalDuration={totalDuration} fps={fps} />
                <span style={{ flex: 1 }} />
                <button
                    type="button"
                    className="st-button"
                    data-size="sm"
                    data-variant={compareMode === 'side' ? 'primary' : 'ghost'}
                    disabled={!compareTake}
                    title={compareTake ? 'Side-by-side with the previous take' : 'Generate a second take to compare'}
                    onClick={() => setCompareMode(compareMode === 'side' ? 'off' : 'side')}
                >
                    <Columns2 size={12} /> Side
                </button>
                <button
                    type="button"
                    className="st-button"
                    data-size="sm"
                    data-variant={compareMode === 'wipe' ? 'primary' : 'ghost'}
                    disabled={!compareTake}
                    title={compareTake ? 'Wipe between takes' : 'Generate a second take to compare'}
                    onClick={() => setCompareMode(compareMode === 'wipe' ? 'off' : 'wipe')}
                >
                    <SquareSplitHorizontal size={12} /> Wipe
                </button>
                <span className="st-status-pill"><Volume2 size={11} /> embedded audio mixed</span>
                <button type="button" className="st-icon-button" aria-label="Fullscreen (F2)">
                    <Maximize size={14} />
                </button>
            </div>

            <StudioTimeline />
        </div>
    );
}

function TimecodeDisplay({ totalDuration, fps }: { totalDuration: number; fps: number }) {
    const currentTime = useTimelineStore((state) => state.currentTime);
    return (
        <span className="st-timecode st-mono">
            {formatDuration(currentTime * fps, fps)} / {formatDuration(totalDuration * fps, fps)}
        </span>
    );
}
