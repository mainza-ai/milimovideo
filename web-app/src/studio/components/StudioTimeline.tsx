import { useMemo } from 'react';
import { Minus, Plus, Scissors } from 'lucide-react';
import { useShallow } from 'zustand/react/shallow';
import { useTimelineStore } from '../../stores/timelineStore';
import { computeTimelineLayout } from '../../utils/timelineUtils';
import { useStudioEditor } from '../state/editorStore';
import { formatDuration } from '../data/useStudioData';

const TRACKS = [
    { id: 0, label: 'V1', tone: 'var(--st-v1)' },
    { id: 1, label: 'V2', tone: 'var(--st-v2)' },
    { id: 2, label: 'A1', tone: 'var(--st-a1)' },
];

export function StudioTimeline() {
    const project = useTimelineStore((state) => state.project);
    const { currentTime, selectedShotId, selectShot, setCurrentTime } = useTimelineStore(useShallow((state) => ({
        currentTime: state.currentTime,
        selectedShotId: state.selectedShotId,
        selectShot: state.selectShot,
        setCurrentTime: state.setCurrentTime,
    })));
    const { timelineZoom, setTimelineZoom } = useStudioEditor();

    const layout = useMemo(() => computeTimelineLayout(project), [project]);
    const duration = Math.max(10, ...layout.map((clip) => clip.end), currentTime) + 2;
    const width = Math.max(900, duration * timelineZoom + 180);
    const ticks = useMemo(() => {
        const step = timelineZoom >= 80 ? 1 : timelineZoom >= 30 ? 5 : 10;
        const result: number[] = [];
        for (let second = 0; second <= duration; second += step) result.push(second);
        return result;
    }, [duration, timelineZoom]);

    const seek = (event: React.MouseEvent<HTMLDivElement>) => {
        const rect = event.currentTarget.getBoundingClientRect();
        const time = Math.max(0, (event.clientX - rect.left) / timelineZoom);
        setCurrentTime(time);
    };

    return (
        <section className="st-timeline" aria-label="Timeline">
            <div className="st-timeline-toolbar">
                <button type="button" className="st-button" data-size="sm" data-variant="ghost" onClick={() => setTimelineZoom(timelineZoom / 1.2)}>
                    <Minus size={12} />
                </button>
                <span className="st-mono" style={{ color: 'var(--st-muted)', fontSize: 10, minWidth: 42, textAlign: 'center' }}>
                    {Math.round(timelineZoom)} px/s
                </span>
                <button type="button" className="st-button" data-size="sm" data-variant="ghost" onClick={() => setTimelineZoom(timelineZoom * 1.2)}>
                    <Plus size={12} />
                </button>
                <span style={{ color: 'var(--st-faint)', fontSize: 10, marginLeft: 8 }}>
                    {layout.length} clips · {formatDuration(duration * (project.fps || 25), project.fps || 25)}
                </span>
                <span style={{ flex: 1 }} />
                <button type="button" className="st-button" data-size="sm" data-variant="ghost" title="Split lands in F2">
                    <Scissors size={12} /> Split
                </button>
            </div>
            <div className="st-timeline-scroll">
                <div className="st-timeline-inner" style={{ width: width + 110 }}>
                    <div className="st-timeline-ruler" style={{ marginLeft: 110, width: width }}>
                        {ticks.map((second) => (
                            <div
                                key={second}
                                className="st-timeline-tick"
                                style={{ left: second * timelineZoom }}
                            >
                                {formatDuration(second * (project.fps || 25), project.fps || 25)}
                            </div>
                        ))}
                    </div>

                    {TRACKS.map((track) => (
                        <div className="st-timeline-track" key={track.id}>
                            <div className="st-timeline-label">
                                <strong style={{ color: track.tone }}>{track.label}</strong>
                                <span>{track.id === 2 ? 'audio' : track.id === 1 ? 'overlay' : 'main'}</span>
                            </div>
                            <div
                                className="st-timeline-lane"
                                style={{ width }}
                                onClick={seek}
                                role="presentation"
                            >
                                {layout.filter((clip) => clip.track === track.id).map((clip) => (
                                    <button
                                        key={clip.id}
                                        type="button"
                                        className="st-timeline-clip"
                                        data-track={track.id}
                                        data-selected={clip.id === selectedShotId}
                                        style={{ left: clip.start * timelineZoom, width: Math.max(8, clip.duration * timelineZoom) }}
                                        onClick={(event) => {
                                            event.stopPropagation();
                                            selectShot(clip.id);
                                            setCurrentTime(clip.start);
                                        }}
                                        title={`${clip.shot.action || clip.shot.prompt || 'Shot'} · ${formatDuration((clip.shot.numFrames || 0), project.fps || 25)}`}
                                    >
                                        <span className="st-timeline-clip-title">
                                            {clip.shot.action || clip.shot.prompt || 'Shot'}
                                        </span>
                                        <span className="st-timeline-clip-meta st-mono">{clip.shot.numFrames}f</span>
                                    </button>
                                ))}
                            </div>
                        </div>
                    ))}

                    <div
                        className="st-playhead"
                        style={{ left: 110 + currentTime * timelineZoom }}
                        aria-hidden="true"
                    />
                </div>
            </div>
        </section>
    );
}
