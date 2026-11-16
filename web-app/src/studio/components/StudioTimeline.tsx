import { useMemo, useRef, useState } from 'react';
import {
    Minus, MousePointer2, Plus, Scissors, Trash2, MoveHorizontal, UnfoldHorizontal,
} from 'lucide-react';
import { useShallow } from 'zustand/react/shallow';
import { useTimelineStore, type Shot } from '../../stores/timelineStore';
import { computeTimelineLayout, type TimelineClip } from '../../utils/timelineUtils';
import { useStudioEditor } from '../state/editorStore';
import { formatDuration } from '../data/useStudioData';
import { bladeSourceFrame, rollCut, slipShot, trimShot } from '../timeline/ops';

const TRACKS = [
    { id: 0, label: 'V1', tone: 'var(--st-v1)' },
    { id: 1, label: 'V2', tone: 'var(--st-v2)' },
    { id: 2, label: 'A1', tone: 'var(--st-a1)' },
];

type Tool = 'select' | 'slip' | 'roll';

interface Gesture {
    kind: 'trim-in' | 'trim-out' | 'slip' | 'roll';
    shotId: string;
    startX: number;
    base: Shot;
    neighbor?: Shot;
    delta: number;
}

interface Preview {
    shotId: string;
    trimIn: number;
    trimOut: number;
    neighborId?: string;
    neighborTrimIn?: number;
    neighborTrimOut?: number;
}

export function StudioTimeline() {
    const project = useTimelineStore((state) => state.project);
    const {
        currentTime, selectedShotId, selectShot, setCurrentTime, updateShot,
        splitShot, deleteShot, addToast,
    } = useTimelineStore(useShallow((state) => ({
        currentTime: state.currentTime,
        selectedShotId: state.selectedShotId,
        selectShot: state.selectShot,
        setCurrentTime: state.setCurrentTime,
        updateShot: state.updateShot,
        splitShot: state.splitShot,
        deleteShot: state.deleteShot,
        addToast: state.addToast,
    })));
    const { timelineZoom, setTimelineZoom } = useStudioEditor();
    const [tool, setTool] = useState<Tool>('select');
    const [preview, setPreview] = useState<Preview | null>(null);
    const gesture = useRef<Gesture | null>(null);
    const fps = project.fps || 25;

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
        setCurrentTime(Math.max(0, (event.clientX - rect.left) / timelineZoom));
    };

    const v1Clips = useMemo(
        () => layout.filter((clip) => clip.track === 0).sort((a, b) => a.start - b.start),
        [layout],
    );

    const findClip = (shotId: string): TimelineClip | undefined =>
        layout.find((clip) => clip.id === shotId);

    const blade = async () => {
        const target = selectedShotId ? findClip(selectedShotId) : undefined;
        const clip = target && currentTime >= target.start && currentTime <= target.end
            ? target
            : v1Clips.find((item) => currentTime > item.start && currentTime < item.end);
        if (!clip) {
            addToast('Place the playhead inside a clip to blade it', 'info');
            return;
        }
        const frame = bladeSourceFrame(clip.shot, currentTime - clip.start, fps);
        if (frame == null) {
            addToast('The cut is too close to a clip edge', 'info');
            return;
        }
        selectShot(clip.id);
        await splitShot(clip.id, frame);
    };

    const rippleDelete = () => {
        if (!selectedShotId) {
            addToast('Select a clip to ripple-delete', 'info');
            return;
        }
        deleteShot(selectedShotId);
        addToast('Clip removed — V1 rippled', 'info');
    };

    const beginGesture = (
        event: React.PointerEvent,
        kind: Gesture['kind'],
        shot: Shot,
    ) => {
        event.stopPropagation();
        selectShot(shot.id);
        const neighbor = kind === 'roll' ? nextV1Shot(shot.id) : undefined;
        if (kind === 'roll' && !neighbor) {
            addToast('Roll needs a following V1 clip', 'info');
            return;
        }
        gesture.current = { kind, shotId: shot.id, startX: event.clientX, base: shot, neighbor, delta: 0 };
        setPreview({
            shotId: shot.id,
            trimIn: shot.trimIn || 0,
            trimOut: shot.trimOut || 0,
            neighborId: neighbor?.id,
            neighborTrimIn: neighbor?.trimIn || 0,
            neighborTrimOut: neighbor?.trimOut || 0,
        });
        (event.currentTarget as HTMLElement).setPointerCapture(event.pointerId);
    };

    const nextV1Shot = (shotId: string): Shot | undefined => {
        const index = v1Clips.findIndex((clip) => clip.id === shotId);
        return index >= 0 ? v1Clips[index + 1]?.shot : undefined;
    };

    const onGestureMove = (event: React.PointerEvent) => {
        const active = gesture.current;
        if (!active) return;
        const delta = Math.round(((event.clientX - active.startX) / timelineZoom) * fps);
        if (delta === active.delta) return;
        active.delta = delta;

        if (active.kind === 'trim-in' || active.kind === 'trim-out') {
            const result = trimShot(active.base, active.kind === 'trim-in' ? 'in' : 'out', delta);
            if (result) {
                setPreview({ shotId: active.shotId, ...result });
            }
        } else if (active.kind === 'slip') {
            const result = slipShot(active.base, delta);
            if (result) {
                setPreview({ shotId: active.shotId, ...result });
            }
        } else if (active.kind === 'roll' && active.neighbor) {
            const result = rollCut(active.base, active.neighbor, delta);
            if (result) {
                setPreview({
                    shotId: active.shotId,
                    ...result.left,
                    neighborId: active.neighbor.id,
                    neighborTrimIn: result.right.trimIn,
                    neighborTrimOut: result.right.trimOut,
                });
            }
        }
    };

    const endGesture = () => {
        const active = gesture.current;
        const pending = preview;
        gesture.current = null;
        setPreview(null);
        if (!active || !pending) return;
        if (active.kind === 'roll' && pending.neighborId) {
            updateShot(pending.shotId, { trimIn: pending.trimIn, trimOut: pending.trimOut });
            updateShot(pending.neighborId, {
                trimIn: pending.neighborTrimIn,
                trimOut: pending.neighborTrimOut,
            });
        } else {
            updateShot(pending.shotId, { trimIn: pending.trimIn, trimOut: pending.trimOut });
        }
    };

    const previewFor = (clipId: string) => (preview?.shotId === clipId ? preview : null);

    return (
        <section className="st-timeline" aria-label="Timeline">
            <div className="st-timeline-toolbar">
                <button
                    type="button"
                    className="st-button"
                    data-size="sm"
                    data-variant={tool === 'select' ? 'primary' : 'ghost'}
                    onClick={() => setTool('select')}
                    title="Select and trim"
                >
                    <MousePointer2 size={12} /> Select
                </button>
                <button
                    type="button"
                    className="st-button"
                    data-size="sm"
                    data-variant={tool === 'slip' ? 'primary' : 'ghost'}
                    onClick={() => setTool('slip')}
                    title="Drag a clip to slip its source window"
                >
                    <MoveHorizontal size={12} /> Slip
                </button>
                <button
                    type="button"
                    className="st-button"
                    data-size="sm"
                    data-variant={tool === 'roll' ? 'primary' : 'ghost'}
                    onClick={() => setTool('roll')}
                    title="Drag a clip to roll the cut with the next V1 clip"
                >
                    <UnfoldHorizontal size={12} /> Roll
                </button>
                <button type="button" className="st-button" data-size="sm" onClick={() => void blade()} title="Blade at playhead">
                    <Scissors size={12} /> Blade
                </button>
                <button type="button" className="st-button" data-size="sm" data-variant="ghost" onClick={rippleDelete} title="Ripple delete selected clip">
                    <Trash2 size={12} /> Ripple
                </button>

                <span style={{ width: 1, height: 18, background: 'var(--st-border-strong)', margin: '0 4px' }} />
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
                    {layout.length} clips · {formatDuration(duration * fps, fps)}
                    {preview ? ` · ${preview.trimIn}–${(preview.trimIn || 0) + preview.trimOut} preview` : ''}
                </span>
                <span style={{ flex: 1 }} />
                <span style={{ color: 'var(--st-faint)', fontSize: 9 }}>
                    drag edges to trim · slip/roll move the source window
                </span>
            </div>

            <div className="st-timeline-scroll">
                <div
                    className="st-timeline-inner"
                    style={{ width: width + 110 }}
                    onPointerMove={onGestureMove}
                    onPointerUp={endGesture}
                    onPointerCancel={endGesture}
                >
                    <div className="st-timeline-ruler" style={{ marginLeft: 110, width }} onPointerDown={seek}>
                        {ticks.map((second) => (
                            <div
                                key={second}
                                className="st-timeline-tick"
                                style={{ left: second * timelineZoom }}
                            >
                                {formatDuration(second * fps, fps)}
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
                                {layout.filter((clip) => clip.track === track.id).map((clip) => {
                                    const activePreview = previewFor(clip.id);
                                    const neighborPreview = preview?.neighborId === clip.id ? preview : null;
                                    const trimIn = neighborPreview?.neighborTrimIn ?? activePreview?.trimIn ?? clip.shot.trimIn ?? 0;
                                    const trimOut = neighborPreview?.neighborTrimOut ?? activePreview?.trimOut ?? clip.shot.trimOut ?? 0;
                                    const durationFrames = Math.max(
                                        1,
                                        (clip.shot.numFrames || 0) - trimIn - trimOut,
                                    );
                                    const durationSeconds = durationFrames / fps;
                                    const left = clip.start * timelineZoom;
                                    const clippedWidth = Math.max(8, durationSeconds * timelineZoom);
                                    return (
                                        <div
                                            key={clip.id}
                                            className="st-timeline-clip"
                                            data-track={track.id}
                                            data-selected={clip.id === selectedShotId}
                                            style={{ left, width: clippedWidth }}
                                            onClick={(event) => {
                                                event.stopPropagation();
                                                selectShot(clip.id);
                                                setCurrentTime(clip.start);
                                            }}
                                            onPointerDown={(event) => {
                                                if (track.id !== 0) return;
                                                if (tool === 'slip') beginGesture(event, 'slip', clip.shot);
                                                else if (tool === 'roll') beginGesture(event, 'roll', clip.shot);
                                            }}
                                            title={`${clip.shot.action || clip.shot.prompt || 'Shot'} · ${durationFrames}f · in ${trimIn} / out ${trimOut}`}
                                        >
                                            {tool === 'select' && (
                                                <>
                                                    <span
                                                        className="st-trim-handle st-trim-in"
                                                        role="presentation"
                                                        onPointerDown={(event) => beginGesture(event, 'trim-in', clip.shot)}
                                                    />
                                                    <span
                                                        className="st-trim-handle st-trim-out"
                                                        role="presentation"
                                                        onPointerDown={(event) => beginGesture(event, 'trim-out', clip.shot)}
                                                    />
                                                </>
                                            )}
                                            <span className="st-timeline-clip-title">
                                                {clip.shot.action || clip.shot.prompt || 'Shot'}
                                            </span>
                                            <span className="st-timeline-clip-meta st-mono">{durationFrames}f</span>
                                        </div>
                                    );
                                })}
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
