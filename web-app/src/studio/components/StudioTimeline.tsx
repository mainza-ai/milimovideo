import { useEffect, useMemo, useRef, useState } from 'react';
import {
    Captions, Music, Minus, MousePointer2, Plus, Scissors, Trash2, MoveHorizontal,
    UnfoldHorizontal, Upload,
} from 'lucide-react';
import { useShallow } from 'zustand/react/shallow';
import { useStudioDocumentStore, type CaptionCue, type Shot } from '../state/documentStore';
import { computeTimelineLayout, type TimelineClip } from '../../utils/timelineUtils';
import { useStudioEditor } from '../state/editorStore';
import { formatDuration } from '../data/useStudioData';
import { bladeSourceFrame, rollCut, slipShot, trimShot } from '../timeline/ops';
import { removeCue, sanitizeCues, upsertCue } from '../captions/cues';
import { parseCaptions } from '../captions/parse';
import { cullTimelineClips } from '../timeline/culling';
import { Waveform } from './Waveform';
import { getAssetUrl } from '../../config';
import { useStudioAssets } from '../data/useStudioData';

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
    const project = useStudioDocumentStore((state) => state.project);
    const {
        selectedShotId, selectShot, setCurrentTime, updateShot,
        splitShot, deleteShot, addToast, setProject, addShot, reorderShots,
        trackStates, toggleTrackMute, toggleTrackSolo,
    } = useStudioDocumentStore(useShallow((state) => ({
        selectedShotId: state.selectedShotId,
        selectShot: state.selectShot,
        setCurrentTime: state.setCurrentTime,
        updateShot: state.updateShot,
        splitShot: state.splitShot,
        deleteShot: state.deleteShot,
        addToast: state.addToast,
        setProject: state.setProject,
        addShot: state.addShot,
        reorderShots: state.reorderShots,
        trackStates: state.trackStates,
        toggleTrackMute: state.toggleTrackMute,
        toggleTrackSolo: state.toggleTrackSolo,
    })));
    const { timelineZoom, setTimelineZoom } = useStudioEditor();
    const [tool, setTool] = useState<Tool>('select');
    const [preview, setPreview] = useState<Preview | null>(null);
    const [selectedCueId, setSelectedCueId] = useState<string | null>(null);
    const [audioPickerOpen, setAudioPickerOpen] = useState(false);
    const [tableView, setTableView] = useState(false);
    const [scrollLeft, setScrollLeft] = useState(0);
    const [viewportWidth, setViewportWidth] = useState(900);
    const scrollRef = useRef<HTMLDivElement>(null);
    const captionFileRef = useRef<HTMLInputElement>(null);
    const { assets } = useStudioAssets();
    const audioAssets = useMemo(() => assets.filter((asset) => asset.type === 'audio'), [assets]);
    const gesture = useRef<Gesture | null>(null);
    const fps = project.fps || 25;
    const captions = project.captions || [];
    const selectedCue = captions.find((cue) => cue.id === selectedCueId) || null;

    useEffect(() => {
        const host = scrollRef.current;
        if (!host) return;
        const observer = new ResizeObserver(() => setViewportWidth(host.clientWidth));
        observer.observe(host);
        setViewportWidth(host.clientWidth);
        return () => observer.disconnect();
    }, [tableView]);

    const layout = useMemo(() => computeTimelineLayout(project), [project]);
    const duration = Math.max(10, ...layout.map((clip) => clip.end)) + 2;
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
    const visibleLayout = useMemo(
        () => cullTimelineClips(layout, scrollLeft, viewportWidth, timelineZoom),
        [layout, scrollLeft, timelineZoom, viewportWidth],
    );

    const findClip = (shotId: string): TimelineClip | undefined =>
        layout.find((clip) => clip.id === shotId);

    const blade = async () => {
        const time = useStudioDocumentStore.getState().currentTime;
        const target = selectedShotId ? findClip(selectedShotId) : undefined;
        const clip = target && time >= target.start && time <= target.end
            ? target
            : v1Clips.find((item) => time > item.start && time < item.end);
        if (!clip) {
            addToast('Place the playhead inside a clip to blade it', 'info');
            return;
        }
        const frame = bladeSourceFrame(clip.shot, time - clip.start, fps);
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

    const slideSelected = (direction: -1 | 1) => {
        if (!selectedShotId) return;
        const v1Indices = project.shots
            .map((shot, index) => (shot.trackIndex === 0 ? index : -1))
            .filter((index) => index >= 0);
        const current = project.shots.findIndex((shot) => shot.id === selectedShotId);
        const position = v1Indices.indexOf(current);
        const target = position >= 0 ? v1Indices[position + direction] : undefined;
        if (target == null) {
            addToast('No adjacent V1 clip to slide against', 'info');
            return;
        }
        reorderShots(current, target);
    };

    const changeCaptions = (next: CaptionCue[]) => {
        setProject({ ...project, captions: next });
    };

    const addCue = () => {
        const startFrame = Math.max(0, Math.round(useStudioDocumentStore.getState().currentTime * fps));
        const cue = {
            id: typeof crypto !== 'undefined' && 'randomUUID' in crypto
                ? crypto.randomUUID()
                : `cue_${Date.now()}`,
            startFrame,
            endFrame: startFrame + Math.max(1, Math.round(fps * 2)),
            text: 'New caption',
        };
        changeCaptions(upsertCue(captions, cue));
        setSelectedCueId(cue.id);
    };

    const editCue = (patch: Partial<CaptionCue>) => {
        if (!selectedCue) return;
        changeCaptions(upsertCue(captions, { ...selectedCue, ...patch }));
    };

    const deleteCue = () => {
        if (!selectedCueId) return;
        changeCaptions(removeCue(captions, selectedCueId));
        setSelectedCueId(null);
    };

    const addAudioClip = (asset: { url: string; filename: string }) => {
        const audio = new Audio(getAssetUrl(asset.url));
        audio.addEventListener('loadedmetadata', () => {
            const frames = Math.max(1, Math.round((audio.duration || 1) * fps));
            addShot({
                prompt: asset.filename,
                trackIndex: 2,
                startFrame: Math.max(0, Math.round(useStudioDocumentStore.getState().currentTime * fps)),
                numFrames: frames,
                videoUrl: asset.url,
                status: 'completed',
                isGenerating: false,
            });
            addToast(`Added audio: ${asset.filename}`, 'success');
            setAudioPickerOpen(false);
        });
        audio.addEventListener('error', () => {
            addToast(`Could not read ${asset.filename}`, 'error');
        });
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

    const nudgeTrim = (shot: Shot, edge: 'in' | 'out', delta: number) => {
        const result = trimShot(shot, edge, delta);
        if (result) updateShot(shot.id, result);
    };

    useEffect(() => {
        const onKeyDown = (event: KeyboardEvent) => {
            const target = event.target as HTMLElement | null;
            const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(target?.tagName || '')
                || !!target?.isContentEditable;
            if (typing || event.metaKey || event.ctrlKey || event.altKey) return;
            if (event.key.toLowerCase() === 'b') {
                event.preventDefault();
                void blade();
            } else if (event.key.toLowerCase() === 'r') {
                event.preventDefault();
                rippleDelete();
            } else if (event.key === '[') {
                event.preventDefault();
                slideSelected(-1);
            } else if (event.key === ']') {
                event.preventDefault();
                slideSelected(1);
            }
        };
        window.addEventListener('keydown', onKeyDown);
        return () => window.removeEventListener('keydown', onKeyDown);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [project, selectedShotId]);

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
                <button
                    type="button"
                    className="st-button"
                    data-size="sm"
                    data-variant="ghost"
                    onClick={() => slideSelected(-1)}
                    title="Slide selected V1 clip earlier"
                >
                    ◀ Slide
                </button>
                <button
                    type="button"
                    className="st-button"
                    data-size="sm"
                    data-variant="ghost"
                    onClick={() => slideSelected(1)}
                    title="Slide selected V1 clip later"
                >
                    Slide ▶
                </button>
                <button type="button" className="st-button" data-size="sm" onClick={addCue} title="Add a caption cue at the playhead">
                    <Captions size={12} /> Caption
                </button>
                <button
                    type="button"
                    className="st-button"
                    data-size="sm"
                    data-variant="ghost"
                    onClick={() => captionFileRef.current?.click()}
                    title="Import SRT or VTT captions"
                >
                    <Upload size={12} /> Import
                </button>
                <input
                    ref={captionFileRef}
                    type="file"
                    accept=".srt,.vtt,text/vtt,application/x-subrip"
                    style={{ display: 'none' }}
                    aria-label="Import caption file"
                    onChange={async (event) => {
                        const file = event.target.files?.[0];
                        event.target.value = '';
                        if (!file) return;
                        try {
                            const parsed = parseCaptions(await file.text(), fps);
                            if (parsed.length === 0) {
                                addToast('No cues found in that file', 'info');
                                return;
                            }
                            changeCaptions(sanitizeCues([...captions, ...parsed]));
                            addToast(`Imported ${parsed.length} caption cues`, 'success');
                        } catch {
                            addToast('Could not read that caption file', 'error');
                        }
                    }}
                />
                <button
                    type="button"
                    className="st-button"
                    data-size="sm"
                    data-variant={audioPickerOpen ? 'primary' : undefined}
                    onClick={() => setAudioPickerOpen(!audioPickerOpen)}
                    title="Place an uploaded audio asset on A1 at the playhead"
                >
                    <Music size={12} /> Audio
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
                    B blade · R ripple · [ ] slide
                </span>
                <button
                    type="button"
                    className="st-button"
                    data-size="sm"
                    data-variant={tableView ? 'primary' : 'ghost'}
                    onClick={() => setTableView(!tableView)}
                >
                    {tableView ? 'Timeline' : 'Table'}
                </button>
            </div>

            {audioPickerOpen && (
                <div className="st-audio-picker">
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}>
                        <Music size={12} />
                        <strong style={{ fontSize: 10 }}>Place audio on A1</strong>
                        <button
                            type="button"
                            className="st-icon-button"
                            style={{ marginLeft: 'auto' }}
                            onClick={() => setAudioPickerOpen(false)}
                            aria-label="Close audio picker"
                        >
                            ×
                        </button>
                    </div>
                    {audioAssets.length === 0 ? (
                        <div style={{ color: 'var(--st-muted)', fontSize: 10 }}>
                            No audio uploads yet. Upload music or voice in the legacy Library.
                        </div>
                    ) : (
                        <div className="st-audio-list">
                            {audioAssets.map((asset) => (
                                <button
                                    key={asset.id}
                                    type="button"
                                    className="st-button"
                                    data-size="sm"
                                    onClick={() => addAudioClip(asset)}
                                >
                                    <Music size={11} /> {asset.filename}
                                </button>
                            ))}
                        </div>
                    )}
                </div>
            )}

            {selectedCue && (
                <div className="st-caption-editor">
                    <Captions size={13} />
                    <input
                        className="st-input"
                        style={{ flex: 1, minWidth: 160 }}
                        value={selectedCue.text}
                        aria-label="Caption text"
                        onChange={(event) => editCue({ text: event.target.value })}
                    />
                    <label className="st-caption-time">
                        in
                        <input
                            className="st-input st-mono"
                            type="number"
                            step={0.1}
                            min={0}
                            value={(selectedCue.startFrame / fps).toFixed(2)}
                            onChange={(event) => editCue({
                                startFrame: Math.round(Number(event.target.value) * fps),
                            })}
                        />
                    </label>
                    <label className="st-caption-time">
                        out
                        <input
                            className="st-input st-mono"
                            type="number"
                            step={0.1}
                            min={0}
                            value={(selectedCue.endFrame / fps).toFixed(2)}
                            onChange={(event) => editCue({
                                endFrame: Math.round(Number(event.target.value) * fps),
                            })}
                        />
                    </label>
                    <button type="button" className="st-button" data-size="sm" onClick={deleteCue}>
                        <Trash2 size={12} /> Delete cue
                    </button>
                </div>
            )}

            {tableView ? (
                <div className="st-list-wrap" style={{ flex: 1 }}>
                    <table className="st-table">
                        <caption>Timeline clips — editable equivalent of the tracks</caption>
                        <thead>
                            <tr>
                                <th scope="col">Track</th>
                                <th scope="col">Clip</th>
                                <th scope="col">Start</th>
                                <th scope="col">Duration</th>
                                <th scope="col">Trim in/out</th>
                                <th scope="col">Speed</th>
                                <th scope="col">Audio</th>
                                <th scope="col">Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {[...layout].sort((a, b) => a.track - b.track || a.start - b.start).map((clip) => {
                                const sourceFrames = Math.max(
                                    1,
                                    (clip.shot.numFrames || 0) - (clip.shot.trimIn || 0) - (clip.shot.trimOut || 0),
                                );
                                const speed = clip.shot.speed || 1;
                                return (
                                    <tr
                                        key={clip.id}
                                        data-selected={clip.id === selectedShotId}
                                        onClick={() => selectShot(clip.id)}
                                    >
                                        <td>T{clip.track + 1}</td>
                                        <td>{clip.shot.action || clip.shot.prompt || 'Shot'}</td>
                                        <td className="st-mono">{clip.start.toFixed(2)}s</td>
                                        <td className="st-mono">{(sourceFrames / fps / speed).toFixed(2)}s</td>
                                        <td onClick={(event) => event.stopPropagation()}>
                                            <div style={{ display: 'flex', gap: 4 }}>
                                                <input
                                                    className="st-input"
                                                    style={{ width: 62, height: 26, padding: '2px 5px' }}
                                                    type="number"
                                                    min={0}
                                                    max={clip.shot.numFrames}
                                                    value={clip.shot.trimIn || 0}
                                                    aria-label="Trim in"
                                                    onChange={(event) => updateShot(clip.id, { trimIn: Number(event.target.value) })}
                                                />
                                                <input
                                                    className="st-input"
                                                    style={{ width: 62, height: 26, padding: '2px 5px' }}
                                                    type="number"
                                                    min={0}
                                                    max={clip.shot.numFrames}
                                                    value={clip.shot.trimOut || 0}
                                                    aria-label="Trim out"
                                                    onChange={(event) => updateShot(clip.id, { trimOut: Number(event.target.value) })}
                                                />
                                            </div>
                                        </td>
                                        <td onClick={(event) => event.stopPropagation()}>
                                            <select
                                                className="st-select"
                                                style={{ width: 76, height: 26, padding: '2px 5px' }}
                                                value={speed}
                                                aria-label="Speed"
                                                onChange={(event) => updateShot(clip.id, { speed: Number(event.target.value) })}
                                            >
                                                {[0.5, 0.75, 1, 1.25, 1.5, 2, 3, 4].map((rate) => (
                                                    <option key={rate} value={rate}>{rate}×</option>
                                                ))}
                                            </select>
                                        </td>
                                        <td onClick={(event) => event.stopPropagation()}>
                                            <button
                                                type="button"
                                                className="st-button"
                                                data-size="sm"
                                                data-variant={clip.shot.audioMuted ? 'danger' : 'ghost'}
                                                onClick={() => updateShot(clip.id, { audioMuted: !clip.shot.audioMuted })}
                                            >
                                                {clip.shot.audioMuted ? 'Muted' : 'Audible'}
                                            </button>
                                        </td>
                                        <td onClick={(event) => event.stopPropagation()}>
                                            <button
                                                type="button"
                                                className="st-button"
                                                data-size="sm"
                                                data-variant="danger"
                                                onClick={() => deleteShot(clip.id)}
                                            >
                                                <Trash2 size={11} /> Delete
                                            </button>
                                        </td>
                                    </tr>
                                );
                            })}
                        </tbody>
                    </table>
                </div>
            ) : (
            <div
                className="st-timeline-scroll"
                ref={scrollRef}
                onScroll={(event) => setScrollLeft(event.currentTarget.scrollLeft)}
            >
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
                                <div style={{ display: 'flex', gap: 3, marginTop: 2 }}>
                                    <button
                                        type="button"
                                        className="st-button"
                                        data-size="sm"
                                        data-variant={trackStates[track.id]?.muted ? 'danger' : 'ghost'}
                                        style={{ padding: '1px 5px', fontSize: 8 }}
                                        onClick={() => toggleTrackMute(track.id)}
                                        aria-pressed={!!trackStates[track.id]?.muted}
                                    >
                                        M
                                    </button>
                                    <button
                                        type="button"
                                        className="st-button"
                                        data-size="sm"
                                        data-variant={trackStates[track.id]?.solo ? 'primary' : 'ghost'}
                                        style={{ padding: '1px 5px', fontSize: 8 }}
                                        onClick={() => toggleTrackSolo(track.id)}
                                        aria-pressed={!!trackStates[track.id]?.solo}
                                    >
                                        S
                                    </button>
                                </div>
                            </div>
                            <div
                                className="st-timeline-lane"
                                style={{ width }}
                                onClick={seek}
                                role="presentation"
                            >
                                {visibleLayout.filter((clip) => clip.track === track.id).map((clip) => {
                                    const activePreview = previewFor(clip.id);
                                    const neighborPreview = preview?.neighborId === clip.id ? preview : null;
                                    const trimIn = neighborPreview?.neighborTrimIn ?? activePreview?.trimIn ?? clip.shot.trimIn ?? 0;
                                    const trimOut = neighborPreview?.neighborTrimOut ?? activePreview?.trimOut ?? clip.shot.trimOut ?? 0;
                                    const sourceFrames = Math.max(
                                        1,
                                        (clip.shot.numFrames || 0) - trimIn - trimOut,
                                    );
                                    const speed = Math.min(4, Math.max(0.25, clip.shot.speed || 1));
                                    const durationFrames = Math.max(1, Math.round(sourceFrames / speed));
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
                                            role="button"
                                            tabIndex={0}
                                            aria-label={`${clip.shot.action || clip.shot.prompt || 'Shot'} clip, ${durationFrames} frames`}
                                            onKeyDown={(event) => {
                                                if (event.key === 'Enter' || event.key === ' ') {
                                                    event.preventDefault();
                                                    selectShot(clip.id);
                                                    setCurrentTime(clip.start);
                                                } else if (event.key === 'Delete' || event.key === 'Backspace') {
                                                    event.preventDefault();
                                                    deleteShot(clip.id);
                                                    addToast('Clip removed — V1 rippled', 'info');
                                                }
                                            }}
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
                                            {track.id === 2 && clip.shot.videoUrl && (
                                                <span className="st-waveform-wrap">
                                                    <Waveform url={clip.shot.videoUrl} height={22} />
                                                </span>
                                            )}
                                            {tool === 'select' && (
                                                <>
                                                    <span
                                                        className="st-trim-handle st-trim-in"
                                                        role="slider"
                                                        tabIndex={0}
                                                        aria-label="Trim in"
                                                        aria-valuenow={trimIn}
                                                        aria-valuemin={0}
                                                        aria-valuemax={Math.max(0, (clip.shot.numFrames || 0) - trimOut - 1)}
                                                        onPointerDown={(event) => beginGesture(event, 'trim-in', clip.shot)}
                                                        onKeyDown={(event) => {
                                                            if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
                                                                event.preventDefault();
                                                                event.stopPropagation();
                                                                nudgeTrim(clip.shot, 'in', event.key === 'ArrowRight' ? 1 : -1);
                                                            }
                                                        }}
                                                    />
                                                    <span
                                                        className="st-trim-handle st-trim-out"
                                                        role="slider"
                                                        tabIndex={0}
                                                        aria-label="Trim out"
                                                        aria-valuenow={trimOut}
                                                        aria-valuemin={0}
                                                        aria-valuemax={Math.max(0, (clip.shot.numFrames || 0) - trimIn - 1)}
                                                        onPointerDown={(event) => beginGesture(event, 'trim-out', clip.shot)}
                                                        onKeyDown={(event) => {
                                                            if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
                                                                event.preventDefault();
                                                                event.stopPropagation();
                                                                nudgeTrim(clip.shot, 'out', event.key === 'ArrowRight' ? 1 : -1);
                                                            }
                                                        }}
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

                    <div className="st-timeline-track">
                        <div className="st-timeline-label">
                            <strong style={{ color: 'var(--st-captions)' }}>C1</strong>
                            <span>captions · {captions.length}</span>
                        </div>
                        <div className="st-timeline-lane" style={{ width }} onClick={seek} role="presentation">
                            {captions.map((cue) => (
                                <button
                                    key={cue.id}
                                    type="button"
                                    className="st-timeline-clip st-caption-clip"
                                    data-selected={cue.id === selectedCueId}
                                    style={{
                                        left: (cue.startFrame / fps) * timelineZoom,
                                        width: Math.max(10, ((cue.endFrame - cue.startFrame) / fps) * timelineZoom),
                                        top: 14,
                                        bottom: 14,
                                    }}
                                    onClick={(event) => {
                                        event.stopPropagation();
                                        setSelectedCueId(cue.id);
                                        setCurrentTime(cue.startFrame / fps);
                                    }}
                                    title={`${cue.text} · ${(cue.startFrame / fps).toFixed(2)}s–${(cue.endFrame / fps).toFixed(2)}s`}
                                >
                                    <span className="st-timeline-clip-title">{cue.text}</span>
                                </button>
                            ))}
                            {captions.length === 0 && (
                                <span className="st-caption-empty">
                                    No captions — add a cue at the playhead
                                </span>
                            )}
                        </div>
                    </div>

                    <Playhead timelineZoom={timelineZoom} />
                </div>
            </div>
            )}
        </section>
    );
}

function Playhead({ timelineZoom }: { timelineZoom: number }) {
    const currentTime = useStudioDocumentStore((state) => state.currentTime);
    return (
        <div
            className="st-playhead"
            style={{ left: 110 + currentTime * timelineZoom }}
            aria-hidden="true"
        />
    );
}
