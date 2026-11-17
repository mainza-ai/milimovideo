import { useMemo, useRef, useState } from 'react';
import { CheckCircle2, Download, Loader2, Scissors, Upload, Zap } from 'lucide-react';
import { useTimelineStore } from '../../stores/timelineStore';
import { getAssetUrl } from '../../config';
import { computeTimelineLayout } from '../../utils/timelineUtils';
import { useStudioCreateRender, useStudioProject, useStudioRenders } from '../data/useStudioData';
import { useStudioEditor } from '../state/editorStore';
import { parseCaptions } from '../captions/parse';
import { buildCandidates, type ShortCue } from '../shorts/candidates';

const PRESETS = [
    { id: 'vertical', label: '9:16', width: 1080, height: 1920 },
    { id: 'square', label: '1:1', width: 1080, height: 1080 },
    { id: 'portrait', label: '4:5', width: 1080, height: 1350 },
] as const;

export function ShortsSurface() {
    const project = useStudioProject();
    const { setSurface } = useStudioEditor();
    const setProject = useTimelineStore((state) => state.setProject);
    const saveProject = useTimelineStore((state) => state.saveProject);
    const setCurrentTime = useTimelineStore((state) => state.setCurrentTime);
    const createRender = useStudioCreateRender(project.id);
    const { renders, refresh } = useStudioRenders(project.id);
    const fileRef = useRef<HTMLInputElement>(null);

    const [cues, setCues] = useState<ShortCue[]>([]);
    const [fromTranscript, setFromTranscript] = useState(false);
    const [transcriptName, setTranscriptName] = useState<string | null>(null);
    const [preset, setPreset] = useState<typeof PRESETS[number]['id']>('vertical');
    const [selectedId, setSelectedId] = useState<string | null>(null);
    const [busy, setBusy] = useState(false);

    const fps = project.fps || 25;
    const layout = useMemo(() => computeTimelineLayout(project), [project]);
    const totalDuration = Math.max(0.1, ...layout.map((clip) => clip.end));

    const sourceCues = useMemo<ShortCue[]>(() => {
        if (cues.length > 0) return cues;
        return layout.map((clip) => ({
            start: clip.start,
            end: clip.end,
            text: clip.shot.action || clip.shot.prompt || `Shot ${clip.id.slice(0, 6)}`,
        }));
    }, [cues, layout]);

    const candidates = useMemo(
        () => buildCandidates(sourceCues, 10, 45),
        [sourceCues],
    );
    const selected = candidates.find((candidate) => candidate.id === selectedId) || candidates[0] || null;

    const importTranscript = async (file: File) => {
        const parsed = parseCaptions(await file.text(), fps);
        const next = parsed.map((cue) => ({
            start: cue.startFrame / fps,
            end: cue.endFrame / fps,
            text: cue.text,
        }));
        setCues(next);
        setFromTranscript(next.length > 0);
        setTranscriptName(file.name);
        setSelectedId(null);
    };

    const exportShort = async () => {
        if (!selected) return;
        const target = PRESETS.find((item) => item.id === preset) || PRESETS[0];
        setBusy(true);
        try {
            setProject({
                ...project,
                captionsBurn: true,
                captions: fromTranscript
                    ? cues.map((cue, index) => ({
                        id: `short_cue_${index}`,
                        startFrame: Math.round(cue.start * fps),
                        endFrame: Math.round(cue.end * fps),
                        text: cue.text,
                    }))
                    : project.captions,
            });
            await saveProject();
            await createRender({
                width: target.width,
                height: target.height,
                fps,
                start_frame: Math.round(selected.start * fps),
                end_frame: Math.round(selected.end * fps),
                fit: 'crop',
                crf: 18,
                preset: 'fast',
            });
            await refresh();
        } finally {
            setBusy(false);
        }
    };

    return (
        <div className="st-page">
            <div className="st-page-head">
                <div>
                    <h2 className="st-page-title">Shorts</h2>
                    <p className="st-page-subtitle">
                        Source cut {totalDuration.toFixed(1)}s · {candidates.length} candidate windows.
                        Transcript cues stay local; the render burns only the selected range.
                    </p>
                </div>
                <div className="st-button-row">
                    <button
                        type="button"
                        className="st-button"
                        data-size="sm"
                        onClick={() => fileRef.current?.click()}
                    >
                        <Upload size={12} /> {transcriptName || 'Import SRT/VTT'}
                    </button>
                    <input
                        ref={fileRef}
                        type="file"
                        accept=".srt,.vtt,text/vtt,application/x-subrip"
                        style={{ display: 'none' }}
                        aria-label="Import transcript"
                        onChange={(event) => {
                            const file = event.target.files?.[0];
                            event.target.value = '';
                            if (file) void importTranscript(file);
                        }}
                    />
                    {cues.length > 0 && (
                        <button
                            type="button"
                            className="st-button"
                            data-size="sm"
                            data-variant="ghost"
                            onClick={() => { setCues([]); setFromTranscript(false); setTranscriptName(null); }}
                        >
                            Clear transcript
                        </button>
                    )}
                </div>
            </div>

            <div className="st-panel" style={{ marginBottom: 14 }}>
                <h3 className="st-panel-title">Reframe preset</h3>
                <div className="st-button-row">
                    {PRESETS.map((item) => (
                        <button
                            key={item.id}
                            type="button"
                            className="st-button"
                            data-size="sm"
                            data-variant={preset === item.id ? 'primary' : 'ghost'}
                            onClick={() => setPreset(item.id)}
                        >
                            {item.label} · {item.width}×{item.height}
                        </button>
                    ))}
                    <button
                        type="button"
                        className="st-button"
                        data-size="sm"
                        data-variant="primary"
                        disabled={!selected || busy}
                        onClick={() => void exportShort()}
                    >
                        {busy ? <Loader2 size={12} className="animate-spin" /> : <Scissors size={12} />}
                        Export selected short
                    </button>
                    {selected && (
                        <button
                            type="button"
                            className="st-button"
                            data-size="sm"
                            data-variant="ghost"
                            onClick={() => {
                                setCurrentTime(selected.start);
                                setSurface('edit');
                            }}
                        >
                            <Zap size={12} /> Open at {selected.start.toFixed(1)}s
                        </button>
                    )}
                </div>
                {!fromTranscript && (
                    <p className="st-value" style={{ color: 'var(--st-faint)', fontSize: 9, marginTop: 6 }}>
                        No transcript: candidates are built from shot text and timing. Import an SRT/VTT for real caption-aware scoring.
                    </p>
                )}
            </div>

            <div className="st-grid">
                {candidates.map((candidate) => (
                    <button
                        key={candidate.id}
                        type="button"
                        className="st-panel"
                        style={{
                            textAlign: 'left',
                            cursor: 'pointer',
                            borderColor: selected?.id === candidate.id ? 'var(--st-accent)' : undefined,
                        }}
                        onClick={() => setSelectedId(candidate.id)}
                    >
                        <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8 }}>
                            <strong style={{ fontSize: 12 }}>
                                {candidate.start.toFixed(1)}–{candidate.end.toFixed(1)}s
                            </strong>
                            <span className="st-status-pill" data-tone={candidate.score > 40 ? 'good' : 'warn'}>
                                {candidate.score}
                            </span>
                        </div>
                        <div className="st-value" style={{ color: 'var(--st-muted)', fontSize: 10, marginTop: 6 }}>
                            {candidate.duration.toFixed(1)}s · {candidate.text.split(/\s+/).length} words
                        </div>
                        <p className="st-value" style={{ fontSize: 10, marginTop: 6, color: 'var(--st-secondary)' }}>
                            {candidate.text.slice(0, 220)}{candidate.text.length > 220 ? '…' : ''}
                        </p>
                        {candidate.id === selected?.id && (
                            <span className="st-chip"><CheckCircle2 size={10} /> selected</span>
                        )}
                    </button>
                ))}
                {candidates.length === 0 && (
                    <div className="st-empty">
                        <strong>No candidates yet</strong>
                        <span>Import a transcript or add shots with at least 10 seconds of text/timing.</span>
                    </div>
                )}
            </div>

            <div className="st-panel" style={{ marginTop: 14 }}>
                <h3 className="st-panel-title">Recent exports</h3>
                {renders.length === 0 ? (
                    <span className="st-value" style={{ color: 'var(--st-muted)' }}>No shorts exported yet.</span>
                ) : (
                    <table className="st-table">
                        <thead><tr><th>Render</th><th>Status</th><th>Progress</th><th>Output</th></tr></thead>
                        <tbody>
                            {renders.slice(0, 8).map((render) => (
                                <tr key={render.id}>
                                    <td className="st-mono">{render.id}</td>
                                    <td>{render.status}</td>
                                    <td>{render.progress}%</td>
                                    <td>
                                        {render.output_path ? (
                                            <a href={getAssetUrl(render.output_path)} target="_blank" rel="noreferrer" style={{ color: 'var(--st-accent)' }}>
                                                <Download size={12} /> open
                                            </a>
                                        ) : render.error_message || '—'}
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                )}
            </div>
        </div>
    );
}
