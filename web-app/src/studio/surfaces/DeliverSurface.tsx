import { useEffect, useMemo, useState } from 'react';
import { Download, Film, Loader2, Send } from 'lucide-react';
import { useStudioDocumentStore } from '../state/documentStore';
import { useStudioCreateRender, useStudioProject, useStudioRenders } from '../data/useStudioData';
import { apiUrl, getAssetUrl } from '../../config';
import { computeTimelineLayout } from '../../utils/timelineUtils';

const PRESETS = [
    { id: 'project', label: 'Project', width: null, height: null },
    { id: '1080p', label: '1080p', width: 1920, height: 1080 },
    { id: '720p', label: '720p', width: 1280, height: 720 },
    { id: 'vertical', label: 'Vertical 9:16', width: 1080, height: 1920 },
    { id: 'square', label: 'Square 1:1', width: 1080, height: 1080 },
    { id: 'portrait', label: 'Portrait 4:5', width: 1080, height: 1350 },
] as const;

export function DeliverSurface() {
    const project = useStudioProject();
    const { renders, refresh } = useStudioRenders(project.id);
    const createRender = useStudioCreateRender(project.id);
    const setProject = useStudioDocumentStore((state) => state.setProject);
    const addToast = useStudioDocumentStore((state) => state.addToast);
    const saveProject = useStudioDocumentStore((state) => state.saveProject);
    const [preset, setPreset] = useState<typeof PRESETS[number]['id']>(
        (PRESETS.find((item) => item.id === project.renderPreset)?.id || 'project'),
    );
    const [busy, setBusy] = useState(false);
    const layout = useMemo(() => computeTimelineLayout(project), [project]);
    const totalDuration = Math.max(0.1, ...layout.map((clip) => clip.end));
    const [rangeStart, setRangeStart] = useState(0);
    const [rangeEnd, setRangeEnd] = useState(totalDuration);
    const [fit, setFit] = useState<'pad' | 'crop'>('pad');
    const [crf, setCrf] = useState(project.renderCrf ?? 18);
    const [speed, setSpeed] = useState(project.renderSpeed ?? 'fast');
    const [audioBitrate, setAudioBitrate] = useState(project.renderAudioBitrate ?? '192k');
    const [audioNormalize, setAudioNormalize] = useState(project.renderAudioNormalize ?? false);

    useEffect(() => {
        setRangeEnd((current) => (current > totalDuration ? totalDuration : current || totalDuration));
    }, [totalDuration]);

    useEffect(() => {
        setCrf(project.renderCrf ?? 18);
        setSpeed(project.renderSpeed ?? 'fast');
        setAudioBitrate(project.renderAudioBitrate ?? '192k');
        setAudioNormalize(project.renderAudioNormalize ?? false);
    }, [project.id, project.renderCrf, project.renderSpeed, project.renderAudioBitrate, project.renderAudioNormalize]);

    const persistRenderSettings = (patch: Partial<typeof project>) => {
        setProject({ ...project, ...patch });
        void saveProject();
    };

    const cancelRender = async (renderId: string) => {
        try {
            const response = await fetch(apiUrl(`/jobs/${renderId}/cancel`), { method: 'POST' });
            const data = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(data.detail || 'Cancel failed');
            addToast(`Render ${renderId} ${data.status || 'cancelled'}`, 'info');
            await refresh();
        } catch (error) {
            addToast(error instanceof Error ? error.message : 'Cancel failed', 'error');
        }
    };

    const start = async () => {
        const selected = PRESETS.find((item) => item.id === preset) || PRESETS[0];
        setBusy(true);
        try {
            await createRender({
                width: selected.width || project.resolutionW,
                height: selected.height || project.resolutionH,
                fps: project.fps,
                start_frame: Math.round(Math.min(rangeStart, rangeEnd) * project.fps),
                end_frame: Math.round(Math.max(rangeStart, rangeEnd) * project.fps),
                fit,
                crf,
                preset: speed,
                audio_bitrate: audioBitrate,
                audio_normalize: audioNormalize,
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
                    <h2 className="st-page-title">Deliver</h2>
                    <p className="st-page-subtitle">
                        Export the computed cut: V1/V2 timing, alpha overlays, embedded audio, music, and captions.
                        The render reads the same sequence order as the board and timeline.
                    </p>
                </div>
                <div className="st-button-row">
                    <span className="st-status-pill" title="Caption cues are burned in by the render">
                        {project.captions?.length || 0} caption cues
                    </span>
                    <label className="st-status-pill" style={{ cursor: 'pointer' }}>
                        <input
                            type="checkbox"
                            checked={project.captionsBurn ?? true}
                            onChange={(event) => setProject({ ...project, captionsBurn: event.target.checked })}
                        />
                        Burn captions
                    </label>
                    <button type="button" className="st-button" data-variant="primary" disabled={busy} onClick={() => void start()}>
                        {busy ? <Loader2 size={13} className="animate-spin" /> : <Send size={13} />}
                        {busy ? 'Queueing…' : 'Create export'}
                    </button>
                </div>
            </div>

            <div className="st-panel" style={{ marginBottom: 14 }}>
                <h3 className="st-panel-title">Preset</h3>
                <div className="st-button-row">
                    {PRESETS.map((item) => (
                        <button
                            key={item.id}
                            type="button"
                            className="st-button"
                            data-size="sm"
                            data-variant={preset === item.id ? 'primary' : 'ghost'}
                            onClick={() => {
                                setPreset(item.id);
                                setProject({ ...project, renderPreset: item.id });
                                void saveProject();
                            }}
                        >
                            {item.label}
                            {item.width ? ` · ${item.width}×${item.height}` : ` · ${project.resolutionW}×${project.resolutionH}`}
                        </button>
                    ))}
                </div>
            </div>

            <div className="st-panel" style={{ marginBottom: 14 }}>
                <h3 className="st-panel-title">Encoding</h3>
                <div className="st-form-grid">
                    <label className="st-field">
                        <span>CRF · {crf} {crf <= 18 ? '(quality)' : crf >= 26 ? '(small file)' : ''}</span>
                        <input
                            type="range"
                            min={12}
                            max={32}
                            step={1}
                            value={crf}
                            aria-label="Render CRF"
                            onChange={(event) => setCrf(Number(event.target.value))}
                            onPointerUp={() => persistRenderSettings({ renderCrf: crf })}
                            onKeyUp={() => persistRenderSettings({ renderCrf: crf })}
                            style={{ accentColor: 'var(--st-accent)' }}
                        />
                    </label>
                    <label className="st-field">
                        <span>Encoder speed</span>
                        <select
                            className="st-select"
                            value={speed}
                            onChange={(event) => {
                                const next = event.target.value;
                                setSpeed(next);
                                persistRenderSettings({ renderSpeed: next });
                            }}
                        >
                            {['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow'].map((item) => (
                                <option key={item} value={item}>{item}</option>
                            ))}
                        </select>
                    </label>
                    <label className="st-field">
                        <span>Audio bitrate</span>
                        <select
                            className="st-select"
                            value={audioBitrate}
                            onChange={(event) => {
                                const next = event.target.value;
                                setAudioBitrate(next);
                                persistRenderSettings({ renderAudioBitrate: next });
                            }}
                        >
                            {['96k', '128k', '192k', '256k', '320k'].map((item) => (
                                <option key={item} value={item}>{item}</option>
                            ))}
                        </select>
                    </label>
                    <div className="st-field" style={{ justifyContent: 'end' }}>
                        <span>Audio</span>
                        <label className="st-chip" style={{ cursor: 'pointer', textTransform: 'none', letterSpacing: 0 }}>
                            <input
                                type="checkbox"
                                checked={audioNormalize}
                                onChange={(event) => {
                                    const next = event.target.checked;
                                    setAudioNormalize(next);
                                    persistRenderSettings({ renderAudioNormalize: next });
                                }}
                            />
                            Normalize loudness
                        </label>
                    </div>
                </div>
                <p className="st-value" style={{ color: 'var(--st-faint)', fontSize: 9, marginTop: 8 }}>
                    Values persist with the project. Normalization applies a single-pass loudnorm at -16 LUFS.
                </p>
            </div>

            <div className="st-panel" style={{ marginBottom: 14 }}>
                <h3 className="st-panel-title">Render range · cut is {totalDuration.toFixed(2)}s</h3>
                <div className="st-row">
                    <label>
                        <span className="st-label">In (seconds)</span>
                        <input
                            className="st-input"
                            type="number"
                            min={0}
                            max={totalDuration}
                            step={0.1}
                            value={rangeStart}
                            onChange={(event) => setRangeStart(Math.max(0, Math.min(totalDuration, Number(event.target.value))))}
                        />
                    </label>
                    <label>
                        <span className="st-label">Out (seconds)</span>
                        <input
                            className="st-input"
                            type="number"
                            min={0}
                            max={totalDuration}
                            step={0.1}
                            value={rangeEnd}
                            onChange={(event) => setRangeEnd(Math.max(0, Math.min(totalDuration, Number(event.target.value))))}
                        />
                    </label>
                </div>
                <div className="st-button-row" style={{ marginTop: 8 }}>
                    <span className="st-chip">duration {Math.max(0, rangeEnd - rangeStart).toFixed(2)}s</span>
                    <button
                        type="button"
                        className="st-button"
                        data-size="sm"
                        data-variant="ghost"
                        onClick={() => { setRangeStart(0); setRangeEnd(totalDuration); }}
                    >
                        Full cut
                    </button>
                    <span className="st-chip">Fit</span>
                    <button
                        type="button"
                        className="st-button"
                        data-size="sm"
                        data-variant={fit === 'pad' ? 'primary' : 'ghost'}
                        onClick={() => setFit('pad')}
                        title="Letterbox to preserve the full frame"
                    >
                        Pad
                    </button>
                    <button
                        type="button"
                        className="st-button"
                        data-size="sm"
                        data-variant={fit === 'crop' ? 'primary' : 'ghost'}
                        onClick={() => setFit('crop')}
                        title="Center-crop to fill the output"
                    >
                        Crop
                    </button>
                </div>
            </div>

            <div className="st-panel">
                <h3 className="st-panel-title">Render history</h3>
                {renders.length === 0 ? (
                    <div className="st-empty" style={{ minHeight: 120 }}>
                        <Film size={18} />
                        <strong>No exports yet</strong>
                        <span>Create one from the computed cut with the controls above.</span>
                    </div>
                ) : (
                    <table className="st-table">
                        <thead>
                            <tr><th>Render</th><th>Status</th><th>Progress</th><th>Created</th><th>Output</th><th>Actions</th></tr>
                        </thead>
                        <tbody>
                            {renders.map((render) => (
                                <tr key={render.id}>
                                    <td className="st-mono">{render.id}</td>
                                    <td>{render.status}</td>
                                    <td>{render.progress}%</td>
                                    <td>{new Date(render.created_at).toLocaleString()}</td>
                                    <td>
                                        {render.output_path ? (
                                            <a href={getAssetUrl(render.output_path)} target="_blank" rel="noreferrer" style={{ color: 'var(--st-accent)' }}>
                                                <Download size={12} /> open
                                            </a>
                                        ) : render.error_message ? (
                                            <span style={{ color: 'var(--st-danger)' }}>{render.error_message}</span>
                                        ) : '—'}
                                    </td>
                                    <td>
                                        {['queued', 'rendering', 'processing'].includes(render.status) && (
                                            <button
                                                type="button"
                                                className="st-button"
                                                data-size="sm"
                                                data-variant="ghost"
                                                aria-label={`Cancel render ${render.id}`}
                                                onClick={() => void cancelRender(render.id)}
                                            >
                                                Cancel
                                            </button>
                                        )}
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
