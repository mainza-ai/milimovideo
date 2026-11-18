import { useEffect, useMemo, useRef, useState } from 'react';
import { Download, Film, Loader2, Plus, Send, Trash2, Upload } from 'lucide-react';
import { useStudioDocumentStore } from '../state/documentStore';
import {
    useStudioCancelJob, useStudioCreateRender, useStudioProject, useStudioRenders,
} from '../data/useStudioData';
import { apiFetch, getAssetUrl } from '../../config';
import { useMediaPreview } from '../state/mediaPreviewStore';
import { MediaThumb } from '../components/MediaThumb';
import { computeTimelineLayout } from '../../utils/timelineUtils';
import { uploadProjectLut } from '../data/lut';

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
    const cancelJob = useStudioCancelJob();
    const openMediaPreview = useMediaPreview((state) => state.open);
    const setProject = useStudioDocumentStore((state) => state.setProject);
    const saveProject = useStudioDocumentStore((state) => state.saveProject);
    const addToast = useStudioDocumentStore((state) => state.addToast);
    const [preset, setPreset] = useState<typeof PRESETS[number]['id']>(
        (PRESETS.find((item) => item.id === project.renderPreset)?.id || 'project'),
    );
    const [busy, setBusy] = useState(false);
    const [otioBusy, setOtioBusy] = useState(false);
    const [otioImportBusy, setOtioImportBusy] = useState(false);
    const otioInputRef = useRef<HTMLInputElement>(null);
    const lutInputRef = useRef<HTMLInputElement>(null);
    const [lutBusy, setLutBusy] = useState(false);
    const [titleText, setTitleText] = useState('');
    const [titleStart, setTitleStart] = useState(0);
    const [titleEnd, setTitleEnd] = useState(3);
    const [titleSize, setTitleSize] = useState(48);
    const [titleColor, setTitleColor] = useState('#ffffff');
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

    const uploadLut = async (file: File) => {
        setLutBusy(true);
        try {
            const data = await uploadProjectLut(project.id, file);
            setProject({ ...project, renderLut: data.url });
            void saveProject();
            addToast(`LUT loaded · ${file.name}`, 'success');
        } catch (error) {
            addToast(error instanceof Error ? error.message : 'LUT upload failed', 'error');
        } finally {
            setLutBusy(false);
        }
    };

    const cancelRender = async (renderId: string) => {
        if (await cancelJob(renderId, `Render ${renderId}`)) await refresh();
    };

    const start = async () => {
        const selected = PRESETS.find((item) => item.id === preset) || PRESETS[0];
        setBusy(true);
        try {
            await saveProject();
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
                lut: project.renderLut,
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
                    <input
                        ref={otioInputRef}
                        type="file"
                        accept=".otio,.fcpxml,.xml,.json,application/json"
                        style={{ display: 'none' }}
                        aria-label="Import OTIO or FCPXML file"
                        onChange={(event) => {
                            const file = event.target.files?.[0];
                            event.target.value = '';
                            if (!file) return;
                            setOtioImportBusy(true);
                            const form = new FormData();
                            form.append('file', file);
                            void apiFetch(`/projects/${project.id}/otio/import`, {
                                method: 'POST',
                                body: form,
                            })
                                .then(async (response) => {
                                    const data = await response.json().catch(() => ({}));
                                    if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
                                    const shots = Array.isArray(data.shots) ? data.shots : [];
                                    setProject({ ...project, fps: data.fps || project.fps, shots });
                                    addToast(`Imported ${shots.length} shot(s) from OTIO`, 'success');
                                })
                                .catch((error) => addToast(
                                    error instanceof Error ? error.message : 'Timeline import failed',
                                    'error',
                                ))
                                .finally(() => setOtioImportBusy(false));
                        }}
                    />
                    <button
                        type="button"
                        className="st-button"
                        disabled={otioImportBusy}
                        title="Import an OTIO or FCPXML timeline as shots"
                        onClick={() => otioInputRef.current?.click()}
                    >
                        {otioImportBusy ? <Loader2 size={13} className="animate-spin" /> : <Download size={13} />}
                        Import OTIO
                    </button>
                    <button
                        type="button"
                        className="st-button"
                        disabled={otioBusy}
                        title="Export the computed cut for Resolve, Premiere, or Final Cut"
                        onClick={() => {
                            setOtioBusy(true);
                            void apiFetch(`/projects/${project.id}/otio`)
                                .then(async (response) => {
                                    if (!response.ok) throw new Error(`HTTP ${response.status}`);
                                    const blob = await response.blob();
                                    const url = URL.createObjectURL(blob);
                                    const anchor = document.createElement('a');
                                    anchor.href = url;
                                    anchor.download = `${project.name || 'milimo'}.otio`;
                                    anchor.click();
                                    URL.revokeObjectURL(url);
                                    addToast('OTIO timeline exported', 'success');
                                })
                                .catch((error) => addToast(
                                    error instanceof Error ? error.message : 'OTIO export failed',
                                    'error',
                                ))
                                .finally(() => setOtioBusy(false));
                        }}
                    >
                        {otioBusy ? <Loader2 size={13} className="animate-spin" /> : <Download size={13} />}
                        Export OTIO
                    </button>
                    <button
                        type="button"
                        className="st-button"
                        title="Export the cut as FCPXML"
                        onClick={() => {
                            void apiFetch(`/projects/${project.id}/otio?format=fcpxml`)
                                .then(async (response) => {
                                    if (!response.ok) throw new Error(`HTTP ${response.status}`);
                                    const blob = await response.blob();
                                    const url = URL.createObjectURL(blob);
                                    const anchor = document.createElement('a');
                                    anchor.href = url;
                                    anchor.download = `${project.name || 'milimo'}.fcpxml`;
                                    anchor.click();
                                    URL.revokeObjectURL(url);
                                    addToast('FCPXML timeline exported', 'success');
                                })
                                .catch((error) => addToast(
                                    error instanceof Error ? error.message : 'FCPXML export failed',
                                    'error',
                                ));
                        }}
                    >
                        <Download size={13} /> Export FCPXML
                    </button>
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
                <div className="st-button-row" style={{ marginTop: 10 }}>
                    <input
                        ref={lutInputRef}
                        type="file"
                        accept=".cube,.3dl,.dat,.m3d"
                        style={{ display: 'none' }}
                        aria-label="Upload 3D LUT"
                        onChange={(event) => {
                            const file = event.target.files?.[0];
                            event.target.value = '';
                            if (file) void uploadLut(file);
                        }}
                    />
                    <button
                        type="button"
                        className="st-button"
                        data-size="sm"
                        disabled={lutBusy}
                        onClick={() => lutInputRef.current?.click()}
                    >
                        {lutBusy ? <Loader2 size={12} className="animate-spin" /> : <Upload size={12} />}
                        {project.renderLut ? 'Replace LUT' : 'Add 3D LUT'}
                    </button>
                    {project.renderLut && (
                        <>
                            <span className="st-chip" title={project.renderLut}>
                                {(project.renderLut.split('/').pop() || 'lut').slice(0, 40)}
                            </span>
                            <button
                                type="button"
                                className="st-button"
                                data-size="sm"
                                data-variant="ghost"
                                onClick={() => {
                                    setProject({ ...project, renderLut: undefined });
                                    void saveProject();
                                    addToast('LUT removed', 'info');
                                }}
                            >
                                <Trash2 size={12} /> Remove
                            </button>
                        </>
                    )}
                </div>
                <p className="st-value" style={{ color: 'var(--st-faint)', fontSize: 11, marginTop: 8 }}>
                    Values persist with the project. Normalization applies a single-pass loudnorm at -16 LUFS.
                    A 3D LUT (`.cube`, `.3dl`, `.dat`, `.m3d`) is applied to the V1/V2 composite before captions.
                </p>
            </div>

            <div className="st-panel" style={{ marginBottom: 14 }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8 }}>
                    <h3 className="st-panel-title" style={{ margin: 0 }}>Titles · {project.titles?.length || 0}</h3>
                    <span className="st-value" style={{ color: 'var(--st-faint)', fontSize: 11 }}>
                        Burned in after LUT, before captions
                    </span>
                </div>
                {(project.titles?.length || 0) > 0 && (
                    <div className="st-link-list" style={{ marginTop: 8 }}>
                        {(project.titles || []).map((title) => (
                            <div key={title.id} className="st-link-row">
                                <span className="st-mono">
                                    {(title.startFrame / project.fps).toFixed(1)}–{(title.endFrame / project.fps).toFixed(1)}s
                                </span>
                                <span style={{ flex: 1 }}>{title.text}</span>
                                <button
                                    type="button"
                                    className="st-button"
                                    data-size="sm"
                                    data-variant="ghost"
                                    onClick={() => {
                                        setProject({
                                            ...project,
                                            titles: (project.titles || []).filter((item) => item.id !== title.id),
                                        });
                                        void saveProject();
                                    }}
                                >
                                    <Trash2 size={12} /> Remove
                                </button>
                            </div>
                        ))}
                    </div>
                )}
                <div className="st-row" style={{ marginTop: 8, alignItems: 'end' }}>
                    <label style={{ flex: 2 }}>
                        <span className="st-label">Text</span>
                        <input
                            className="st-input"
                            value={titleText}
                            placeholder="Chapter One"
                            onChange={(event) => setTitleText(event.target.value)}
                        />
                    </label>
                    <label>
                        <span className="st-label">In (s)</span>
                        <input
                            className="st-input"
                            type="number"
                            min={0}
                            step={0.1}
                            value={titleStart}
                            onChange={(event) => setTitleStart(Number(event.target.value))}
                        />
                    </label>
                    <label>
                        <span className="st-label">Out (s)</span>
                        <input
                            className="st-input"
                            type="number"
                            min={0}
                            step={0.1}
                            value={titleEnd}
                            onChange={(event) => setTitleEnd(Number(event.target.value))}
                        />
                    </label>
                    <label>
                        <span className="st-label">Size</span>
                        <select
                            className="st-select"
                            value={titleSize}
                            onChange={(event) => setTitleSize(Number(event.target.value))}
                        >
                            {[32, 48, 64, 96].map((size) => (
                                <option key={size} value={size}>{size}px</option>
                            ))}
                        </select>
                    </label>
                    <label>
                        <span className="st-label">Color</span>
                        <input
                            className="st-input"
                            type="color"
                            value={titleColor}
                            onChange={(event) => setTitleColor(event.target.value)}
                        />
                    </label>
                    <button
                        type="button"
                        className="st-button"
                        data-size="sm"
                        disabled={!titleText.trim()}
                        onClick={() => {
                            const text = titleText.trim();
                            if (!text) return;
                            const start = Math.max(0, Math.round(titleStart * project.fps));
                            const end = Math.max(start + 1, Math.round(titleEnd * project.fps));
                            setProject({
                                ...project,
                                titles: [
                                    ...(project.titles || []),
                                    {
                                        id: globalThis.crypto?.randomUUID?.() || `title-${Date.now()}`,
                                        text,
                                        startFrame: start,
                                        endFrame: end,
                                        x: 0.5,
                                        y: 0.5,
                                        size: titleSize,
                                        color: titleColor,
                                        box: true,
                                    },
                                ],
                            });
                            void saveProject();
                            setTitleText('');
                            addToast('Title added', 'success');
                        }}
                    >
                        <Plus size={12} /> Add title
                    </button>
                </div>
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
                            <tr><th>Preview</th><th>Render</th><th>Status</th><th>Progress</th><th>Created</th><th>Output</th><th>Actions</th></tr>
                        </thead>
                        <tbody>
                            {renders.map((render) => (
                                <tr key={render.id}>
                                    <td>
                                        <button
                                            type="button"
                                            className="st-media-open st-render-thumb"
                                            disabled={!render.output_path}
                                            aria-label={`Preview render ${render.id}`}
                                            onClick={() => render.output_path && openMediaPreview({
                                                url: getAssetUrl(render.output_path) || render.output_path,
                                                title: render.id,
                                                subtitle: `${render.status} · ${render.progress}%`,
                                                kind: 'video',
                                            })}
                                        >
                                            <MediaThumb
                                                src={render.output_path}
                                                poster={render.thumbnail_path}
                                                alt={`Render ${render.id}`}
                                            />
                                        </button>
                                    </td>
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
