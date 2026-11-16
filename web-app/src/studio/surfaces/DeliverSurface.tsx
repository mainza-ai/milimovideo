import { useState } from 'react';
import { Download, Film, Loader2, Send } from 'lucide-react';
import { useStudioCreateRender, useStudioProject, useStudioRenders } from '../data/useStudioData';
import { getAssetUrl } from '../../config';

const PRESETS = [
    { id: 'project', label: 'Project', width: null, height: null },
    { id: '1080p', label: '1080p', width: 1920, height: 1080 },
    { id: '720p', label: '720p', width: 1280, height: 720 },
    { id: 'vertical', label: 'Vertical 9:16', width: 1080, height: 1920 },
] as const;

export function DeliverSurface() {
    const project = useStudioProject();
    const { renders, refresh } = useStudioRenders(project.id);
    const createRender = useStudioCreateRender(project.id);
    const [preset, setPreset] = useState<typeof PRESETS[number]['id']>('project');
    const [busy, setBusy] = useState(false);

    const start = async () => {
        const selected = PRESETS.find((item) => item.id === preset) || PRESETS[0];
        setBusy(true);
        try {
            await createRender({
                width: selected.width || project.resolutionW,
                height: selected.height || project.resolutionH,
                fps: project.fps,
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
                    <h2 className="st-page-title">Deliver</h2>
                    <p className="st-page-subtitle">
                        Export the computed cut: V1/V2 timing, alpha overlays, embedded audio, music, and captions.
                        The render reads the same sequence order as the board and timeline.
                    </p>
                </div>
                <button type="button" className="st-button" data-variant="primary" disabled={busy} onClick={() => void start()}>
                    {busy ? <Loader2 size={13} className="animate-spin" /> : <Send size={13} />}
                    {busy ? 'Queueing…' : 'Create export'}
                </button>
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
                            onClick={() => setPreset(item.id)}
                        >
                            {item.label}
                            {item.width ? ` · ${item.width}×${item.height}` : ` · ${project.resolutionW}×${project.resolutionH}`}
                        </button>
                    ))}
                </div>
            </div>

            <div className="st-panel">
                <h3 className="st-panel-title">Render history</h3>
                {renders.length === 0 ? (
                    <div className="st-empty" style={{ minHeight: 120 }}>
                        <Film size={18} />
                        <strong>No exports yet</strong>
                        <span>Create one from the computed cut. The old export modal remains available in the legacy editor.</span>
                    </div>
                ) : (
                    <table className="st-table">
                        <thead>
                            <tr><th>Render</th><th>Status</th><th>Progress</th><th>Created</th><th>Output</th></tr>
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
                                </tr>
                            ))}
                        </tbody>
                    </table>
                )}
            </div>
        </div>
    );
}
