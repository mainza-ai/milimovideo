import { X } from 'lucide-react';
import type { Project, Shot } from '../../stores/types';
import { useTimelineStore } from '../../stores/timelineStore';
import { computeTimelineLayout } from '../../utils/timelineUtils';
import {
    opacityAtFrame, removeKeyframeAt, sanitizeKeyframes, upsertKeyframe,
} from '../timeline/keyframes';

interface Props {
    shot: Shot;
    project: Project;
    updateShot: (id: string, updates: Partial<Shot>) => void;
}

export function MotionPanel({ shot, project, updateShot }: Props) {
    const currentTime = useTimelineStore((state) => state.currentTime);
    const keyframes = sanitizeKeyframes(shot.keyframes || []);
    const clip = computeTimelineLayout(project).find((item) => item.id === shot.id);
    const fps = project.fps || 25;
    const localFrame = Math.max(0, Math.min(
        Math.max(0, (shot.numFrames || 1) - 1),
        clip ? Math.round((currentTime - clip.start) * fps) : 0,
    ));

    return (
        <div className="st-inspector-section">
            <span className="st-label">Motion keyframes · frame {localFrame}</span>
            <div className="st-value" style={{ color: 'var(--st-muted)', fontSize: 10 }}>
                Piecewise-linear opacity, scale, pan (x/y), and volume. Opacity/transform are burned into video; volume drives audio.
            </div>
            <div className="st-button-row" style={{ marginTop: 8 }}>
                <button
                    type="button"
                    className="st-button"
                    data-size="sm"
                    data-variant="primary"
                    onClick={() => updateShot(shot.id, {
                        keyframes: upsertKeyframe(keyframes, {
                            frame: localFrame,
                            opacity: opacityAtFrame(keyframes, localFrame),
                        }),
                    })}
                >
                    Add at {localFrame}
                </button>
                <button
                    type="button"
                    className="st-button"
                    data-size="sm"
                    data-variant="ghost"
                    disabled={keyframes.length === 0}
                    onClick={() => updateShot(shot.id, { keyframes: [] })}
                >
                    Clear
                </button>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginTop: 10 }}>
                {keyframes.length === 0 && (
                    <span className="st-value" style={{ color: 'var(--st-faint)', fontSize: 10 }}>
                        No keyframes. The clip stays fully opaque.
                    </span>
                )}
                {keyframes.map((keyframe) => {
                    const patchKeyframe = (patch: Partial<typeof keyframe>) => updateShot(shot.id, {
                        keyframes: upsertKeyframe(keyframes, { ...keyframe, ...patch }),
                    });
                    return (
                        <div key={keyframe.frame} className="st-keyframe">
                            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                                <span className="st-mono" style={{ width: 44, fontSize: 10 }}>
                                    f{keyframe.frame}
                                </span>
                                <span className="st-value" style={{ color: 'var(--st-faint)', fontSize: 9, flex: 1 }}>
                                    opacity {Math.round((keyframe.opacity ?? 1) * 100)}% · scale {Math.round((keyframe.scale ?? 1) * 100)}%
                                </span>
                                <button
                                    type="button"
                                    className="st-icon-button"
                                    aria-label={`Remove keyframe at ${keyframe.frame}`}
                                    onClick={() => updateShot(shot.id, {
                                        keyframes: removeKeyframeAt(keyframes, keyframe.frame),
                                    })}
                                >
                                    <X size={11} />
                                </button>
                            </div>
                            <label className="st-keyframe-row">
                                <span>Opacity</span>
                                <input
                                    type="range"
                                    min={0}
                                    max={100}
                                    value={Math.round((keyframe.opacity ?? 1) * 100)}
                                    aria-label={`Opacity at frame ${keyframe.frame}`}
                                    onChange={(event) => patchKeyframe({ opacity: Number(event.target.value) / 100 })}
                                />
                            </label>
                            <label className="st-keyframe-row">
                                <span>Scale</span>
                                <input
                                    type="range"
                                    min={25}
                                    max={400}
                                    value={Math.round((keyframe.scale ?? 1) * 100)}
                                    aria-label={`Scale at frame ${keyframe.frame}`}
                                    onChange={(event) => patchKeyframe({ scale: Number(event.target.value) / 100 })}
                                />
                            </label>
                            <label className="st-keyframe-row">
                                <span>Volume</span>
                                <input
                                    type="range"
                                    min={0}
                                    max={200}
                                    value={Math.round((keyframe.volume ?? 1) * 100)}
                                    aria-label={`Volume at frame ${keyframe.frame}`}
                                    onChange={(event) => patchKeyframe({ volume: Number(event.target.value) / 100 })}
                                />
                            </label>
                            <div className="st-row">
                                <label>
                                    <span className="st-label">Pan X · {Math.round((keyframe.x ?? 0) * 100)}</span>
                                    <input
                                        className="st-input"
                                        type="number"
                                        min={-100}
                                        max={100}
                                        value={Math.round((keyframe.x ?? 0) * 100)}
                                        onChange={(event) => patchKeyframe({
                                            x: Math.max(-1, Math.min(1, Number(event.target.value) / 100)),
                                        })}
                                    />
                                </label>
                                <label>
                                    <span className="st-label">Pan Y · {Math.round((keyframe.y ?? 0) * 100)}</span>
                                    <input
                                        className="st-input"
                                        type="number"
                                        min={-100}
                                        max={100}
                                        value={Math.round((keyframe.y ?? 0) * 100)}
                                        onChange={(event) => patchKeyframe({
                                            y: Math.max(-1, Math.min(1, Number(event.target.value) / 100)),
                                        })}
                                    />
                                </label>
                            </div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
}
