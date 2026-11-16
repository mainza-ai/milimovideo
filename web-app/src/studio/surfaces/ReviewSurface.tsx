import { useMemo } from 'react';
import { CheckCircle2, MessageSquare, Play } from 'lucide-react';
import { useTimelineStore } from '../../stores/timelineStore';
import { getAssetUrl } from '../../config';
import { useStudioSelection, useAssetUrl } from '../data/useStudioData';
import { computeTimelineLayout } from '../../utils/timelineUtils';

export function ReviewSurface() {
    const project = useTimelineStore((state) => state.project);
    const { selectedShotId, selectShot } = useStudioSelection();
    const shot = project.shots.find((item) => item.id === selectedShotId) || project.shots[0] || null;
    const mediaUrl = useAssetUrl(shot?.videoUrl || shot?.thumbnailUrl);
    const isImage = !!mediaUrl?.match(/\.(jpg|jpeg|png|webp)$/i);
    const layout = useMemo(() => computeTimelineLayout(project), [project]);

    return (
        <div className="st-page">
            <div className="st-page-head">
                <div>
                    <h2 className="st-page-title">Review</h2>
                    <p className="st-page-subtitle">
                        The cut is the source of truth. Version stacks, frame comments, and approvals land in F4;
                        this is the review surface shell over the current cut.
                    </p>
                </div>
                <span className="st-status-pill"><CheckCircle2 size={11} /> {project.shots.filter((s) => s.status === 'completed').length} ready</span>
            </div>

            <div className="st-review-grid">
                <div>
                    <div className="st-panel" style={{ padding: 0, overflow: 'hidden' }}>
                        <div style={{ aspectRatio: '16 / 9', background: '#000' }}>
                            {mediaUrl ? (
                                isImage ? <img src={mediaUrl} alt="" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
                                    : <video src={mediaUrl} muted controls style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
                            ) : (
                                <div className="st-stage-empty"><Play size={18} /><strong>No media selected</strong></div>
                            )}
                        </div>
                    </div>

                    <div className="st-grid" style={{ marginTop: 14 }}>
                        {layout.filter((clip) => clip.track === 0).map((clip, index) => (
                            <button
                                key={clip.id}
                                type="button"
                                className="st-take"
                                style={{ textAlign: 'left', borderColor: clip.id === shot?.id ? 'var(--st-accent)' : undefined }}
                                onClick={() => selectShot(clip.id)}
                            >
                                <div className="st-take-media">
                                    {clip.shot.thumbnailUrl || clip.shot.videoUrl ? (
                                        <img src={getAssetUrl(clip.shot.thumbnailUrl || clip.shot.videoUrl)} alt="" loading="lazy" />
                                    ) : (
                                        <div className="st-stage-empty"><Play size={14} /></div>
                                    )}
                                </div>
                                <div className="st-take-body">
                                    <strong style={{ color: 'var(--st-text)' }}>{index + 1}. {clip.shot.action || clip.shot.prompt || 'Shot'}</strong>
                                    <div>{clip.shot.status || 'pending'} · {clip.shot.numFrames}f</div>
                                </div>
                            </button>
                        ))}
                    </div>
                </div>

                <aside className="st-panel">
                    <h3 className="st-panel-title">Comments</h3>
                    <div className="st-comment-list">
                        <div className="st-comment">
                            <MessageSquare size={13} />
                            <p style={{ margin: '6px 0 0' }}>
                                Frame-accurate comments, mentions, and approvals are the F4 review contract.
                                The current cut has {project.shots.length} shots and {layout.length} timeline clips.
                            </p>
                        </div>
                        <div className="st-comment">
                            <strong style={{ color: 'var(--st-secondary)' }}>Needs you</strong>
                            <p style={{ margin: '6px 0 0' }}>No pending review items. Share links and reviewer inbox arrive with the review API.</p>
                        </div>
                    </div>
                </aside>
            </div>
        </div>
    );
}
