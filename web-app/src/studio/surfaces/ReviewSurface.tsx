import { useMemo, useRef, useState } from 'react';
import {
    CheckCircle2, Check, MessageSquare, Play, Trash2, TriangleAlert, X,
} from 'lucide-react';
import { useTimelineStore } from '../../stores/timelineStore';
import { getAssetUrl } from '../../config';
import { useStudioEditor } from '../state/editorStore';
import {
    useAssetUrl,
    useStudioReview,
    useStudioSelectTake,
    useStudioSelection,
    useStudioTakes,
} from '../data/useStudioData';
import { computeTimelineLayout } from '../../utils/timelineUtils';

export function ReviewSurface() {
    const project = useTimelineStore((state) => state.project);
    const { selectedShotId, selectShot, setCurrentTime } = useStudioSelection();
    const { setInspectorTab, setCommandDraft, setCommandMode } = useStudioEditor();
    const shot = project.shots.find((item) => item.id === selectedShotId) || project.shots[0] || null;
    const mediaUrl = useAssetUrl(shot?.videoUrl || shot?.thumbnailUrl);
    const isImage = !!mediaUrl?.match(/\.(jpg|jpeg|png|webp)$/i);
    const layout = useMemo(() => computeTimelineLayout(project), [project]);
    const { takes, refresh: refreshTakes } = useStudioTakes(shot?.id);
    const selectTake = useStudioSelectTake();
    const {
        review, loading, addComment, updateComment, deleteComment, setApproval,
    } = useStudioReview(project.id);

    const videoRef = useRef<HTMLVideoElement>(null);
    const [frame, setFrame] = useState(0);
    const [pendingFrame, setPendingFrame] = useState<number | null>(null);
    const [draft, setDraft] = useState('');
    const [busy, setBusy] = useState(false);

    const fps = project.fps || 25;
    const approval = shot ? review.approvals.find((item) => item.shot_id === shot.id) : undefined;
    const comments = review.comments;
    const shotComments = shot ? comments.filter((item) => item.shot_id === shot.id) : [];

    const seekToFrame = (target: number, shotId?: string | null) => {
        if (shotId && shotId !== shot?.id) {
            selectShot(shotId);
            setPendingFrame(target);
            setFrame(target);
            return;
        }
        const video = videoRef.current;
        if (video) {
            try { video.currentTime = target / fps; } catch { /* metadata not ready */ }
        }
        setFrame(target);
        setPendingFrame(null);
    };

    const submitComment = async () => {
        if (!draft.trim()) return;
        setBusy(true);
        try {
            if (await addComment(shot?.id || null, frame, draft.trim())) setDraft('');
        } finally {
            setBusy(false);
        }
    };

    return (
        <div className="st-page">
            <div className="st-page-head">
                <div>
                    <h2 className="st-page-title">Review</h2>
                    <p className="st-page-subtitle">
                        Frame-accurate comments and per-shot decisions persist on the project.
                        The cut stays the source of truth; comments attach to a shot and frame.
                    </p>
                </div>
                <div className="st-button-row">
                    <span className="st-status-pill">
                        <MessageSquare size={11} /> {review.summary.open} open · {review.summary.comments} total
                    </span>
                    <span className="st-status-pill" data-tone={review.summary.changes_requested > 0 ? 'warn' : 'good'}>
                        <CheckCircle2 size={11} /> {review.summary.approved} approved
                    </span>
                </div>
            </div>

            <div className="st-review-grid">
                <div>
                    <div className="st-panel" style={{ padding: 0, overflow: 'hidden' }}>
                        <div style={{ aspectRatio: '16 / 9', background: '#000', position: 'relative' }}>
                            {mediaUrl ? (
                                isImage ? (
                                    <img src={mediaUrl} alt="" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
                                ) : (
                                    <video
                                        ref={videoRef}
                                        src={mediaUrl}
                                        muted
                                        controls
                                        playsInline
                                        style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                                        onTimeUpdate={(event) => setFrame(Math.floor(event.currentTarget.currentTime * fps))}
                                        onLoadedMetadata={(event) => {
                                            if (pendingFrame != null) {
                                                try { event.currentTarget.currentTime = pendingFrame / fps; } catch { /* ignore */ }
                                                setFrame(pendingFrame);
                                                setPendingFrame(null);
                                            }
                                        }}
                                    />
                                )
                            ) : (
                                <div className="st-stage-empty"><Play size={18} /><strong>No media selected</strong></div>
                            )}
                            <div className="st-stage-overlay">
                                <span className="st-status-pill st-mono">frame {frame}</span>
                                <span className="st-status-pill st-mono">{shot?.id ? shot.id.slice(0, 8) : 'shot'}</span>
                                {approval && (
                                    <span
                                        className="st-status-pill"
                                        data-tone={approval.status === 'approved' ? 'good' : 'warn'}
                                    >
                                        {approval.status === 'approved' ? <CheckCircle2 size={10} /> : <TriangleAlert size={10} />}
                                        {approval.status.replace('_', ' ')}
                                    </span>
                                )}
                            </div>
                        </div>
                    </div>

                    <div className="st-grid" style={{ marginTop: 14 }}>
                        {layout.filter((clip) => clip.track === 0).map((clip, index) => (
                            <button
                                key={clip.id}
                                type="button"
                                className="st-take"
                                style={{ textAlign: 'left', borderColor: clip.id === shot?.id ? 'var(--st-accent)' : undefined }}
                                onClick={() => {
                                    selectShot(clip.id);
                                    setCurrentTime(0);
                                    seekToFrame(0, clip.id);
                                }}
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
                                    {review.approvals.find((item) => item.shot_id === clip.id) && (
                                        <div style={{ color: review.approvals.find((item) => item.shot_id === clip.id)?.status === 'approved' ? 'var(--st-success)' : 'var(--st-warning)' }}>
                                            {review.approvals.find((item) => item.shot_id === clip.id)?.status.replace('_', ' ')}
                                        </div>
                                    )}
                                </div>
                            </button>
                        ))}
                    </div>
                </div>

                <aside className="st-panel">
                    <h3 className="st-panel-title">Decision · {shot?.id?.slice(0, 8) || 'shot'}</h3>
                    <div className="st-button-row" style={{ marginBottom: 14 }}>
                        <button
                            type="button"
                            className="st-button"
                            data-variant={approval?.status === 'approved' ? 'primary' : undefined}
                            disabled={!shot}
                            onClick={() => shot && void setApproval(shot.id, 'approved', draft.trim())}
                        >
                            <CheckCircle2 size={12} /> Approve shot
                        </button>
                        <button
                            type="button"
                            className="st-button"
                            data-variant={approval?.status === 'changes_requested' ? 'danger' : undefined}
                            disabled={!shot}
                            onClick={() => shot && void setApproval(shot.id, 'changes_requested', draft.trim())}
                        >
                            <TriangleAlert size={12} /> Request changes
                        </button>
                    </div>

                    <h3 className="st-panel-title">Comment at frame {frame}</h3>
                    <textarea
                        className="st-textarea"
                        style={{ minHeight: 64 }}
                        placeholder="What needs to change at this frame? Mention @Agent to route it to the plan dock."
                        value={draft}
                        onChange={(event) => setDraft(event.target.value)}
                    />
                    <div className="st-button-row" style={{ marginTop: 8 }}>
                        <button
                            type="button"
                            className="st-button"
                            data-variant="primary"
                            data-size="sm"
                            disabled={busy || !draft.trim()}
                            onClick={() => void submitComment()}
                        >
                            <MessageSquare size={12} /> Add comment
                        </button>
                        <button
                            type="button"
                            className="st-button"
                            data-size="sm"
                            onClick={() => setInspectorTab('review')}
                        >
                            Inspector
                        </button>
                    </div>

                    <h3 className="st-panel-title" style={{ marginTop: 16 }}>
                        Comments · {shotComments.length} on this shot
                    </h3>
                    <div className="st-comment-list">
                        {loading && <div className="st-comment">Loading review…</div>}
                        {!loading && comments.length === 0 && (
                            <div className="st-comment">
                                No comments yet. Pause on a frame, write the note, and it persists on the project.
                            </div>
                        )}
                        {comments.map((comment) => (
                            <div
                                key={comment.id}
                                className="st-comment"
                                data-resolved={comment.status === 'resolved'}
                                style={comment.status === 'resolved' ? { opacity: 0.55 } : undefined}
                            >
                                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 6 }}>
                                    <button
                                        type="button"
                                        className="st-button"
                                        data-size="sm"
                                        data-variant="ghost"
                                        onClick={() => seekToFrame(comment.frame, comment.shot_id)}
                                        title="Jump to this frame"
                                    >
                                        <span className="st-mono">f{comment.frame}</span>
                                        <span>{comment.shot_id?.slice(0, 8) || 'project'}</span>
                                    </button>
                                    <span style={{ display: 'flex', gap: 4 }}>
                                        <button
                                            type="button"
                                            className="st-icon-button"
                                            aria-label={comment.status === 'resolved' ? 'Reopen comment' : 'Resolve comment'}
                                            onClick={() => void updateComment(comment.id, {
                                                status: comment.status === 'resolved' ? 'open' : 'resolved',
                                            })}
                                        >
                                            {comment.status === 'resolved' ? <X size={12} /> : <Check size={12} />}
                                        </button>
                                        <button
                                            type="button"
                                            className="st-icon-button"
                                            aria-label="Delete comment"
                                            onClick={() => void deleteComment(comment.id)}
                                        >
                                            <Trash2 size={12} />
                                        </button>
                                    </span>
                                </div>
                                <p style={{ margin: '6px 0 0' }}>{comment.body}</p>
                                <div style={{ marginTop: 4, color: 'var(--st-faint)', fontSize: 9 }}>
                                    {comment.author} · {comment.status}
                                    {comment.created_at ? ` · ${new Date(comment.created_at).toLocaleString()}` : ''}
                                </div>
                                {/@agent/i.test(comment.body) && (
                                    <button
                                        type="button"
                                        className="st-button"
                                        data-size="sm"
                                        style={{ marginTop: 6 }}
                                        onClick={() => {
                                            if (comment.shot_id) selectShot(comment.shot_id);
                                            setCommandMode('direct');
                                            setCommandDraft(comment.body.replace(/@agent/gi, '').trim());
                                        }}
                                    >
                                        <Check size={12} /> Plan fix in command dock
                                    </button>
                                )}
                            </div>
                        ))}
                    </div>

                    <h3 className="st-panel-title" style={{ marginTop: 16 }}>Version stack · {takes.length}</h3>
                    <div className="st-comment-list">
                        {takes.length === 0 && <div className="st-comment">No completed takes for this shot yet.</div>}
                        {takes.map((take) => (
                            <button
                                key={take.job_id}
                                type="button"
                                className="st-take"
                                style={{ textAlign: 'left', borderColor: take.selected ? 'var(--st-accent)' : undefined }}
                                onClick={async () => {
                                    if (shot && await selectTake(shot, take)) await refreshTakes();
                                }}
                            >
                                <div className="st-take-media">
                                    {take.thumbnail_url || take.url
                                        ? <img src={getAssetUrl(take.thumbnail_url || take.url)} alt="" loading="lazy" />
                                        : <div className="st-stage-empty"><Play size={12} /></div>}
                                </div>
                                <div className="st-take-body">
                                    <strong style={{ color: 'var(--st-text)' }}>{take.selected ? 'Selected' : 'Take'} · {take.model || take.type}</strong>
                                    <div className="st-mono">{take.job_id.slice(0, 12)} · seed {take.seed ?? '—'}</div>
                                </div>
                            </button>
                        ))}
                    </div>
                </aside>
            </div>
        </div>
    );
}
