import { useEffect, useMemo, useRef, useState } from 'react';
import {
    Check, CheckCircle2, Loader2, MessageSquare, Play, TriangleAlert, X,
} from 'lucide-react';
import { apiUrl, getAssetUrl } from '../config';
import { SharePayloadSchema } from '../api/contracts';
import './studio.css';

interface SharedShot {
    id: string;
    sequence_index: number | null;
    action?: string | null;
    prompt?: string | null;
    dialogue?: string | null;
    status?: string | null;
    num_frames?: number | null;
    video_url?: string | null;
    thumbnail_url?: string | null;
}

interface SharedComment {
    id: string;
    shot_id: string | null;
    kind: 'comment' | 'approval';
    frame: number;
    body: string;
    author: string;
    status: string;
    created_at: string | null;
}

interface SharedPayload {
    shared: boolean;
    project: { id: string; name: string; resolution_w: number; resolution_h: number; fps: number };
    shots: SharedShot[];
    comments: SharedComment[];
    approvals: SharedComment[];
    summary: { comments: number; open: number; approved: number; changes_requested: number };
}

export function SharedReviewPage({ token }: { token: string }) {
    const [payload, setPayload] = useState<SharedPayload | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [loading, setLoading] = useState(true);
    const [selectedShotId, setSelectedShotId] = useState<string | null>(null);
    const [author, setAuthor] = useState('Reviewer');
    const [draft, setDraft] = useState('');
    const [frame, setFrame] = useState(0);
    const [busy, setBusy] = useState(false);
    const videoRef = useRef<HTMLVideoElement>(null);

    const refresh = async () => {
        try {
            const response = await fetch(apiUrl(`/share/${token}`));
            const data = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
            const parsed = SharePayloadSchema.safeParse(data);
            if (!parsed.success) throw new Error('Share response failed contract validation');
            setPayload(parsed.data as SharedPayload);
            setSelectedShotId((current) => current || data.shots?.[0]?.id || null);
            setError(null);
        } catch (exc) {
            setError(exc instanceof Error ? exc.message : 'Could not open this review link');
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => { void refresh(); }, [token]);

    const shot = useMemo(
        () => payload?.shots.find((item) => item.id === selectedShotId) || payload?.shots[0] || null,
        [payload, selectedShotId],
    );
    const fps = payload?.project.fps || 25;
    const mediaUrl = getAssetUrl(shot?.video_url || shot?.thumbnail_url);
    const isImage = !!mediaUrl?.match(/\.(jpg|jpeg|png|webp)$/i);
    const approval = payload?.approvals.find((item) => item.shot_id === shot?.id);

    const submitComment = async () => {
        if (!shot || !draft.trim()) return;
        setBusy(true);
        try {
            const response = await fetch(apiUrl(`/share/${token}/comments`), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    shot_id: shot.id,
                    frame,
                    body: draft.trim(),
                    author: author.trim() || 'Reviewer',
                }),
            });
            if (response.ok) {
                setDraft('');
                await refresh();
            }
        } finally {
            setBusy(false);
        }
    };

    const decide = async (status: 'approved' | 'changes_requested') => {
        if (!shot) return;
        setBusy(true);
        try {
            const response = await fetch(apiUrl(`/share/${token}/approvals`), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    shot_id: shot.id,
                    status,
                    note: draft.trim(),
                    author: author.trim() || 'Reviewer',
                }),
            });
            if (response.ok) {
                await refresh();
            }
        } finally {
            setBusy(false);
        }
    };

    if (loading) {
        return (
            <div className="studio-root" style={{ alignItems: 'center', justifyContent: 'center' }}>
                <Loader2 size={20} className="animate-spin" />
            </div>
        );
    }

    if (error || !payload) {
        return (
            <div className="studio-root" style={{ alignItems: 'center', justifyContent: 'center', gap: 8 }}>
                <TriangleAlert size={20} style={{ color: 'var(--st-danger)' }} />
                <strong>{error || 'Share link unavailable'}</strong>
                <span style={{ color: 'var(--st-muted)', fontSize: 11 }}>
                    The link may have been revoked by the editor.
                </span>
            </div>
        );
    }

    return (
        <div className="studio-root" style={{ overflowY: 'auto', display: 'block' }}>
            <div className="st-page" style={{ maxWidth: 1400, margin: '0 auto' }}>
                <div className="st-page-head">
                    <div>
                        <h2 className="st-page-title">{payload.project.name}</h2>
                        <p className="st-page-subtitle">
                            Shared review · read-only cut. Comments and decisions are saved to the project.
                        </p>
                    </div>
                    <div className="st-button-row">
                        <span className="st-status-pill"><MessageSquare size={11} /> {payload.summary.open} open</span>
                        <span className="st-status-pill" data-tone={payload.summary.changes_requested ? 'warn' : 'good'}>
                            <CheckCircle2 size={11} /> {payload.summary.approved} approved
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
                                            controls
                                            playsInline
                                            style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                                            onTimeUpdate={(event) => setFrame(Math.floor(event.currentTarget.currentTime * fps))}
                                        />
                                    )
                                ) : (
                                    <div className="st-stage-empty"><Play size={18} /><strong>No media for this shot</strong></div>
                                )}
                                <div className="st-stage-overlay">
                                    <span className="st-status-pill st-mono">frame {frame}</span>
                                    {approval && (
                                        <span className="st-status-pill" data-tone={approval.status === 'approved' ? 'good' : 'warn'}>
                                            {approval.status.replace('_', ' ')}
                                        </span>
                                    )}
                                </div>
                            </div>
                        </div>

                        <div className="st-button-row" style={{ marginTop: 10 }}>
                            {payload.shots.map((item, index) => (
                                <button
                                    key={item.id}
                                    type="button"
                                    className="st-button"
                                    data-size="sm"
                                    data-variant={item.id === shot?.id ? 'primary' : 'ghost'}
                                    onClick={() => {
                                        setSelectedShotId(item.id);
                                        setFrame(0);
                                    }}
                                >
                                    {item.sequence_index != null ? item.sequence_index + 1 : index + 1}. {item.action || item.prompt || 'Shot'}
                                </button>
                            ))}
                        </div>
                    </div>

                    <aside className="st-panel">
                        <h3 className="st-panel-title">Your review</h3>
                        <label className="st-label" htmlFor="share-author">Name</label>
                        <input
                            id="share-author"
                            className="st-input"
                            value={author}
                            onChange={(event) => setAuthor(event.target.value)}
                        />
                        <label className="st-label" htmlFor="share-comment" style={{ marginTop: 10 }}>
                            Comment at frame {frame}
                        </label>
                        <textarea
                            id="share-comment"
                            className="st-textarea"
                            style={{ minHeight: 64 }}
                            value={draft}
                            placeholder="What needs to change at this frame?"
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
                                data-variant={approval?.status === 'approved' ? 'primary' : undefined}
                                disabled={busy}
                                onClick={() => void decide('approved')}
                            >
                                <Check size={12} /> Approve
                            </button>
                            <button
                                type="button"
                                className="st-button"
                                data-size="sm"
                                data-variant={approval?.status === 'changes_requested' ? 'danger' : undefined}
                                disabled={busy}
                                onClick={() => void decide('changes_requested')}
                            >
                                <X size={12} /> Changes
                            </button>
                        </div>

                        <h3 className="st-panel-title" style={{ marginTop: 16 }}>Comments</h3>
                        <div className="st-comment-list">
                            {payload.comments.length === 0 && (
                                <div className="st-comment">No comments yet.</div>
                            )}
                            {payload.comments.map((comment) => (
                                <div key={comment.id} className="st-comment" style={comment.status === 'resolved' ? { opacity: 0.55 } : undefined}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', gap: 6 }}>
                                        <button
                                            type="button"
                                            className="st-button"
                                            data-size="sm"
                                            data-variant="ghost"
                                            onClick={() => {
                                                if (comment.shot_id) setSelectedShotId(comment.shot_id);
                                                setFrame(comment.frame);
                                                if (videoRef.current && comment.shot_id === shot?.id) {
                                                    try { videoRef.current.currentTime = comment.frame / fps; } catch { /* ignore */ }
                                                }
                                            }}
                                        >
                                            <span className="st-mono">f{comment.frame}</span>
                                        </button>
                                        <span style={{ color: 'var(--st-faint)', fontSize: 11 }}>{comment.status}</span>
                                    </div>
                                    <p style={{ margin: '6px 0 0' }}>{comment.body}</p>
                                    <div style={{ color: 'var(--st-faint)', fontSize: 11, marginTop: 4 }}>
                                        {comment.author}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </aside>
                </div>
            </div>
        </div>
    );
}
