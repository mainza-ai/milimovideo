import { Clapperboard, Copy, Eye, GripVertical, Image as ImageIcon, Loader2, Sparkles } from 'lucide-react';
import type { Shot } from '../state/documentStore';
import { useAssetUrl } from '../data/useStudioData';
import type { BoardPosition } from '../state/editorStore';

interface Props {
    shot: Shot;
    index: number;
    position: BoardPosition;
    selected: boolean;
    pinned: boolean;
    linkSource?: boolean;
    onSelect: (event: React.MouseEvent) => void;
    onDragStart: (event: React.PointerEvent, shotId: string) => void;
    onResize: (event: React.PointerEvent, shotId: string) => void;
    onGenerate: (shotId: string) => void;
    onEdit: (shotId: string) => void;
    onPreview: (shotId: string) => void;
    onDuplicate: (shotId: string) => void;
    onContextMenu: (event: React.MouseEvent, shotId: string) => void;
}

export function ShotCard({
    shot, index, position, selected, pinned, linkSource, onSelect, onDragStart, onResize,
    onGenerate, onEdit, onPreview, onDuplicate, onContextMenu,
}: Props) {
    const videoUrl = useAssetUrl(shot.videoUrl);
    const thumbnailUrl = useAssetUrl(shot.thumbnailUrl);
    const media = videoUrl || thumbnailUrl;
    const isImage = !!media?.match(/\.(jpg|jpeg|png|webp)$/i);

    return (
        <div
            className="st-shot-card"
            data-selected={selected}
            data-pinned={pinned}
            data-link-source={linkSource || undefined}
            style={{
                left: position.x,
                top: position.y,
                width: position.width,
                height: position.height,
            }}
            onPointerDown={(event) => onDragStart(event, shot.id)}
            onContextMenu={(event) => onContextMenu(event, shot.id)}
        >
            <button
                type="button"
                className="st-shot-card-select"
                onClick={(event) => onSelect(event)}
                onDoubleClick={() => onPreview(shot.id)}
                aria-label={`Shot ${index + 1}: ${shot.action || shot.prompt || 'Untitled'}`}
                title="Click to select · double-click to preview"
            />
            <div
                className="st-shot-card-media"
                style={{ height: Math.max(70, Math.min(200, position.height * 0.52)) }}
            >
                {media ? (
                    isImage ? <img src={media} alt="" loading="lazy" /> : <video src={media} muted playsInline preload="metadata" />
                ) : (
                    <div className="st-media-fallback">
                        <ImageIcon size={22} />
                        <span style={{ marginLeft: 6 }}>No preview</span>
                    </div>
                )}
                <span className="st-shot-card-badge">{shot.shotType || `S${index + 1}`}</span>
                <span className="st-shot-card-status" data-status={shot.status || 'pending'}>
                    {shot.isGenerating ? <Loader2 size={9} className="animate-spin" /> : (shot.status || 'pending')}
                </span>
                <div
                    className="st-shot-card-actions"
                    onPointerDown={(event) => event.stopPropagation()}
                    onClick={(event) => event.stopPropagation()}
                >
                    <button type="button" aria-label={`Generate ${shot.action || shot.prompt || 'shot'}`} title="Generate" onClick={() => onGenerate(shot.id)}>
                        <Sparkles size={12} />
                    </button>
                    <button type="button" aria-label={`Preview ${shot.action || shot.prompt || 'shot'}`} title="Preview media" onClick={() => onPreview(shot.id)}>
                        <Eye size={12} />
                    </button>
                    <button type="button" aria-label={`Open ${shot.action || shot.prompt || 'shot'} in Edit`} title="Open in Edit" onClick={() => onEdit(shot.id)}>
                        <Clapperboard size={12} />
                    </button>
                    <button type="button" aria-label={`Duplicate ${shot.action || shot.prompt || 'shot'}`} title="Duplicate" onClick={() => onDuplicate(shot.id)}>
                        <Copy size={12} />
                    </button>
                </div>
            </div>
            <div className="st-shot-card-body">
                <div className="st-shot-card-title">{shot.action || shot.prompt || 'Untitled shot'}</div>
                {shot.matchedElements?.length ? (
                    <div className="st-chip-row">
                        {shot.matchedElements.slice(0, 3).map((match) => (
                            <span className="st-chip" key={match.element_id}>@{match.trigger_word.replace(/^@/, '')}</span>
                        ))}
                    </div>
                ) : null}
                <div className="st-shot-card-meta">
                    <span className="st-mono">{shot.numFrames}f · {shot.width}×{shot.height}</span>
                    <span><GripVertical size={10} /></span>
                </div>
            </div>
            <span
                className="st-shot-card-resize"
                aria-hidden="true"
                onPointerDown={(event) => onResize(event, shot.id)}
            />
        </div>
    );
}
