import { GripVertical, Image as ImageIcon, Loader2 } from 'lucide-react';
import type { Shot } from '../../stores/timelineStore';
import { useAssetUrl } from '../data/useStudioData';
import type { BoardPosition } from '../state/editorStore';

interface Props {
    shot: Shot;
    index: number;
    position: BoardPosition;
    selected: boolean;
    pinned: boolean;
    onSelect: () => void;
    onDragStart: (event: React.PointerEvent, shotId: string) => void;
}

export function ShotCard({ shot, index, position, selected, pinned, onSelect, onDragStart }: Props) {
    const videoUrl = useAssetUrl(shot.videoUrl);
    const thumbnailUrl = useAssetUrl(shot.thumbnailUrl);
    const media = videoUrl || thumbnailUrl;
    const isImage = !!media?.match(/\.(jpg|jpeg|png|webp)$/i);

    return (
        <button
            type="button"
            className="st-shot-card"
            data-selected={selected}
            data-pinned={pinned}
            style={{
                left: position.x,
                top: position.y,
                width: position.width,
                height: position.height,
            }}
            onPointerDown={(event) => onDragStart(event, shot.id)}
            onClick={onSelect}
            aria-label={`Shot ${index + 1}: ${shot.action || shot.prompt || 'Untitled'}`}
        >
            <div className="st-shot-card-media">
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
        </button>
    );
}
