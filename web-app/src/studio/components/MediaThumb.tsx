import { useEffect, useRef, useState } from 'react';
import { ImageOff, Play } from 'lucide-react';
import { getAssetUrl } from '../../config';

interface Props {
    src?: string | null;
    poster?: string | null;
    alt?: string;
    hovered?: boolean;
}

function isVideo(url: string): boolean {
    return /\.(mp4|webm|mov|m4v)(\?|$)/i.test(url);
}

/**
 * Video posters + muted hover playback for Library/Story/Generate/Review cards.
 * The parent click still opens the global media viewer. A failed video or image
 * becomes an explicit placeholder instead of the browser's broken-image icon.
 */
export function MediaThumb({ src, poster, alt = '', hovered }: Props) {
    const [localHover, setLocalHover] = useState(false);
    const videoRef = useRef<HTMLVideoElement>(null);
    const [failed, setFailed] = useState(false);
    const video = !!src && isVideo(src);
    const active = hovered ?? localHover;

    useEffect(() => { setFailed(false); }, [src, poster]);

    useEffect(() => {
        const element = videoRef.current;
        if (!element || !video || failed) return;
        if (active) {
            element.play().catch(() => { /* autoplay can be denied; poster remains */ });
        } else {
            element.pause();
            try { element.currentTime = 0; } catch { /* metadata may not be ready */ }
        }
    }, [active, video, failed]);

    if (!src && !poster) {
        return <div className="st-media-thumb st-media-thumb-empty" role="img" aria-label={alt || 'No media'} />;
    }

    if (failed) {
        return (
            <div className="st-media-thumb st-media-thumb-empty" role="img" aria-label={`${alt || 'Media'} unavailable`}>
                <ImageOff size={16} />
                <span>Media unavailable</span>
            </div>
        );
    }

    return (
        <div
            className="st-media-thumb"
            onPointerEnter={() => setLocalHover(true)}
            onPointerLeave={() => setLocalHover(false)}
            data-hover={active || undefined}
        >
            {video ? (
                <video
                    ref={videoRef}
                    src={getAssetUrl(src)}
                    poster={poster ? getAssetUrl(poster) : undefined}
                    muted
                    loop
                    playsInline
                    preload="metadata"
                    aria-label={alt}
                    onError={() => setFailed(true)}
                />
            ) : (
                <img
                    src={getAssetUrl(poster || src)}
                    alt={alt}
                    loading="lazy"
                    onError={() => setFailed(true)}
                />
            )}
            {video && <span className="st-media-thumb-play" aria-hidden="true"><Play size={12} /></span>}
        </div>
    );
}
