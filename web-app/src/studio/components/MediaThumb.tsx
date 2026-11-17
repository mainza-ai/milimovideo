import { useEffect, useRef, useState } from 'react';
import { Play } from 'lucide-react';
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
 * The parent click still opens the global media viewer.
 */
export function MediaThumb({ src, poster, alt = '', hovered }: Props) {
    const [localHover, setLocalHover] = useState(false);
    const videoRef = useRef<HTMLVideoElement>(null);
    const [failed, setFailed] = useState(false);
    const video = !!src && isVideo(src);
    const active = hovered ?? localHover;

    useEffect(() => {
        const element = videoRef.current;
        if (!element || !video) return;
        if (active) {
            element.play().catch(() => { /* autoplay can be denied; poster remains */ });
        } else {
            element.pause();
            try { element.currentTime = 0; } catch { /* metadata may not be ready */ }
        }
    }, [active, video]);

    if (!src && !poster) {
        return <div className="st-media-thumb st-media-thumb-empty" role="img" aria-label={alt || 'No media'} />;
    }

    return (
        <div
            className="st-media-thumb"
            onPointerEnter={() => setLocalHover(true)}
            onPointerLeave={() => setLocalHover(false)}
            data-hover={active || undefined}
        >
            {video && !failed ? (
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
                <img src={getAssetUrl(poster || src)} alt={alt} loading="lazy" onError={() => setFailed(true)} />
            )}
            {video && <span className="st-media-thumb-play" aria-hidden="true"><Play size={12} /></span>}
        </div>
    );
}
