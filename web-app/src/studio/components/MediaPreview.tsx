import { useEffect, useState } from 'react';
import { Download, ExternalLink, X } from 'lucide-react';
import { useMediaPreview } from '../state/mediaPreviewStore';

function inferKind(url: string): 'image' | 'video' | 'audio' | 'text' {
    if (/\.(mp4|webm|mov|m4v)(\?|$)/i.test(url)) return 'video';
    if (/\.(mp3|wav|ogg|m4a|aac|flac)(\?|$)/i.test(url)) return 'audio';
    if (/\.(srt|vtt|json|txt|md)(\?|$)/i.test(url)) return 'text';
    return 'image';
}

export function MediaPreview() {
    const { item, close } = useMediaPreview();
    const [text, setText] = useState<string | null>(null);

    useEffect(() => {
        if (!item) return;
        const onKeyDown = (event: KeyboardEvent) => {
            if (event.key === 'Escape') close();
        };
        window.addEventListener('keydown', onKeyDown);
        return () => window.removeEventListener('keydown', onKeyDown);
    }, [close, item]);

    const kind = item ? (item.kind || inferKind(item.url)) : 'image';

    useEffect(() => {
        if (!item || kind !== 'text') {
            setText(null);
            return;
        }
        let alive = true;
        fetch(item.url)
            .then((response) => response.text())
            .then((body) => { if (alive) setText(body); })
            .catch(() => { if (alive) setText('Could not read this file.'); });
        return () => { alive = false; };
    }, [item, kind]);

    if (!item) return null;

    return (
        <div className="st-media-backdrop" role="presentation" onPointerDown={close}>
            <div
                className="st-media-preview"
                role="dialog"
                aria-modal="true"
                aria-label={item.title || 'Media preview'}
                onPointerDown={(event) => event.stopPropagation()}
            >
                <div className="st-media-preview-head">
                    <div style={{ minWidth: 0 }}>
                        <strong>{item.title || 'Preview'}</strong>
                        {item.subtitle && <span>{item.subtitle}</span>}
                    </div>
                    <div className="st-button-row">
                        <a className="st-icon-button" href={item.url} target="_blank" rel="noreferrer" aria-label="Open original" title="Open original">
                            <ExternalLink size={15} />
                        </a>
                        <a className="st-icon-button" href={item.url} download aria-label="Download media" title="Download">
                            <Download size={15} />
                        </a>
                        <button type="button" className="st-icon-button" aria-label="Close media preview" onClick={close}>
                            <X size={16} />
                        </button>
                    </div>
                </div>
                <div className="st-media-preview-body">
                    {kind === 'video' ? (
                        <video src={item.url} controls autoPlay playsInline />
                    ) : kind === 'audio' ? (
                        <audio src={item.url} controls autoPlay />
                    ) : kind === 'text' ? (
                        <pre className="st-media-text">{text ?? 'Loading…'}</pre>
                    ) : (
                        <img src={item.url} alt={item.title || 'Media preview'} />
                    )}
                </div>
            </div>
        </div>
    );
}
