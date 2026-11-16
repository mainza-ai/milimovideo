import { useMemo, useState } from 'react';
import { Image as ImageIcon, Library, RefreshCw, Video } from 'lucide-react';
import { getAssetUrl } from '../../config';
import { useStudioAssets, useStudioElements, useStudioProject } from '../data/useStudioData';

export function LibrarySurface() {
    const project = useStudioProject();
    const elements = useStudioElements(project.id);
    const { assets, loading, refresh } = useStudioAssets();
    const [tab, setTab] = useState<'assets' | 'references'>('assets');
    const [query, setQuery] = useState('');

    const filteredAssets = useMemo(() => {
        const q = query.trim().toLowerCase();
        return q ? assets.filter((asset) => asset.filename.toLowerCase().includes(q)) : assets;
    }, [assets, query]);
    const filteredElements = useMemo(() => {
        const q = query.trim().toLowerCase();
        return q ? elements.filter((element) => `${element.name} ${element.trigger_word} ${element.description}`.toLowerCase().includes(q)) : elements;
    }, [elements, query]);

    return (
        <div className="st-page">
            <div className="st-page-head">
                <div>
                    <h2 className="st-page-title">Library</h2>
                    <p className="st-page-subtitle">Project assets and master references. Semantic search and lineage arrive with the library API.</p>
                </div>
                <div className="st-button-row">
                    <button type="button" className="st-button" data-size="sm" onClick={() => void refresh()}>
                        <RefreshCw size={12} className={loading ? 'animate-spin' : ''} /> Refresh
                    </button>
                </div>
            </div>

            <div className="st-button-row" style={{ marginBottom: 12 }}>
                <button type="button" className="st-button" data-size="sm" data-variant={tab === 'assets' ? 'primary' : 'ghost'} onClick={() => setTab('assets')}>
                    <Library size={12} /> Assets ({assets.length})
                </button>
                <button type="button" className="st-button" data-size="sm" data-variant={tab === 'references' ? 'primary' : 'ghost'} onClick={() => setTab('references')}>
                    <ImageIcon size={12} /> Master references ({elements.length})
                </button>
                <input
                    className="st-input"
                    style={{ width: 240, height: 28 }}
                    placeholder="Filter…"
                    value={query}
                    onChange={(event) => setQuery(event.target.value)}
                />
            </div>

            {tab === 'assets' ? (
                filteredAssets.length === 0 ? (
                    <div className="st-empty"><strong>No assets</strong><span>Upload media in the legacy library while F1 lands.</span></div>
                ) : (
                    <div className="st-grid">
                        {filteredAssets.map((asset) => (
                            <article className="st-take" key={asset.id}>
                                <div className="st-take-media">
                                    {asset.type === 'video' ? (
                                        <video src={getAssetUrl(asset.url)} muted playsInline preload="metadata" />
                                    ) : asset.type === 'image' ? (
                                        <img src={getAssetUrl(asset.url)} alt="" loading="lazy" />
                                    ) : (
                                        <div className="st-stage-empty"><Video size={14} /><span>{asset.type}</span></div>
                                    )}
                                </div>
                                <div className="st-take-body">
                                    <strong style={{ color: 'var(--st-text)' }}>{asset.filename}</strong>
                                    <div>{asset.type} · {new Date(asset.created_at * 1000).toLocaleDateString()}</div>
                                </div>
                            </article>
                        ))}
                    </div>
                )
            ) : filteredElements.length === 0 ? (
                <div className="st-empty"><strong>No master references</strong><span>Create Characters, Locations, and Objects in the legacy Elements view.</span></div>
            ) : (
                <div className="st-grid">
                    {filteredElements.map((element) => (
                        <article className="st-take" key={element.id}>
                            <div className="st-take-media">
                                {element.image_path ? <img src={getAssetUrl(element.image_path)} alt="" loading="lazy" /> : <div className="st-stage-empty"><ImageIcon size={14} /><span>{element.type}</span></div>}
                            </div>
                            <div className="st-take-body">
                                <strong style={{ color: 'var(--st-text)' }}>{element.name}</strong>
                                <div>@{element.trigger_word.replace(/^@/, '')} · {element.type}</div>
                                <div style={{ marginTop: 4, color: 'var(--st-faint)' }}>{element.description}</div>
                            </div>
                        </article>
                    ))}
                </div>
            )}
        </div>
    );
}
