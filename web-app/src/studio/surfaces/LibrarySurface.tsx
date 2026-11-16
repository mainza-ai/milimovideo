import { useMemo, useState } from 'react';
import { CheckCircle2, Film, Image as ImageIcon, Library, RefreshCw, Video } from 'lucide-react';
import { getAssetUrl } from '../../config';
import {
    useStudioAssets,
    useStudioElements,
    useStudioProject,
    useStudioProjectTakes,
    useStudioSelectTake,
} from '../data/useStudioData';

export function LibrarySurface() {
    const project = useStudioProject();
    const elements = useStudioElements(project.id);
    const { assets, loading, refresh } = useStudioAssets();
    const { takes, loading: takesLoading, refresh: refreshTakes } = useStudioProjectTakes(project.id);
    const selectTake = useStudioSelectTake();
    const [tab, setTab] = useState<'assets' | 'references' | 'takes'>('assets');
    const [query, setQuery] = useState('');

    const filteredAssets = useMemo(() => {
        const q = query.trim().toLowerCase();
        return q ? assets.filter((asset) => asset.filename.toLowerCase().includes(q)) : assets;
    }, [assets, query]);
    const filteredElements = useMemo(() => {
        const q = query.trim().toLowerCase();
        return q ? elements.filter((element) => `${element.name} ${element.trigger_word} ${element.description}`.toLowerCase().includes(q)) : elements;
    }, [elements, query]);
    const filteredTakes = useMemo(() => {
        const q = query.trim().toLowerCase();
        return q
            ? takes.filter((take) =>
                `${take.shot_label || ''} ${take.prompt || ''} ${take.model || ''} ${take.job_id}`
                    .toLowerCase()
                    .includes(q),
            )
            : takes;
    }, [query, takes]);

    return (
        <div className="st-page">
            <div className="st-page-head">
                <div>
                    <h2 className="st-page-title">Library</h2>
                    <p className="st-page-subtitle">Project assets, master references, and every completed take with its lineage. Filtering is local over the real project data.</p>
                </div>
                <div className="st-button-row">
                    <button
                        type="button"
                        className="st-button"
                        data-size="sm"
                        onClick={() => void Promise.all([refresh(), refreshTakes()])}
                    >
                        <RefreshCw size={12} className={loading || takesLoading ? 'animate-spin' : ''} /> Refresh
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
                <button type="button" className="st-button" data-size="sm" data-variant={tab === 'takes' ? 'primary' : 'ghost'} onClick={() => setTab('takes')}>
                    <Film size={12} /> Takes ({takes.length})
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
            ) : tab === 'takes' ? (
                filteredTakes.length === 0 ? (
                    <div className="st-empty">
                        <strong>No completed takes</strong>
                        <span>Generate a shot; completed jobs appear here with seed, model, and shot lineage.</span>
                    </div>
                ) : (
                    <div className="st-grid">
                        {filteredTakes.map((take) => {
                            const shot = project.shots.find((item) => item.id === take.shot_id);
                            return (
                                <article className="st-take" key={take.job_id} style={{ borderColor: take.selected ? 'var(--st-accent)' : undefined }}>
                                    <div className="st-take-media">
                                        {take.thumbnail_url || take.url ? (
                                            <img src={getAssetUrl(take.thumbnail_url || take.url)} alt="" loading="lazy" />
                                        ) : (
                                            <div className="st-stage-empty"><Film size={14} /></div>
                                        )}
                                    </div>
                                    <div className="st-take-body">
                                        <div style={{ display: 'flex', justifyContent: 'space-between', gap: 6 }}>
                                            <strong style={{ color: 'var(--st-text)' }}>
                                                {take.shot_label || take.shot_id || 'Unassigned take'}
                                            </strong>
                                            {take.selected && <CheckCircle2 size={12} style={{ color: 'var(--st-accent)' }} />}
                                        </div>
                                        <div className="st-mono" style={{ marginTop: 3 }}>{take.job_id.slice(0, 12)} · seed {take.seed ?? '—'}</div>
                                        <div>{take.model || take.type} · {take.num_frames ?? '—'}f</div>
                                        <div style={{ marginTop: 4, color: 'var(--st-faint)' }}>
                                            {take.created_at ? new Date(take.created_at).toLocaleString() : ''}
                                        </div>
                                        {shot && !take.selected && (
                                            <button
                                                type="button"
                                                className="st-button"
                                                data-size="sm"
                                                style={{ marginTop: 6 }}
                                                onClick={async () => {
                                                    if (await selectTake(shot, take)) await refreshTakes();
                                                }}
                                            >
                                                Select for shot
                                            </button>
                                        )}
                                    </div>
                                </article>
                            );
                        })}
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
