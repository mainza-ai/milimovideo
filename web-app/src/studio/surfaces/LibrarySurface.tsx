import { useMemo, useRef, useState } from 'react';
import {
    CheckCircle2, Film, Image as ImageIcon, Library, RefreshCw, Search, Upload, Video,
} from 'lucide-react';
import { rankSearch, type SearchRecord } from '../search/localSearch';
import { useStudioEditor } from '../state/editorStore';
import { useStudioDocumentStore, type Shot } from '../state/documentStore';
import type { StudioAsset, StudioElement, StudioProjectTake } from '../data/useStudioData';

type SearchItem =
    | { type: 'shot'; id: string; shot: Shot }
    | { type: 'asset'; id: string; asset: StudioAsset }
    | { type: 'reference'; id: string; element: StudioElement }
    | { type: 'take'; id: string; take: StudioProjectTake };
import { getAssetUrl } from '../../config';
import {
    useStudioAssets,
    useStudioElements,
    useStudioProject,
    useStudioProjectTakes,
    useStudioSelectTake,
    useStudioUploadAsset,
} from '../data/useStudioData';

export function LibrarySurface() {
    const project = useStudioProject();
    const elements = useStudioElements(project.id);
    const { assets, loading, refresh } = useStudioAssets();
    const { takes, loading: takesLoading, refresh: refreshTakes } = useStudioProjectTakes(project.id);
    const selectTake = useStudioSelectTake();
    const uploadAsset = useStudioUploadAsset(project.id);
    const uploadRef = useRef<HTMLInputElement>(null);
    const [uploading, setUploading] = useState(false);
    const [tab, setTab] = useState<'assets' | 'references' | 'takes' | 'search'>('assets');
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

    const searchResults = useMemo(() => {
        const records: Array<SearchRecord<SearchItem>> = [
            ...project.shots.map((shot) => ({
                kind: 'shot',
                item: { type: 'shot' as const, id: shot.id, shot },
                text: `${shot.action || ''} ${shot.prompt || ''} ${shot.character || ''} ${shot.dialogue || ''}`,
            })),
            ...assets.map((asset) => ({
                kind: 'asset',
                item: { type: 'asset' as const, id: asset.id, asset },
                text: `${asset.filename} ${asset.type}`,
            })),
            ...elements.map((element) => ({
                kind: 'reference',
                item: { type: 'reference' as const, id: element.id, element },
                text: `${element.name} ${element.trigger_word} ${element.description} ${element.type}`,
            })),
            ...takes.map((take) => ({
                kind: 'take',
                item: { type: 'take' as const, id: take.job_id, take },
                text: `${take.shot_label || ''} ${take.prompt || ''} ${take.model || ''} ${take.job_id}`,
            })),
        ];
        return rankSearch(query, records);
    }, [assets, elements, project.shots, query, takes]);

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
                        data-variant="primary"
                        disabled={uploading}
                        onClick={() => uploadRef.current?.click()}
                    >
                        <Upload size={12} /> {uploading ? 'Uploading…' : 'Upload media'}
                    </button>
                    <input
                        ref={uploadRef}
                        type="file"
                        accept="image/*,video/*,audio/*,.srt,.vtt"
                        style={{ display: 'none' }}
                        aria-label="Upload media"
                        onChange={async (event) => {
                            const file = event.target.files?.[0];
                            event.target.value = '';
                            if (!file) return;
                            setUploading(true);
                            try {
                                const result = await uploadAsset(file);
                                if (result) await refresh();
                            } finally {
                                setUploading(false);
                            }
                        }}
                    />
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
                <button type="button" className="st-button" data-size="sm" data-variant={tab === 'search' ? 'primary' : 'ghost'} onClick={() => setTab('search')}>
                    <Search size={12} /> Search ({searchResults.length})
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
                    <div className="st-empty"><strong>No assets</strong><span>Upload images, video, audio, or caption files.</span></div>
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
            ) : tab === 'search' ? (
                <div className="st-panel">
                    <h3 className="st-panel-title">Ranked local results · {searchResults.length}</h3>
                    {!query.trim() && (
                        <div className="st-value" style={{ color: 'var(--st-muted)' }}>
                            Type a query to search shots, assets, master references, and takes. Everything runs locally.
                        </div>
                    )}
                    {query.trim() && searchResults.length === 0 && (
                        <div className="st-empty" style={{ minHeight: 100 }}>
                            <strong>No matches</strong>
                            <span>Try a shorter or broader phrase.</span>
                        </div>
                    )}
                    <div className="st-link-list">
                        {searchResults.map((result) => {
                            const item = result.item;
                            const label = item.type === 'shot'
                                ? item.shot.action || item.shot.prompt || 'Shot'
                                : item.type === 'asset'
                                    ? item.asset.filename
                                    : item.type === 'reference'
                                        ? item.element.name
                                        : item.take.shot_label || item.take.job_id;
                            return (
                                <button
                                    key={`${result.kind}-${result.item.id}`}
                                    type="button"
                                    className="st-link-row"
                                    style={{ cursor: 'pointer', background: 'transparent', textAlign: 'left', width: '100%' }}
                                    onClick={() => {
                                        const shotId = item.type === 'shot'
                                            ? item.shot.id
                                            : item.type === 'take'
                                                ? item.take.shot_id
                                                : null;
                                        if (shotId) {
                                            useStudioDocumentStore.getState().selectShot(shotId);
                                            useStudioEditor.getState().setSurface('story');
                                        }
                                    }}
                                >
                                    <span className="st-chip">{result.kind}</span>
                                    <span style={{ flex: 1 }}>{label}</span>
                                    <span className="st-mono" style={{ color: 'var(--st-faint)' }}>score {result.score}</span>
                                </button>
                            );
                        })}
                    </div>
                </div>
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
