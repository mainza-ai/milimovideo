import { useMemo, useRef, useState } from 'react';
import {
    CheckCircle2, FileText, Film, Image as ImageIcon, Library, Pencil, Plus, RefreshCw, Search,
    Trash2, Upload, Video, X,
} from 'lucide-react';
import { rankSearch, type SearchRecord } from '../search/localSearch';
import { useStudioEditor } from '../state/editorStore';
import { useMediaPreview } from '../state/mediaPreviewStore';
import { MediaThumb } from '../components/MediaThumb';
import { useStudioDocumentStore, type Shot } from '../state/documentStore';
import type { StudioAsset, StudioElement, StudioProjectTake } from '../data/useStudioData';

type SearchItem =
    | { type: 'shot'; id: string; shot: Shot }
    | { type: 'asset'; id: string; asset: StudioAsset }
    | { type: 'reference'; id: string; element: StudioElement }
    | { type: 'take'; id: string; take: StudioProjectTake };
import { apiUrl, getAssetUrl } from '../../config';
import {
    useStudioAssets,
    useStudioElementMutations,
    useStudioElements,
    useStudioProject,
    useStudioProjectTakes,
    useStudioSelectTake,
    useStudioUploadAsset,
    type ElementDraft,
} from '../data/useStudioData';

function assetKind(asset: StudioAsset): 'image' | 'video' | 'audio' | 'text' {
    const name = `${asset.filename || ''} ${asset.url || ''}`.toLowerCase();
    if (/\.(png|jpe?g|webp|gif|bmp|tiff)(\?|\s|$)/.test(name)) return 'image';
    if (/\.(mp4|mov|m4v|webm|mkv|avi)(\?|\s|$)/.test(name)) return 'video';
    if (/\.(mp3|wav|m4a|aac|ogg|flac)(\?|\s|$)/.test(name)) return 'audio';
    if (/\.(srt|vtt|json|txt|md)(\?|\s|$)/.test(name)) return 'text';
    return asset.type === 'video' || asset.type === 'image' || asset.type === 'audio' ? asset.type : 'text';
}

const EMPTY_DRAFT: ElementDraft = {
    name: '',
    type: 'character',
    description: '',
    trigger_word: '',
    image_path: null,
};

export function LibrarySurface() {
    const project = useStudioProject();
    const openMediaPreview = useMediaPreview((state) => state.open);
    const { elements, refresh: refreshElements } = useStudioElements(project.id);
    const elementMutations = useStudioElementMutations(project.id);
    const [editingElement, setEditingElement] = useState<StudioElement | null>(null);
    const [draft, setDraft] = useState<ElementDraft | null>(null);
    const [savingElement, setSavingElement] = useState(false);
    const [elementImageBusy, setElementImageBusy] = useState(false);
    const { assets, loading, refresh } = useStudioAssets();
    const { takes, loading: takesLoading, refresh: refreshTakes } = useStudioProjectTakes(project.id);
    const selectTake = useStudioSelectTake();
    const uploadAsset = useStudioUploadAsset(project.id);
    const uploadRef = useRef<HTMLInputElement>(null);
    const [uploading, setUploading] = useState(false);
    const [tab, setTab] = useState<'assets' | 'references' | 'takes' | 'search'>('assets');
    const [query, setQuery] = useState('');

    const openElementEditor = (element?: StudioElement) => {
        setEditingElement(element ?? null);
        setDraft(element ? {
            name: element.name,
            type: element.type,
            description: element.description,
            trigger_word: element.trigger_word,
            image_path: element.image_path ?? null,
        } : { ...EMPTY_DRAFT });
    };

    const saveElement = async () => {
        if (!draft || !draft.name.trim()) return;
        setSavingElement(true);
        try {
            const saved = editingElement
                ? await elementMutations.update(editingElement.id, draft)
                : await elementMutations.create(draft);
            if (saved) {
                await refreshElements();
                setDraft(null);
                setEditingElement(null);
            }
        } finally {
            setSavingElement(false);
        }
    };

    const deleteElement = async (element: StudioElement) => {
        if (!window.confirm(`Delete the reference "${element.name}"?`)) return;
        if (await elementMutations.remove(element)) await refreshElements();
    };

    const deleteAsset = async (asset: StudioAsset) => {
        if (!window.confirm(`Delete "${asset.filename}"? This removes the file and its poster.`)) return;
        try {
            const response = await fetch(apiUrl(`/assets/${asset.id}`), { method: 'DELETE' });
            const data = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
            const store = useStudioDocumentStore.getState();
            for (const shot of store.project.shots) {
                if (shot.videoUrl !== asset.url && shot.thumbnailUrl !== asset.url) continue;
                store.updateShot(shot.id, {
                    videoUrl: shot.videoUrl === asset.url ? null : shot.videoUrl,
                    thumbnailUrl: shot.thumbnailUrl === asset.url ? undefined : shot.thumbnailUrl,
                });
            }
            store.addToast(`Deleted ${asset.filename}`, 'success');
            await refresh();
            await refreshTakes();
        } catch (error) {
            useStudioDocumentStore.getState().addToast(
                error instanceof Error ? error.message : 'Delete failed',
                'error',
            );
        }
    };

    const uploadElementImage = async (file: File) => {
        setElementImageBusy(true);
        try {
            const form = new FormData();
            form.append('file', file);
            form.append('project_id', project.id);
            form.append('type', 'element');
            const response = await fetch(apiUrl('/upload'), { method: 'POST', body: form });
            const data = await response.json().catch(() => ({}));
            if (!response.ok) {
                useStudioDocumentStore.getState().addToast(data.detail || 'Upload failed', 'error');
                return;
            }
            setDraft((current) => current ? { ...current, image_path: data.url } : current);
        } finally {
            setElementImageBusy(false);
        }
    };

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
                                <button
                                    type="button"
                                    className="st-take-media st-media-open"
                                    aria-label={`Preview ${asset.filename}`}
                                    onClick={() => openMediaPreview({
                                        url: getAssetUrl(asset.url) || asset.url,
                                        title: asset.filename,
                                        subtitle: `${assetKind(asset)} · ${new Date(asset.created_at * 1000).toLocaleDateString()}`,
                                        kind: assetKind(asset),
                                    })}
                                >
                                    {assetKind(asset) === 'video' || assetKind(asset) === 'image' ? (
                                        <MediaThumb
                                            src={asset.url}
                                            poster={asset.thumbnail}
                                            alt={asset.filename}
                                        />
                                    ) : assetKind(asset) === 'audio' ? (
                                        <div className="st-stage-empty"><Video size={14} /><span>audio</span></div>
                                    ) : (
                                        <div className="st-stage-empty"><FileText size={14} /><span>{assetKind(asset)}</span></div>
                                    )}
                                </button>
                                <div className="st-take-body">
                                    <strong style={{ color: 'var(--st-text)' }}>{asset.filename}</strong>
                                    <div>{asset.type} · {new Date(asset.created_at * 1000).toLocaleDateString()}</div>
                                    <button
                                        type="button"
                                        className="st-button"
                                        data-size="sm"
                                        data-variant="danger"
                                        onClick={() => void deleteAsset(asset)}
                                        title="Delete this asset and its file"
                                    >
                                        <Trash2 size={11} /> Delete
                                    </button>
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
                                        if (item.type === 'asset') {
                                            openMediaPreview({
                                                url: getAssetUrl(item.asset.url) || item.asset.url,
                                                title: item.asset.filename,
                                                subtitle: item.asset.type,
                                                kind: assetKind(item.asset),
                                            });
                                            return;
                                        }
                                        if (item.type === 'reference' && item.element.image_path) {
                                            openMediaPreview({
                                                url: getAssetUrl(item.element.image_path) || item.element.image_path,
                                                title: item.element.name,
                                                subtitle: `${item.element.type} · @${item.element.trigger_word.replace(/^@/, '')}`,
                                            });
                                            return;
                                        }
                                        if (item.type === 'take') {
                                            openMediaPreview({
                                                url: getAssetUrl(item.take.url || item.take.thumbnail_url) || item.take.url || item.take.thumbnail_url || '',
                                                title: item.take.shot_label || item.take.job_id.slice(0, 12),
                                                subtitle: `${item.take.model || item.take.type} · seed ${item.take.seed ?? '—'}`,
                                            });
                                            return;
                                        }
                                        const shotId = item.type === 'shot' ? item.shot.id : null;
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
                                    <button
                                        type="button"
                                        className="st-take-media st-media-open"
                                        aria-label={`Preview take ${take.job_id.slice(0, 12)}`}
                                        onClick={() => openMediaPreview({
                                            url: getAssetUrl(take.url || take.thumbnail_url) || take.url || take.thumbnail_url || '',
                                            title: take.shot_label || take.job_id.slice(0, 12),
                                            subtitle: `${take.model || take.type} · seed ${take.seed ?? '—'}`,
                                        })}
                                    >
                                        <MediaThumb
                                            src={take.url}
                                            poster={take.thumbnail_url}
                                            alt={take.shot_label || take.job_id}
                                        />
                                    </button>
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
            ) : (
                <>
                    <div className="st-button-row" style={{ marginBottom: 12 }}>
                        <button
                            type="button"
                            className="st-button"
                            data-size="sm"
                            data-variant="primary"
                            onClick={() => openElementEditor()}
                        >
                            <Plus size={12} /> New reference
                        </button>
                        <span className="st-value" style={{ color: 'var(--st-muted)' }}>
                            Master references are project-wide; @mention them in prompts.
                        </span>
                    </div>

                    {draft && (
                        <div className="st-panel" style={{ marginBottom: 12 }}>
                            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8 }}>
                                <h3 className="st-panel-title" style={{ margin: 0 }}>
                                    {editingElement ? `Edit ${editingElement.name}` : 'New master reference'}
                                </h3>
                                <button
                                    type="button"
                                    className="st-icon-button"
                                    aria-label="Close reference editor"
                                    onClick={() => { setDraft(null); setEditingElement(null); }}
                                >
                                    <X size={12} />
                                </button>
                            </div>
                            <div className="st-form-grid" style={{ marginTop: 10 }}>
                                <label className="st-field">
                                    <span>Name</span>
                                    <input
                                        className="st-input"
                                        value={draft.name}
                                        onChange={(event) => setDraft({ ...draft, name: event.target.value })}
                                        placeholder="Mara"
                                    />
                                </label>
                                <label className="st-field">
                                    <span>Type</span>
                                    <select
                                        className="st-select"
                                        value={draft.type}
                                        onChange={(event) => setDraft({ ...draft, type: event.target.value })}
                                    >
                                        <option value="character">Character</option>
                                        <option value="location">Location</option>
                                        <option value="object">Object</option>
                                        <option value="style">Style</option>
                                    </select>
                                </label>
                                <label className="st-field">
                                    <span>Trigger word · optional</span>
                                    <input
                                        className="st-input"
                                        value={draft.trigger_word}
                                        onChange={(event) => setDraft({ ...draft, trigger_word: event.target.value })}
                                        placeholder="auto from name"
                                    />
                                </label>
                                <label className="st-field">
                                    <span>Reference image</span>
                                    <input
                                        type="file"
                                        className="st-input"
                                        accept="image/*"
                                        aria-label="Reference image"
                                        disabled={elementImageBusy}
                                        onChange={(event) => {
                                            const file = event.target.files?.[0];
                                            event.target.value = '';
                                            if (file) void uploadElementImage(file);
                                        }}
                                    />
                                </label>
                            </div>
                            <label className="st-field" style={{ marginTop: 8 }}>
                                <span>Description</span>
                                <textarea
                                    className="st-textarea"
                                    style={{ minHeight: 64 }}
                                    aria-label="Reference description"
                                    value={draft.description}
                                    onChange={(event) => setDraft({ ...draft, description: event.target.value })}
                                    placeholder="Wardrobe, silhouette, and continuity notes…"
                                />
                            </label>
                            {draft.image_path && (
                                <img
                                    src={getAssetUrl(draft.image_path)}
                                    alt="Reference preview"
                                    style={{ marginTop: 8, maxHeight: 120, borderRadius: 6, objectFit: 'cover' }}
                                />
                            )}
                            <div className="st-button-row" style={{ marginTop: 10 }}>
                                <button
                                    type="button"
                                    className="st-button"
                                    data-variant="primary"
                                    disabled={savingElement || elementImageBusy || !draft.name.trim()}
                                    onClick={() => void saveElement()}
                                >
                                    {savingElement ? 'Saving…' : editingElement ? 'Save reference' : 'Create reference'}
                                </button>
                                <button
                                    type="button"
                                    className="st-button"
                                    onClick={() => { setDraft(null); setEditingElement(null); }}
                                >
                                    Cancel
                                </button>
                                {elementImageBusy && (
                                    <span className="st-value" style={{ color: 'var(--st-muted)' }}>Uploading image…</span>
                                )}
                            </div>
                        </div>
                    )}

                    {filteredElements.length === 0 ? (
                        <div className="st-empty">
                            <strong>No master references</strong>
                            <span>Create a character, location, or object to reuse it across shots.</span>
                        </div>
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
                                        <div className="st-button-row" style={{ marginTop: 8 }}>
                                            <button
                                                type="button"
                                                className="st-button"
                                                data-size="sm"
                                                aria-label={`Edit ${element.name}`}
                                                onClick={() => openElementEditor(element)}
                                            >
                                                <Pencil size={11} /> Edit
                                            </button>
                                            <button
                                                type="button"
                                                className="st-button"
                                                data-size="sm"
                                                data-variant="ghost"
                                                aria-label={`Delete ${element.name}`}
                                                onClick={() => void deleteElement(element)}
                                            >
                                                <Trash2 size={11} /> Delete
                                            </button>
                                        </div>
                                    </div>
                                </article>
                            ))}
                        </div>
                    )}
                </>
            )}
        </div>
    );
}
