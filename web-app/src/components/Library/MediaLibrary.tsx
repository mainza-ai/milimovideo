import { useState, useEffect } from 'react';
import { useTimelineStore } from '../../stores/timelineStore';
import type { ConditioningItem } from '../../stores/timelineStore';
import { Upload, Image as ImageIcon, Film, Plus, Trash2 } from 'lucide-react';
import { clsx } from 'clsx';

interface Asset {
    id: string;
    url: string;
    path: string; // internal path
    type: 'image' | 'video';
    filename: string;
    thumbnail?: string | null;
}

export const MediaLibrary = () => {
    const { selectedShotId, addConditioningToShot } = useTimelineStore();
    const [assets, setAssets] = useState<Asset[]>([]);
    const [isUploading, setIsUploading] = useState(false);
    const [tab, setTab] = useState<'project' | 'history'>('history');

    const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        if (!e.target.files) return;
        setIsUploading(true);

        const formData = new FormData();
        // Handle multiple files? For now just one
        formData.append('file', e.target.files[0]);

        try {
            const res = await fetch('http://localhost:8000/upload', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();

            const newAsset: Asset = {
                id: data.asset_id,
                url: `http://localhost:8000${data.url}`,
                path: data.access_path,
                type: data.type,
                filename: data.filename,
                thumbnail: data.thumbnail ? `http://localhost:8000${data.thumbnail}` : null
            };

            setAssets(prev => [newAsset, ...prev]);
        } catch (err) {
            console.error(err);
        } finally {
            setIsUploading(false);
        }
    };

    const [previewAsset, setPreviewAsset] = useState<Asset | null>(null);

    const handleAssetClick = (asset: Asset) => {
        setPreviewAsset(asset);
    };

    const confirmAddToShot = () => {
        if (!previewAsset || !selectedShotId) return;

        // Add to currently selected shot
        const item: Omit<ConditioningItem, 'id'> = {
            type: previewAsset.type,
            path: previewAsset.url, // Use web URL, not FS path
            frameIndex: 0,
            strength: 0.8 // defaults
        };
        addConditioningToShot(selectedShotId, item);
        setPreviewAsset(null);
    };

    const fetchHistory = async () => {
        try {
            const res = await fetch('http://localhost:8000/uploads');
            const data = await res.json();
            const mapped = data.map((a: any) => ({
                ...a,
                url: a.url.startsWith('http') ? a.url : `http://localhost:8000${a.url}`,
                thumbnail: a.thumbnail && !a.thumbnail.startsWith('http')
                    ? `http://localhost:8000${a.thumbnail}`
                    : a.thumbnail
            }));
            setAssets(mapped);
        } catch (e) { console.error(e); }
    };

    // Refresh history when tab changes or refresh triggered
    const { assetRefreshVersion } = useTimelineStore();
    useEffect(() => {
        fetchHistory();
    }, [tab, assetRefreshVersion]);

    const handleDeleteAsset = async (e: React.MouseEvent, filename: string) => {
        e.stopPropagation();
        if (!confirm("Delete this asset?")) return;
        try {
            await fetch(`http://localhost:8000/upload/${filename}`, { method: 'DELETE' });
            setAssets(prev => prev.filter(a => a.filename !== filename));
        } catch (e) { console.error(e); }
    };

    return (
        <div className="w-72 h-full bg-[#0a0a0a] border-r border-white/5 flex flex-col relative">
            {/* Tabs */}
            {/* ... */}
            <div className="flex border-b border-white/5">
                <button
                    onClick={() => { setTab('project'); }}
                    className={clsx(
                        "flex-1 py-3 text-[10px] uppercase font-bold tracking-widest transition-colors",
                        tab === 'project' ? "text-white bg-white/5" : "text-white/40 hover:text-white"
                    )}
                >
                    Assets
                </button>
                <button
                    onClick={() => { setTab('history'); }}
                    className={clsx(
                        "flex-1 py-3 text-[10px] uppercase font-bold tracking-widest transition-colors",
                        tab === 'history' ? "text-white bg-white/5" : "text-white/40 hover:text-white"
                    )}
                >
                    History
                </button>
            </div>

            {/* Content */}
            <div className="flex-1 overflow-y-auto p-4">
                <div className="space-y-4">
                    {/* Upload Button */}
                    <label className="flex flex-col items-center justify-center h-24 border-2 border-dashed border-white/10 rounded-xl hover:border-white/30 hover:bg-white/5 cursor-pointer transition-all group">
                        <input type="file" className="hidden" onChange={handleUpload} disabled={isUploading} />
                        <Upload className="mb-2 text-white/20 group-hover:text-milimo-400 transition-colors" size={20} />
                        <span className="text-[10px] uppercase font-bold text-white/30">
                            {isUploading ? "Uploading..." : "Upload Media"}
                        </span>
                    </label>

                    {/* Grid */}
                    <div className="grid grid-cols-2 gap-2">
                        {assets.map(asset => (
                            <div
                                key={asset.id}
                                draggable
                                onDragStart={(e) => {
                                    e.dataTransfer.setData('application/json', JSON.stringify(asset));
                                }}
                                onClick={() => handleAssetClick(asset)}
                                className="relative aspect-square bg-white/5 rounded-lg overflow-hidden border border-white/5 hover:border-milimo-500 cursor-pointer group"
                            >
                                {asset.type === 'image' ? (
                                    <img src={asset.url} className="w-full h-full object-cover" />
                                ) : (
                                    asset.thumbnail ? (
                                        <img src={asset.thumbnail} className="w-full h-full object-cover" />
                                    ) : (
                                        <video src={asset.url} className="w-full h-full object-cover" muted loop onMouseOver={e => e.currentTarget.play()} onMouseOut={e => e.currentTarget.pause()} />
                                    )
                                )}

                                {/* Type icon */}
                                <div className="absolute top-1 left-1 bg-black/50 p-1 rounded backdrop-blur-sm">
                                    {asset.type === 'image' ? <ImageIcon size={10} className="text-white/70" /> : <Film size={10} className="text-white/70" />}
                                </div>

                                {/* Delete Button */}
                                <button
                                    onClick={(e) => handleDeleteAsset(e, asset.filename)}
                                    className="absolute top-1 right-1 bg-red-500/80 p-1 rounded opacity-0 group-hover:opacity-100 transition-opacity hover:bg-red-600"
                                >
                                    <Trash2 size={10} className="text-white" />
                                </button>
                            </div>
                        ))}
                    </div>
                    {assets.length === 0 && (
                        <div className="text-center py-8 text-white/20 text-xs italic">
                            No assets found.
                        </div>
                    )}
                </div>
            </div>

            {/* Preview Modal */}
            {previewAsset && (
                <div className="fixed inset-0 z-[1000] flex items-center justify-center p-8 bg-black/80 backdrop-blur-sm" onClick={() => setPreviewAsset(null)}>
                    <div className="bg-[#111] border border-white/10 rounded-2xl overflow-hidden max-w-4xl max-h-full flex flex-col shadow-2xl" onClick={e => e.stopPropagation()}>
                        <div className="flex-1 overflow-hidden bg-black/50 flex items-center justify-center min-h-[300px]">
                            {previewAsset.type === 'image' ? (
                                <img src={previewAsset.url} className="max-w-full max-h-[70vh] object-contain" />
                            ) : (
                                <video src={previewAsset.url} controls autoPlay className="max-w-full max-h-[70vh]" />
                            )}
                        </div>
                        <div className="p-4 bg-[#151515] border-t border-white/5 flex items-center justify-between gap-4">
                            <div className="min-w-0">
                                <h4 className="text-sm font-medium text-white truncate">{previewAsset.filename}</h4>
                                <p className="text-xs text-white/40 uppercase tracking-wider">{previewAsset.type}</p>
                            </div>
                            <div className="flex gap-2">
                                <button
                                    onClick={() => setPreviewAsset(null)}
                                    className="px-4 py-2 rounded-lg text-xs font-bold uppercase tracking-wider text-white/60 hover:text-white hover:bg-white/5 transition-colors"
                                >
                                    Cancel
                                </button>
                                <button
                                    onClick={confirmAddToShot}
                                    disabled={!selectedShotId}
                                    className={clsx(
                                        "px-6 py-2 rounded-lg text-xs font-bold uppercase tracking-wider transition-all flex items-center gap-2",
                                        selectedShotId
                                            ? "bg-milimo-500 hover:bg-milimo-400 text-black"
                                            : "bg-white/5 text-white/20 cursor-not-allowed"
                                    )}
                                    title={!selectedShotId ? "Select a shot first" : ""}
                                >
                                    <Plus size={14} /> Add to Shot
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};
