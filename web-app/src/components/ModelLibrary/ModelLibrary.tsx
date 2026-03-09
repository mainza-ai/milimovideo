import React, { useEffect, useState } from 'react';
import { useTimelineStore } from '../../stores/timelineStore';
import { useShallow } from 'zustand/react/shallow';
import {
    HardDrive, Download, Trash2, RotateCw, CheckCircle2, XCircle,
    AlertTriangle, Loader2, Film, Image, Scissors, RefreshCw, Zap, ZapOff
} from 'lucide-react';
import type { ModelInfo, DownloadProgress } from '../../stores/slices/modelSlice';
import { LoRAManager } from './LoRAManager';
import { ModelSettings } from './ModelSettings';
import './ModelLibrary.css';

// ── Helpers ──────────────────────────────────────────────────────

const formatBytes = (bytes: number | null) => {
    if (!bytes) return '—';
    if (bytes >= 1e12) return `${(bytes / 1e12).toFixed(1)} TB`;
    if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(1)} GB`;
    if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(0)} MB`;
    return `${(bytes / 1e3).toFixed(0)} KB`;
};

const formatEta = (seconds: number) => {
    if (seconds < 60) return `${Math.round(seconds)}s`;
    if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
    return `${(seconds / 3600).toFixed(1)}h`;
};

const PIPELINE_ICONS: Record<string, React.ReactNode> = {
    video: <Film size={12} />,
    image: <Image size={12} />,
    segmentation: <Scissors size={12} />,
};

const PIPELINE_COLORS: Record<string, string> = {
    video: '#a78bfa',
    image: '#60a5fa',
    segmentation: '#34d399',
};

const STATUS_CONFIG: Record<string, { icon: React.ReactNode; label: string; color: string }> = {
    not_downloaded: { icon: <Download size={12} />, label: 'Not Downloaded', color: '#6b7280' },
    downloading: { icon: <Loader2 size={12} className="spin" />, label: 'Downloading...', color: '#f59e0b' },
    downloaded: { icon: <CheckCircle2 size={12} />, label: 'Downloaded', color: '#10b981' },
    active: { icon: <CheckCircle2 size={12} />, label: 'Active (GPU)', color: '#a78bfa' },
    error: { icon: <XCircle size={12} />, label: 'Error', color: '#ef4444' },
    incompatible: { icon: <AlertTriangle size={12} />, label: 'Incompatible', color: '#6b7280' },
};

// ── Model Card ───────────────────────────────────────────────────

interface ModelCardProps {
    model: ModelInfo;
    progress?: DownloadProgress;
    onDownload: (id: string) => void;
    onCancel: (id: string) => void;
    onDelete: (id: string) => void;
    onActivate: (id: string) => void;
    onDeactivate: (id: string) => void;
}

const ModelCard: React.FC<ModelCardProps> = ({ model, progress, onDownload, onCancel, onDelete, onActivate, onDeactivate }) => {
    const statusCfg = STATUS_CONFIG[model.status] || STATUS_CONFIG.error;
    const isDownloading = model.status === 'downloading';
    const pipelineColor = PIPELINE_COLORS[model.pipeline] || '#6b7280';

    return (
        <div className={`model-card model-card--${model.status}`}>
            {/* Header */}
            <div className="model-card__header">
                <div className="model-card__name">{model.name}</div>
                {model.required && <span className="model-card__badge model-card__badge--required">Required</span>}
            </div>

            {/* Meta Row */}
            <div className="model-card__meta">
                <span className="model-card__pipeline" style={{ color: pipelineColor }}>
                    {PIPELINE_ICONS[model.pipeline]} {model.pipeline}
                </span>
                <span className="model-card__role">{model.role}</span>
                <span className="model-card__size">{formatBytes(model.size_bytes)}</span>
                {model.vram_estimate_gb > 0 && (
                    <span className="model-card__vram" title="Estimated VRAM">
                        🧠 {model.vram_estimate_gb} GB
                    </span>
                )}
            </div>

            {/* Type Badge */}
            {model.type && model.type !== 'base' && (
                <span className={`model-card__type-badge model-card__type-badge--${model.type}`}>
                    {model.type.replace(/_/g, ' ')}
                </span>
            )}

            {/* Description */}
            <p className="model-card__desc">{model.description}</p>

            {/* Progress Bar (downloading state) */}
            {isDownloading && progress && (
                <div className="model-card__progress">
                    <div className="model-card__progress-bar">
                        <div
                            className="model-card__progress-fill"
                            style={{ width: `${Math.round(progress.progress * 100)}%` }}
                        />
                    </div>
                    <div className="model-card__progress-info">
                        <span>{Math.round(progress.progress * 100)}%</span>
                        <span>{progress.speed_mbps.toFixed(1)} MB/s</span>
                        <span>{formatEta(progress.eta_seconds)} remaining</span>
                    </div>
                </div>
            )}

            {/* Error Message */}
            {model.error_message && model.status === 'error' && (
                <div className="model-card__error">
                    <XCircle size={11} /> {model.error_message}
                </div>
            )}

            {/* Incompatible Warning */}
            {model.status === 'incompatible' && (
                <div className="model-card__warning">
                    <AlertTriangle size={11} /> {model.error_message || 'Not compatible with your device'}
                </div>
            )}

            {/* Footer: Status + Actions */}
            <div className="model-card__footer">
                <span className="model-card__status" style={{ color: statusCfg.color }}>
                    {statusCfg.icon} {statusCfg.label}
                </span>
                <div className="model-card__actions">
                    {model.status === 'not_downloaded' && (
                        <button className="model-card__btn model-card__btn--download" onClick={() => onDownload(model.id)}>
                            <Download size={12} /> Download
                        </button>
                    )}
                    {model.status === 'downloading' && (
                        <button className="model-card__btn model-card__btn--cancel" onClick={() => onCancel(model.id)}>
                            <XCircle size={12} /> Cancel
                        </button>
                    )}
                    {model.status === 'error' && (
                        <button className="model-card__btn model-card__btn--download" onClick={() => onDownload(model.id)}>
                            <RotateCw size={12} /> Retry
                        </button>
                    )}
                    {model.status === 'downloaded' && (
                        <>
                            <button className="model-card__btn model-card__btn--activate" onClick={() => onActivate(model.id)}>
                                <Zap size={12} /> Activate
                            </button>
                            <button className="model-card__btn model-card__btn--delete" onClick={() => onDelete(model.id)}>
                                <Trash2 size={12} /> Delete
                            </button>
                        </>
                    )}
                    {model.status === 'active' && (
                        <button className="model-card__btn model-card__btn--deactivate" onClick={() => onDeactivate(model.id)}>
                            <ZapOff size={12} /> Deactivate
                        </button>
                    )}
                </div>
            </div>
        </div>
    );
};

// ── Model Library Panel ──────────────────────────────────────────

type FilterPipeline = 'all' | 'video' | 'image' | 'segmentation';

export const ModelLibrary: React.FC = () => {
    const {
        models, modelsLoading, downloadProgress, pipelineReadiness,
        fetchModels, fetchPipelineReadiness, startDownload, cancelDownload,
        deleteModel, activateModel, deactivateModel, scanDisk
    } = useTimelineStore(useShallow(state => ({
        models: state.models,
        modelsLoading: state.modelsLoading,
        downloadProgress: state.downloadProgress,
        pipelineReadiness: state.pipelineReadiness,
        fetchModels: state.fetchModels,
        fetchPipelineReadiness: state.fetchPipelineReadiness,
        startDownload: state.startDownload,
        cancelDownload: state.cancelDownload,
        deleteModel: state.deleteModel,
        activateModel: state.activateModel,
        deactivateModel: state.deactivateModel,
        scanDisk: state.scanDisk,
    })));

    const [filter, setFilter] = useState<FilterPipeline>('all');
    const [confirmDelete, setConfirmDelete] = useState<string | null>(null);

    // Fetch models on mount
    useEffect(() => {
        fetchModels();
        fetchPipelineReadiness();
    }, []);

    // Filter models
    const filteredModels = filter === 'all'
        ? models
        : models.filter(m => m.pipeline === filter);

    // Group by family
    const families: Record<string, ModelInfo[]> = {};
    for (const m of filteredModels) {
        if (!families[m.family]) families[m.family] = [];
        families[m.family].push(m);
    }

    const handleDelete = (modelId: string) => {
        if (confirmDelete === modelId) {
            deleteModel(modelId);
            setConfirmDelete(null);
        } else {
            setConfirmDelete(modelId);
            setTimeout(() => setConfirmDelete(null), 3000);
        }
    };

    const FAMILY_LABELS: Record<string, string> = {
        ltx2: 'LTX-2 Video Generation',
        ltx23: 'LTX-2.3 Video Generation',
        flux2: 'Flux 2 Image Generation',
        sam3: 'SAM 3 Segmentation',
    };

    return (
        <div className="model-library">
            {/* Header */}
            <div className="model-library__header">
                <div className="model-library__title">
                    <HardDrive size={16} />
                    <span>Model Library</span>
                </div>
                <button className="model-library__scan" onClick={() => { scanDisk(); fetchPipelineReadiness(); }}>
                    <RefreshCw size={12} /> Scan Disk
                </button>
            </div>

            {/* Pipeline Readiness Indicators */}
            <div className="model-library__readiness">
                {Object.entries(pipelineReadiness).map(([pipe, info]) => (
                    <div key={pipe} className={`readiness-pill ${info.ready ? 'readiness-pill--ready' : 'readiness-pill--missing'}`}>
                        {PIPELINE_ICONS[pipe]}
                        <span>{pipe}</span>
                        <span className="readiness-pill__count">{info.downloaded}/{info.total}</span>
                        {info.ready
                            ? <CheckCircle2 size={10} />
                            : <AlertTriangle size={10} />
                        }
                    </div>
                ))}
            </div>

            {/* LoRA Stack Manager */}
            <LoRAManager />

            {/* Filter Bar */}
            <div className="model-library__filters">
                {(['all', 'video', 'image', 'segmentation'] as FilterPipeline[]).map(f => (
                    <button
                        key={f}
                        className={`filter-btn ${filter === f ? 'filter-btn--active' : ''}`}
                        onClick={() => setFilter(f)}
                    >
                        {f === 'all' ? 'All' : f.charAt(0).toUpperCase() + f.slice(1)}
                    </button>
                ))}
            </div>

            {/* Model Grid */}
            {modelsLoading ? (
                <div className="model-library__loading">
                    <Loader2 size={20} className="spin" />
                    <span>Loading models...</span>
                </div>
            ) : (
                <div className="model-library__content">
                    {Object.entries(families).map(([family, groupModels]) => (
                        <div key={family} className="model-family">
                            <h3 className="model-family__title">
                                {FAMILY_LABELS[family] || family}
                            </h3>
                            <div className="model-family__grid">
                                {groupModels.map(m => (
                                    <ModelCard
                                        key={m.id}
                                        model={m}
                                        progress={downloadProgress[m.id]}
                                        onDownload={startDownload}
                                        onCancel={cancelDownload}
                                        onDelete={handleDelete}
                                        onActivate={activateModel}
                                        onDeactivate={deactivateModel}
                                    />
                                ))}
                            </div>
                        </div>
                    ))}
                    {filteredModels.length === 0 && (
                        <div className="model-library__empty">No models match the selected filter.</div>
                    )}
                </div>
            )}

            {/* Settings & Storage Dashboard */}
            <ModelSettings />
        </div>
    );
};
