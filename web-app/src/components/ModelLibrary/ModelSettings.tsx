import React, { useState, useEffect, useCallback } from 'react';
import { useShallow } from 'zustand/react/shallow';
import { useTimelineStore } from '../../stores/timelineStore';
import {
    Settings, HardDrive, Zap, RefreshCw, AlertTriangle,
    ArrowUpCircle, Database, Gauge
} from 'lucide-react';
import './ModelSettings.css';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface DiskInfo {
    total_gb: number;
    used_gb: number;
    free_gb: number;
    usage_pct: number;
}

interface ModelDiskBreakdown {
    path: string;
    size_bytes: number;
    size_gb: number;
}

interface QueueItem {
    model_id: string;
    name: string;
    priority: number;
    size_bytes: number;
    position: number;
}

interface VersionInfo {
    model_id: string;
    update_available: boolean;
    remote_date?: string;
    error?: string;
}

export const ModelSettings: React.FC = () => {
    const { addToast } = useTimelineStore(useShallow(state => ({
        addToast: state.addToast,
    })));

    const [disk, setDisk] = useState<DiskInfo | null>(null);
    const [modelBreakdown, setModelBreakdown] = useState<Record<string, ModelDiskBreakdown>>({});
    const [hfToken, setHfTokenConfigured] = useState(false);
    const [hfTransfer, setHfTransfer] = useState(false);
    const [queue, setQueue] = useState<QueueItem[]>([]);
    const [versions, setVersions] = useState<VersionInfo[]>([]);
    const [checkingVersions, setCheckingVersions] = useState(false);
    const [isLoading, setIsLoading] = useState(true);

    // Fetch all settings on mount
    const fetchSettings = useCallback(async () => {
        setIsLoading(true);
        try {
            const [settingsRes, queueRes] = await Promise.all([
                fetch(`${API_BASE_URL}/settings`),
                fetch(`${API_BASE_URL}/models/queue`),
            ]);

            if (settingsRes.ok) {
                const data = await settingsRes.json();
                setHfTokenConfigured(data.hf_token_configured || false);
                setHfTransfer(data.hf_transfer_enabled || false);
                if (data.disk_usage?.disk) {
                    setDisk(data.disk_usage.disk);
                }
                if (data.disk_usage?.models?.breakdown) {
                    setModelBreakdown(data.disk_usage.models.breakdown);
                }
            }

            if (queueRes.ok) {
                const data = await queueRes.json();
                setQueue(data.queue || []);
            }
        } catch (err) {
            console.error('[ModelSettings] fetch failed:', err);
        } finally {
            setIsLoading(false);
        }
    }, []);

    useEffect(() => { fetchSettings(); }, [fetchSettings]);

    // Toggle HF Transfer
    const toggleHfTransfer = async () => {
        try {
            const res = await fetch(`${API_BASE_URL}/settings/hf-transfer`, {
                method: 'PATCH',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ enabled: !hfTransfer }),
            });
            if (res.ok) {
                const data = await res.json();
                setHfTransfer(data.enabled);
                addToast(data.message, data.enabled ? 'success' : 'info');
            }
        } catch (err) {
            addToast('Failed to toggle HF Transfer', 'error');
        }
    };

    // Check for model updates
    const checkVersions = async () => {
        setCheckingVersions(true);
        try {
            const res = await fetch(`${API_BASE_URL}/models/versions`);
            if (res.ok) {
                const data = await res.json();
                setVersions(data.models || []);
                const updates = data.updates_available || 0;
                addToast(
                    updates > 0
                        ? `${updates} model update${updates > 1 ? 's' : ''} available`
                        : 'All models are up to date',
                    updates > 0 ? 'info' : 'success'
                );
            }
        } catch (err) {
            addToast('Version check failed', 'error');
        } finally {
            setCheckingVersions(false);
        }
    };

    // Remove from queue
    const removeFromQueue = async (modelId: string) => {
        try {
            const res = await fetch(`${API_BASE_URL}/models/${modelId}/enqueue`, {
                method: 'DELETE',
            });
            if (res.ok) {
                setQueue(prev => prev.filter(q => q.model_id !== modelId));
                addToast('Removed from queue', 'info');
            }
        } catch (err) {
            addToast('Failed to remove from queue', 'error');
        }
    };

    const updatesAvailable = versions.filter(v => v.update_available);

    if (isLoading) {
        return (
            <div className="model-settings model-settings--loading">
                <RefreshCw size={14} className="spin" /> Loading settings…
            </div>
        );
    }

    return (
        <div className="model-settings">
            {/* Header */}
            <div className="model-settings__header">
                <div className="model-settings__title">
                    <Settings size={14} />
                    Settings & Storage
                </div>
            </div>

            {/* Disk Space */}
            {disk && (
                <div className="settings-section">
                    <div className="settings-section__title">
                        <HardDrive size={11} /> Disk Usage
                    </div>
                    <div className="disk-bar">
                        <div
                            className={`disk-bar__fill ${disk.usage_pct > 90 ? 'disk-bar__fill--critical' : disk.usage_pct > 75 ? 'disk-bar__fill--warning' : ''}`}
                            style={{ width: `${Math.min(disk.usage_pct, 100)}%` }}
                        />
                    </div>
                    <div className="disk-info">
                        <span>{disk.free_gb} GB free</span>
                        <span>{disk.usage_pct}% used</span>
                        <span>{disk.total_gb} GB total</span>
                    </div>

                    {/* Model breakdown */}
                    {Object.keys(modelBreakdown).length > 0 && (
                        <div className="disk-breakdown">
                            {Object.entries(modelBreakdown).map(([name, info]) => (
                                <div key={name} className="disk-breakdown__item">
                                    <Database size={9} />
                                    <span className="disk-breakdown__name">{name}</span>
                                    <span className="disk-breakdown__size">
                                        {info.size_gb.toFixed(1)} GB
                                    </span>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            )}

            {/* Download Queue */}
            {queue.length > 0 && (
                <div className="settings-section">
                    <div className="settings-section__title">
                        <Gauge size={11} /> Download Queue ({queue.length})
                    </div>
                    <div className="queue-list">
                        {queue.map((item) => (
                            <div key={item.model_id} className="queue-item">
                                <span className="queue-item__pos">#{item.position + 1}</span>
                                <span className="queue-item__name">{item.name}</span>
                                <span className={`queue-item__priority queue-item__priority--${item.priority}`}>
                                    P{item.priority}
                                </span>
                                <button
                                    className="queue-item__remove"
                                    onClick={() => removeFromQueue(item.model_id)}
                                    title="Remove from queue"
                                >
                                    ×
                                </button>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Toggles */}
            <div className="settings-section">
                <div className="settings-section__title">
                    <Zap size={11} /> Performance
                </div>

                <div className="settings-toggle">
                    <div className="settings-toggle__info">
                        <span className="settings-toggle__label">HF Transfer</span>
                        <span className="settings-toggle__desc">Rust-accelerated downloads (5-10x)</span>
                    </div>
                    <button
                        className={`toggle-switch ${hfTransfer ? 'toggle-switch--on' : ''}`}
                        onClick={toggleHfTransfer}
                    >
                        <div className="toggle-switch__knob" />
                    </button>
                </div>

                <div className="settings-toggle">
                    <div className="settings-toggle__info">
                        <span className="settings-toggle__label">HF Token</span>
                        <span className="settings-toggle__desc">
                            {hfToken ? '✓ Configured' : '✗ Not set'}
                        </span>
                    </div>
                    <span className={`settings-badge ${hfToken ? 'settings-badge--ok' : 'settings-badge--warn'}`}>
                        {hfToken ? 'Active' : 'Needed'}
                    </span>
                </div>
            </div>

            {/* Version Check */}
            <div className="settings-section">
                <div className="settings-section__title">
                    <ArrowUpCircle size={11} /> Updates
                </div>
                <button
                    className="settings-action-btn"
                    onClick={checkVersions}
                    disabled={checkingVersions}
                >
                    {checkingVersions ? (
                        <><RefreshCw size={10} className="spin" /> Checking…</>
                    ) : (
                        <><RefreshCw size={10} /> Check for Updates</>
                    )}
                </button>

                {updatesAvailable.length > 0 && (
                    <div className="version-updates">
                        {updatesAvailable.map(v => (
                            <div key={v.model_id} className="version-item">
                                <AlertTriangle size={9} className="version-item__icon" />
                                <span>{v.model_id}</span>
                                {v.remote_date && (
                                    <span className="version-item__date">
                                        {new Date(v.remote_date).toLocaleDateString()}
                                    </span>
                                )}
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
};
