import type { StateCreator } from 'zustand';
import type { TimelineState } from '../types';
import { API_BASE_URL } from '../../config';

// ── Types ────────────────────────────────────────────────────────

export interface ModelInfo {
    id: string;
    name: string;
    family: string;
    role: string;
    pipeline: string;
    required: boolean;
    size_bytes: number | null;
    relative_path: string;
    absolute_path: string;
    description: string;
    device_compatibility: string[];
    depends_on: string[];
    alternatives: string[];
    huggingface: { repo?: string; filename?: string; subdir?: string };
    status: 'not_downloaded' | 'downloading' | 'downloaded' | 'active' | 'error' | 'incompatible';
    downloaded_bytes: number;
    download_progress: number;
    error_message: string | null;
    // v2 fields
    type: string;                    // base | quantized | ic_lora | camera_lora | adapter
    pipeline_tag: string;            // HF pipeline tag
    requires_base: boolean;
    base_repo_id: string | null;
    vram_estimate_gb: number;
    recommended_dtype: string;
    gated: boolean;
    license: string;
}

export interface DownloadProgress {
    model_id: string;
    progress: number;
    speed_mbps: number;
    eta_seconds: number;
    downloaded_bytes: number;
    total_bytes: number;
}

export interface PipelineReadiness {
    [pipeline: string]: {
        ready: boolean;
        missing: string[];
        total: number;
        downloaded: number;
    };
}

export interface ModelSlice {
    models: ModelInfo[];
    modelsLoading: boolean;
    downloadProgress: Record<string, DownloadProgress>;
    pipelineReadiness: PipelineReadiness;
    fetchModels: () => Promise<void>;
    fetchPipelineReadiness: () => Promise<void>;
    startDownload: (modelId: string) => Promise<void>;
    cancelDownload: (modelId: string) => Promise<void>;
    deleteModel: (modelId: string) => Promise<void>;
    activateModel: (modelId: string) => Promise<void>;
    deactivateModel: (modelId: string) => Promise<void>;
    scanDisk: () => Promise<void>;
    handleModelEvent: (type: string, data: any) => void;
}

// ── Slice ─────────────────────────────────────────────────────────

export const createModelSlice: StateCreator<TimelineState, [], [], ModelSlice> = (set, get) => ({
    models: [],
    modelsLoading: false,
    downloadProgress: {},
    pipelineReadiness: {},

    fetchModels: async () => {
        set({ modelsLoading: true });
        try {
            const res = await fetch(`${API_BASE_URL}/models`);
            if (!res.ok) throw new Error(`Failed to fetch models: ${res.status}`);
            const data = await res.json();
            set({ models: data.models || [], modelsLoading: false });
        } catch (err) {
            console.error('[ModelSlice] fetchModels failed:', err);
            set({ modelsLoading: false });
        }
    },

    fetchPipelineReadiness: async () => {
        try {
            const res = await fetch(`${API_BASE_URL}/models/pipelines`);
            if (!res.ok) return;
            const data = await res.json();
            set({ pipelineReadiness: data });
        } catch (err) {
            console.error('[ModelSlice] fetchPipelineReadiness failed:', err);
        }
    },

    startDownload: async (modelId: string) => {
        try {
            const res = await fetch(`${API_BASE_URL}/models/${modelId}/download`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ force: false }),
            });
            if (!res.ok) {
                const err = await res.json();
                get().addToast(err.detail || 'Download failed', 'error');
                return;
            }
            // Update model status locally
            const models = get().models.map(m =>
                m.id === modelId ? { ...m, status: 'downloading' as const } : m
            );
            set({ models });
            get().addToast('Download started', 'info');
        } catch (err) {
            console.error('[ModelSlice] startDownload failed:', err);
            get().addToast('Failed to start download', 'error');
        }
    },

    cancelDownload: async (modelId: string) => {
        try {
            await fetch(`${API_BASE_URL}/models/${modelId}/download`, { method: 'DELETE' });
            const models = get().models.map(m =>
                m.id === modelId ? { ...m, status: 'not_downloaded' as const, download_progress: 0 } : m
            );
            const progress = { ...get().downloadProgress };
            delete progress[modelId];
            set({ models, downloadProgress: progress });
            get().addToast('Download cancelled', 'info');
        } catch (err) {
            console.error('[ModelSlice] cancelDownload failed:', err);
        }
    },

    deleteModel: async (modelId: string) => {
        try {
            const res = await fetch(`${API_BASE_URL}/models/${modelId}`, { method: 'DELETE' });
            if (!res.ok) {
                const err = await res.json();
                get().addToast(err.detail || 'Delete failed', 'error');
                return;
            }
            const models = get().models.map(m =>
                m.id === modelId ? { ...m, status: 'not_downloaded' as const, downloaded_bytes: 0 } : m
            );
            set({ models });
            get().addToast('Model deleted', 'success');
        } catch (err) {
            console.error('[ModelSlice] deleteModel failed:', err);
        }
    },

    activateModel: async (modelId: string) => {
        try {
            const res = await fetch(`${API_BASE_URL}/models/${modelId}/activate`, {
                method: 'POST',
            });
            if (!res.ok) {
                const err = await res.json();
                get().addToast(err.detail || 'Activation failed', 'error');
                return;
            }
            const models = get().models.map(m =>
                m.id === modelId ? { ...m, status: 'active' as const } : m
            );
            set({ models });
            get().addToast('Model activated', 'success');
            get().fetchPipelineReadiness();
        } catch (err) {
            console.error('[ModelSlice] activateModel failed:', err);
            get().addToast('Failed to activate model', 'error');
        }
    },

    deactivateModel: async (modelId: string) => {
        try {
            const res = await fetch(`${API_BASE_URL}/models/${modelId}/deactivate`, {
                method: 'POST',
            });
            if (!res.ok) {
                const err = await res.json();
                get().addToast(err.detail || 'Deactivation failed', 'error');
                return;
            }
            const models = get().models.map(m =>
                m.id === modelId ? { ...m, status: 'downloaded' as const } : m
            );
            set({ models });
            get().addToast('Model deactivated', 'info');
            get().fetchPipelineReadiness();
        } catch (err) {
            console.error('[ModelSlice] deactivateModel failed:', err);
            get().addToast('Failed to deactivate model', 'error');
        }
    },

    scanDisk: async () => {
        try {
            const res = await fetch(`${API_BASE_URL}/models/scan`, { method: 'POST' });
            if (!res.ok) return;
            const data = await res.json();
            set({ models: data.models || [] });
            get().addToast('Disk scan complete', 'info');
        } catch (err) {
            console.error('[ModelSlice] scanDisk failed:', err);
        }
    },

    handleModelEvent: (type: string, data: any) => {
        if (type === 'download_progress') {
            const progress = { ...get().downloadProgress };
            progress[data.model_id] = data;
            // Also update model's download_progress
            const models = get().models.map(m =>
                m.id === data.model_id
                    ? { ...m, status: 'downloading' as const, download_progress: data.progress }
                    : m
            );
            set({ downloadProgress: progress, models });

        } else if (type === 'download_complete') {
            const progress = { ...get().downloadProgress };
            delete progress[data.model_id];
            const models = get().models.map(m =>
                m.id === data.model_id
                    ? { ...m, status: 'downloaded' as const, download_progress: 1.0 }
                    : m
            );
            set({ downloadProgress: progress, models });
            get().addToast('Model download complete', 'success');
            // Refresh pipeline readiness
            get().fetchPipelineReadiness();

        } else if (type === 'download_error') {
            const progress = { ...get().downloadProgress };
            delete progress[data.model_id];
            const models = get().models.map(m =>
                m.id === data.model_id
                    ? { ...m, status: 'error' as const, error_message: data.error }
                    : m
            );
            set({ downloadProgress: progress, models });
            get().addToast(`Download failed: ${data.error}`, 'error');

        } else if (type === 'model_status_change') {
            const models = get().models.map(m =>
                m.id === data.model_id ? { ...m, status: data.new_status } : m
            );
            set({ models });

        } else if (type === 'lora_updated') {
            // LoRA stack changed — refresh pipeline readiness
            get().fetchPipelineReadiness();
            get().fetchModels();

        } else if (type === 'download_queued') {
            get().addToast(`Download queued (#${(data.position ?? 0) + 1})`, 'info');

        } else if (type === 'queue_updated') {
            // Queue changed — refresh models to update statuses
            get().fetchModels();
        }
    },
});
