import React, { useState, useEffect, useRef } from 'react';
import { API_BASE_URL } from '../config';
import { Settings, ChevronDown, Cpu, Check, Loader2, AlertCircle, Eye, ExternalLink } from 'lucide-react';

/**
 * Legacy header popover. The full configuration plane lives in Studio Setup
 * (`/studio` → Setup → AI & LLM); this stays a two-provider quick switch so the
 * frozen NLE route keeps working during the cutover window.
 */
interface ProviderConfig {
    base_url: string;
    default_model: string;
    keep_alive: string;
    is_local: boolean;
    has_key: boolean;
}

interface RoleAssignment {
    role: string;
    provider: string;
    model: string;
}

interface Snapshot {
    schema_version: number;
    revision: number;
    active_provider: string;
    active_model: string;
    air_gapped: boolean;
    temperature: number;
    max_tokens: number;
    providers: Record<string, ProviderConfig>;
    roles: Record<string, RoleAssignment>;
}

interface ModelRow {
    id: string;
    name: string;
    size?: number | null;
    parameter_size?: string | null;
    vision?: boolean;
}

export const LLMSettings: React.FC = () => {
    const [open, setOpen] = useState(false);
    const [config, setConfig] = useState<Snapshot | null>(null);
    const [models, setModels] = useState<ModelRow[]>([]);
    const [loading, setLoading] = useState(false);
    const [saving, setSaving] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const ref = useRef<HTMLDivElement>(null);

    // Close on outside click
    useEffect(() => {
        const handler = (e: MouseEvent) => {
            if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
        };
        document.addEventListener('mousedown', handler);
        return () => document.removeEventListener('mousedown', handler);
    }, []);

    // Fetch config on open
    useEffect(() => {
        if (!open) return;
        setLoading(true);
        setError(null);
        fetch(`${API_BASE_URL}/settings/llm`)
            .then((r) => {
                if (!r.ok) throw new Error(`HTTP ${r.status}`);
                return r.json();
            })
            .then((snapshot: Snapshot) => {
                setConfig(snapshot);
                const provider = snapshot.roles?.prompt_video?.provider || snapshot.active_provider;
                return fetch(`${API_BASE_URL}/settings/llm/models?provider=${encodeURIComponent(provider)}`);
            })
            .then((r) => (r.ok ? r.json() : { models: [] }))
            .then((data) => setModels(Array.isArray(data.models) ? data.models : []))
            .catch(() => setError('Failed to load LLM settings'))
            .finally(() => setLoading(false));
    }, [open]);

    const save = async (next: Snapshot) => {
        if (!config) return;
        setSaving(true);
        setError(null);
        try {
            const response = await fetch(
                `${API_BASE_URL}/settings/llm?expected_revision=${config.revision}`,
                {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        schema_version: next.schema_version,
                        active_provider: next.active_provider,
                        active_model: next.active_model,
                        air_gapped: next.air_gapped,
                        temperature: next.temperature,
                        max_tokens: next.max_tokens,
                        providers: next.providers,
                        roles: next.roles,
                    }),
                },
            );
            const data = await response.json().catch(() => ({}));
            if (response.status === 409) {
                setError(data?.detail?.message || 'Settings changed elsewhere; reopen to reload.');
                return;
            }
            if (!response.ok) {
                setError(typeof data.detail === 'string' ? data.detail : 'Failed to update');
                return;
            }
            setConfig(data as Snapshot);
        } finally {
            setSaving(false);
        }
    };

    const setProvider = (provider: 'gemma' | 'ollama') => {
        if (!config) return;
        const target = config.providers[provider];
        if (!target) return;
        const model = provider === 'gemma' ? 'ltx-gemma3' : (target.default_model || config.active_model);
        const next: Snapshot = {
            ...config,
            active_provider: provider,
            active_model: model,
            roles: {
                ...config.roles,
                prompt_video: { ...config.roles.prompt_video, provider, model },
            },
        };
        setConfig(next);
        void save(next);
    };

    const setOllamaModel = (model: string) => {
        if (!config) return;
        const next: Snapshot = {
            ...config,
            active_provider: config.roles.prompt_video.provider === 'ollama' ? 'ollama' : config.active_provider,
            active_model: config.roles.prompt_video.provider === 'ollama' ? model : config.active_model,
            providers: {
                ...config.providers,
                ollama: { ...config.providers.ollama, default_model: model },
            },
            roles: {
                ...config.roles,
                prompt_video: config.roles.prompt_video.provider === 'ollama'
                    ? { ...config.roles.prompt_video, model }
                    : config.roles.prompt_video,
            },
        };
        setConfig(next);
        void save(next);
    };

    const formatSize = (bytes?: number | null) => {
        if (!bytes) return '';
        const gb = bytes / (1024 ** 3);
        return gb >= 1 ? `${gb.toFixed(1)}GB` : `${(bytes / (1024 ** 2)).toFixed(0)}MB`;
    };

    const roleProvider = config?.roles?.prompt_video?.provider || config?.active_provider || 'gemma';
    const isOllama = roleProvider === 'ollama';
    const keepAlive = config?.providers?.ollama?.keep_alive !== '0';

    return (
        <div ref={ref} className="relative">
            {/* Gear Icon */}
            <button
                onClick={() => setOpen(!open)}
                className={`p-1.5 rounded transition-colors ${open
                    ? 'bg-white/10 text-white'
                    : 'text-white/40 hover:text-white hover:bg-white/10'
                    }`}
                title="AI Settings"
            >
                <Settings size={16} />
            </button>

            {/* Popover */}
            {open && (
                <div className="absolute right-0 top-full mt-2 w-80 bg-[#141414] border border-white/10 rounded-xl shadow-2xl shadow-black/50 overflow-hidden z-[9999]">
                    {/* Header */}
                    <div className="px-4 py-3 border-b border-white/5 flex items-center gap-2">
                        <Cpu size={14} className="text-milimo-400" />
                        <span className="text-xs font-semibold text-white/80">AI Engine Settings</span>
                        {saving && <Loader2 size={12} className="animate-spin text-white/30 ml-auto" />}
                    </div>

                    {loading ? (
                        <div className="flex items-center justify-center py-8">
                            <Loader2 size={20} className="animate-spin text-white/30" />
                        </div>
                    ) : error ? (
                        <div className="flex items-center gap-2 px-4 py-4 text-red-400 text-xs">
                            <AlertCircle size={14} />
                            {error}
                        </div>
                    ) : config ? (
                        <div className="p-4 space-y-4">
                            {/* Provider Toggle */}
                            <div>
                                <label className="text-[10px] uppercase tracking-wider text-white/30 font-semibold block mb-2">
                                    Video prompt rewriting
                                </label>
                                <div className="flex bg-white/5 p-0.5 rounded-lg border border-white/5">
                                    <button
                                        onClick={() => setProvider('gemma')}
                                        disabled={saving}
                                        className={`flex-1 px-3 py-1.5 text-[11px] font-medium rounded-md transition-all flex items-center justify-center gap-1.5 ${!isOllama
                                            ? 'bg-milimo-500/20 text-milimo-400 shadow-sm border border-milimo-500/20'
                                            : 'text-white/40 hover:text-white/60'
                                            }`}
                                    >
                                        {!isOllama && <Check size={10} />}
                                        Gemma (Built-in)
                                    </button>
                                    <button
                                        onClick={() => setProvider('ollama')}
                                        disabled={saving || !config.providers.ollama}
                                        className={`flex-1 px-3 py-1.5 text-[11px] font-medium rounded-md transition-all flex items-center justify-center gap-1.5 ${isOllama
                                            ? 'bg-milimo-500/20 text-milimo-400 shadow-sm border border-milimo-500/20'
                                            : 'text-white/40 hover:text-white/60'
                                            }`}
                                    >
                                        {isOllama && <Check size={10} />}
                                        Ollama (Local)
                                    </button>
                                </div>
                            </div>

                            {/* Ollama Model Selector — only shown when Ollama is active */}
                            {isOllama && (
                                <div className="space-y-3">
                                    <div>
                                        <label className="text-[10px] uppercase tracking-wider text-white/30 font-semibold block mb-1.5">
                                            Model
                                        </label>
                                        {models.length > 0 ? (
                                            <div className="relative">
                                                <select
                                                    value={config.roles.prompt_video.model}
                                                    onChange={(e) => setOllamaModel(e.target.value)}
                                                    disabled={saving}
                                                    className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-xs text-white appearance-none cursor-pointer hover:bg-white/8 focus:outline-none focus:border-milimo-500/50 transition-colors"
                                                >
                                                    {models.map((m) => (
                                                        <option key={m.id} value={m.id} className="bg-[#1a1a1a] text-white">
                                                            {m.vision ? '👁️ ' : ''}{m.id} {m.parameter_size ? `(${m.parameter_size})` : formatSize(m.size) && `(${formatSize(m.size)})`}
                                                        </option>
                                                    ))}
                                                </select>
                                                <ChevronDown size={12} className="absolute right-3 top-1/2 -translate-y-1/2 text-white/30 pointer-events-none" />
                                            </div>
                                        ) : (
                                            <div className="text-[11px] text-amber-400/70 bg-amber-400/5 border border-amber-400/10 rounded-lg px-3 py-2">
                                                No models found — is Ollama running?
                                            </div>
                                        )}
                                        {models.find((m) => m.id === config.roles.prompt_video.model)?.vision && (
                                            <div className="flex items-center gap-1.5 mt-1.5 text-[10px] text-cyan-400/70">
                                                <Eye size={10} />
                                                Vision model — used for image-conditioned video prompts
                                            </div>
                                        )}
                                    </div>

                                    {/* Keep Model Loaded Toggle */}
                                    <div>
                                        <div className="flex items-center justify-between">
                                            <label className="text-[10px] uppercase tracking-wider text-white/30 font-semibold">
                                                Keep Model Loaded
                                            </label>
                                            <button
                                                onClick={() => {
                                                    const next = {
                                                        ...config,
                                                        providers: {
                                                            ...config.providers,
                                                            ollama: {
                                                                ...config.providers.ollama,
                                                                keep_alive: keepAlive ? '0' : '5m',
                                                            },
                                                        },
                                                    };
                                                    setConfig(next);
                                                    void save(next);
                                                }}
                                                disabled={saving}
                                                aria-label="Keep Ollama model loaded"
                                                className={`relative w-8 h-4 rounded-full transition-colors ${keepAlive ? 'bg-milimo-500/50' : 'bg-white/10'}`}
                                            >
                                                <div className={`absolute top-0.5 w-3 h-3 bg-white rounded-full transition-transform ${keepAlive ? 'translate-x-4' : 'translate-x-0.5'}`} />
                                            </button>
                                        </div>
                                        <p className="text-[10px] text-white/20 mt-1">
                                            {keepAlive ? 'Model stays in RAM (faster prompts, more memory used)' : 'Model unloads after use (saves RAM for generation)'}
                                        </p>
                                    </div>
                                </div>
                            )}

                            {/* Current Status */}
                            <div className="pt-2 border-t border-white/5 flex items-center justify-between gap-2">
                                <div className="flex items-center gap-2 text-[10px] text-white/25">
                                    <div className={`w-1.5 h-1.5 rounded-full ${isOllama ? 'bg-emerald-400' : 'bg-amber-400'}`} />
                                    {isOllama
                                        ? `Ollama → ${config.roles.prompt_video.model}${!keepAlive ? ' (auto-unload)' : ''}`
                                        : 'Gemma (LTX-2 text encoder)'
                                    }
                                </div>
                                <a
                                    href="/studio"
                                    className="text-[10px] text-milimo-400/80 hover:text-milimo-300 flex items-center gap-1"
                                    title="Providers, discovery, role routing, and API keys live in Studio Setup"
                                >
                                    All settings <ExternalLink size={9} />
                                </a>
                            </div>
                        </div>
                    ) : null}
                </div>
            )}
        </div>
    );
};
