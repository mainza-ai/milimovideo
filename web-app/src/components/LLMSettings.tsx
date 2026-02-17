import React, { useState, useEffect, useRef } from 'react';
import { Settings, ChevronDown, Cpu, Check, Loader2, AlertCircle, Eye } from 'lucide-react';

interface LLMConfig {
    provider: string;
    ollama_base_url: string;
    ollama_model: string;
    ollama_keep_alive: string;
}

interface OllamaModel {
    name: string;
    size: number;
    modified_at: string;
    is_vision: boolean;
    parameter_size: string;
}

export const LLMSettings: React.FC = () => {
    const [open, setOpen] = useState(false);
    const [config, setConfig] = useState<LLMConfig | null>(null);
    const [models, setModels] = useState<OllamaModel[]>([]);
    const [loading, setLoading] = useState(false);
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
        Promise.all([
            fetch('http://localhost:8000/settings/llm').then(r => r.json()),
            fetch('http://localhost:8000/settings/llm/models').then(r => r.json()).catch(() => ({ models: [] }))
        ]).then(([cfg, mdls]) => {
            setConfig(cfg);
            setModels(mdls.models || []);
        }).catch(() => setError('Failed to load settings'))
            .finally(() => setLoading(false));
    }, [open]);

    const updateConfig = async (patch: Partial<LLMConfig>) => {
        try {
            const res = await fetch('http://localhost:8000/settings/llm', {
                method: 'PATCH',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(patch)
            });
            const updated = await res.json();
            setConfig(updated);
        } catch {
            setError('Failed to update');
        }
    };

    const formatSize = (bytes: number) => {
        if (!bytes) return '';
        const gb = bytes / (1024 ** 3);
        return gb >= 1 ? `${gb.toFixed(1)}GB` : `${(bytes / (1024 ** 2)).toFixed(0)}MB`;
    };

    const isOllama = config?.provider === 'ollama';
    const keepAlive = config?.ollama_keep_alive !== '0';

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
                                    Prompt Enhancement
                                </label>
                                <div className="flex bg-white/5 p-0.5 rounded-lg border border-white/5">
                                    <button
                                        onClick={() => updateConfig({ provider: 'gemma' })}
                                        className={`flex-1 px-3 py-1.5 text-[11px] font-medium rounded-md transition-all flex items-center justify-center gap-1.5 ${!isOllama
                                            ? 'bg-milimo-500/20 text-milimo-400 shadow-sm border border-milimo-500/20'
                                            : 'text-white/40 hover:text-white/60'
                                            }`}
                                    >
                                        {!isOllama && <Check size={10} />}
                                        Gemma (Built-in)
                                    </button>
                                    <button
                                        onClick={() => updateConfig({ provider: 'ollama' })}
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
                                <div className="space-y-3 animate-in fade-in duration-200">
                                    {/* Model Dropdown */}
                                    <div>
                                        <label className="text-[10px] uppercase tracking-wider text-white/30 font-semibold block mb-1.5">
                                            Model
                                        </label>
                                        {models.length > 0 ? (
                                            <div className="relative">
                                                <select
                                                    value={config.ollama_model}
                                                    onChange={(e) => updateConfig({ ollama_model: e.target.value })}
                                                    className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-xs text-white appearance-none cursor-pointer hover:bg-white/8 focus:outline-none focus:border-milimo-500/50 transition-colors"
                                                >
                                                    {models.map(m => (
                                                        <option key={m.name} value={m.name} className="bg-[#1a1a1a] text-white">
                                                            {m.is_vision ? '👁️ ' : ''}{m.name} {m.parameter_size ? `(${m.parameter_size})` : formatSize(m.size) && `(${formatSize(m.size)})`}
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
                                        {/* Vision indicator for selected model */}
                                        {models.find(m => m.name === config.ollama_model)?.is_vision && (
                                            <div className="flex items-center gap-1.5 mt-1.5 text-[10px] text-cyan-400/70">
                                                <Eye size={10} />
                                                Vision model — can analyze reference images
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
                                                onClick={() => updateConfig({ ollama_keep_alive: keepAlive ? '0' : '5m' })}
                                                className={`relative w-8 h-4 rounded-full transition-colors ${keepAlive ? 'bg-milimo-500/50' : 'bg-white/10'}`}
                                            >
                                                <div className={`absolute top-0.5 w-3 h-3 bg-white rounded-full transition-transform ${keepAlive ? 'translate-x-4' : 'translate-x-0.5'}`} />
                                            </button>
                                        </div>
                                        <p className="text-[10px] text-white/20 mt-1">
                                            {keepAlive ? 'Model stays in RAM (faster prompts, uses ~49GB)' : 'Model unloads after use (saves RAM for generation)'}
                                        </p>
                                    </div>

                                    {/* Ollama URL */}
                                    <div>
                                        <label className="text-[10px] uppercase tracking-wider text-white/30 font-semibold block mb-1.5">
                                            Ollama URL
                                        </label>
                                        <input
                                            type="text"
                                            value={config.ollama_base_url}
                                            onChange={(e) => updateConfig({ ollama_base_url: e.target.value })}
                                            className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-xs text-white/70 font-mono focus:outline-none focus:border-milimo-500/50 transition-colors"
                                        />
                                    </div>
                                </div>
                            )}

                            {/* Current Status */}
                            <div className="pt-2 border-t border-white/5">
                                <div className="flex items-center gap-2 text-[10px] text-white/25">
                                    <div className={`w-1.5 h-1.5 rounded-full ${isOllama ? 'bg-emerald-400' : 'bg-amber-400'}`} />
                                    {isOllama
                                        ? `Ollama → ${config.ollama_model}${!keepAlive ? ' (auto-unload)' : ''}`
                                        : 'Gemma (LTX-2 text encoder)'
                                    }
                                </div>
                            </div>
                        </div>
                    ) : null}
                </div>
            )}
        </div>
    );
};
