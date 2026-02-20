import React, { useEffect, useState } from 'react';
import { useTimelineStore } from '../stores/timelineStore';
import { X, Play, Download, Loader, Calendar } from 'lucide-react';

interface RenderJob {
    id: string;
    project_id: string;
    status: string;
    progress: number;
    output_path: string | null;
    error_message: string | null;
    created_at: string;
}

interface ExportModalProps {
    onClose: () => void;
}

export const ExportModal: React.FC<ExportModalProps> = ({ onClose }) => {
    const project = useTimelineStore(state => state.project);
    const saveProject = useTimelineStore(state => state.saveProject);
    const addToast = useTimelineStore(state => state.addToast);

    const [renders, setRenders] = useState<RenderJob[]>([]);
    const [isLoading, setIsLoading] = useState(true);

    const fetchRenders = async () => {
        try {
            const res = await fetch(`http://localhost:8000/projects/${project.id}/renders`);
            const data = await res.json();
            setRenders(data);
        } catch (e) {
            console.error("Failed to fetch renders", e);
        } finally {
            setIsLoading(false);
        }
    };

    // Initial fetch + Setup polling for active renders
    useEffect(() => {
        fetchRenders();
        const interval = setInterval(() => {
            // Only poll if we have processing jobs
            if (renders.some(r => r.status === 'processing' || r.status === 'pending')) {
                fetchRenders();
            }
        }, 3000);
        return () => clearInterval(interval);
    }, [project.id, renders]);

    const handleNewExport = async () => {
        addToast("Saving project configuration...", "info");
        await saveProject();

        try {
            const res = await fetch(`http://localhost:8000/projects/${project.id}/render`, { method: 'POST' });
            if (!res.ok) {
                const errorText = await res.text();
                throw new Error(errorText);
            }
            addToast("Export queued successfully", "success");
            fetchRenders(); // Refresh list immediately to show new job
        } catch (e: any) {
            addToast(`Failed to start export: ${e.message || 'Server error'}`, "error");
        }
    };

    return (
        <div className="fixed inset-0 z-[100] bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
            <div className="bg-[#111] border border-white/10 rounded-xl shadow-2xl w-full max-w-2xl max-h-[85vh] flex flex-col">
                <div className="flex items-center justify-between p-4 flex-shrink-0 border-b border-white/5 bg-[#161616] rounded-t-xl">
                    <h2 className="text-lg font-bold">Exports & Renders</h2>
                    <button onClick={onClose} className="p-1 hover:bg-white/10 rounded-lg text-white/60 hover:text-white transition-colors">
                        <X size={20} />
                    </button>
                </div>

                <div className="p-4 flex-shrink-0 border-b border-white/5 flex gap-4">
                    <button
                        onClick={handleNewExport}
                        className="px-6 py-2.5 bg-milimo-500 rounded-lg text-sm font-semibold text-black hover:bg-milimo-400 flex items-center justify-center gap-2 transition-colors shadow-lg shadow-milimo-500/20 w-fit"
                    >
                        <Play size={16} fill="currentColor" /> Request New Export
                    </button>
                    <div className="flex flex-col justify-center text-xs text-white/50">
                        <p>Exports combine all V1/V2 video tracks and A1 audio tracks.</p>
                        <p>Resolution corresponds to Project Settings ({project.resolutionW}x{project.resolutionH}).</p>
                    </div>
                </div>

                <div className="flex-1 overflow-y-auto p-4 space-y-3">
                    {isLoading ? (
                        <div className="flex items-center justify-center py-10 text-white/30">
                            <Loader className="animate-spin mr-2" size={16} /> Loading Render Database...
                        </div>
                    ) : renders.length === 0 ? (
                        <div className="text-center py-12 text-white/30 text-sm">
                            <p>No exports found for this project.</p>
                            <p className="mt-1">Click "Request New Export" to generate the final MP4.</p>
                        </div>
                    ) : (
                        renders.map(r => (
                            <div key={r.id} className="bg-white/5 border border-white/10 rounded-lg p-3 flex flex-col">
                                <div className="flex justify-between items-start mb-2">
                                    <div className="flex items-center gap-2">
                                        <span className="font-mono text-xs text-white/60">{r.id}</span>
                                        <span className={`text-[10px] uppercase font-bold px-2 py-0.5 rounded-full ${r.status === 'completed' ? 'bg-green-500/20 text-green-400' :
                                            r.status === 'failed' ? 'bg-red-500/20 text-red-400' :
                                                'bg-blue-500/20 text-blue-400 animate-pulse'
                                            }`}>
                                            {r.status}
                                        </span>
                                    </div>
                                    <div className="flex gap-2">
                                        {r.status === 'completed' && r.output_path && (
                                            <a
                                                href={`http://localhost:8000${r.output_path}`}
                                                target="_blank" rel="noreferrer"
                                                download
                                                className="p-1.5 bg-white/10 hover:bg-white/20 rounded transition-colors"
                                                title="View / Download MP4"
                                            >
                                                <Download size={14} className="text-white" />
                                            </a>
                                        )}
                                    </div>
                                </div>

                                {r.status === 'processing' && (
                                    <div className="w-full bg-black rounded-full h-1.5 mb-2 overflow-hidden border border-white/10 relative">
                                        <div
                                            className="bg-milimo-500 h-1.5 rounded-full transition-all duration-300"
                                            style={{ width: `${Math.max(2, r.progress)}%` }}
                                        />
                                    </div>
                                )}

                                <div className="flex justify-between items-end text-xs text-white/40">
                                    <span className="flex items-center gap-1">
                                        <Calendar size={12} /> {new Date(r.created_at).toLocaleString()}
                                    </span>
                                    {r.error_message && (
                                        <span className="text-red-400 max-w-[60%] truncate" title={r.error_message}>{r.error_message}</span>
                                    )}
                                </div>
                            </div>
                        ))
                    )}
                </div>
            </div>
        </div>
    );
};
