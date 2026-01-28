import { useState, useEffect } from 'react';
import { useTimelineStore } from '../stores/timelineStore';
import { X, Plus, FolderOpen, Trash2, Clock } from 'lucide-react';
import { clsx } from 'clsx';

interface ProjectInfo {
    id: string;
    name: string;
    created_at: number;
    updated_at: number;
}

export const ProjectManager = ({ onClose }: { onClose: () => void }) => {
    const { project: currentProject, loadProject, createNewProject, deleteProject } = useTimelineStore();
    const [projects, setProjects] = useState<ProjectInfo[]>([]);
    const [isCreating, setIsCreating] = useState(false);
    const [newName, setNewName] = useState('');

    useEffect(() => {
        fetchProjects();
    }, []);

    const fetchProjects = async () => {
        try {
            const res = await fetch('http://localhost:8000/projects');
            const data = await res.json();
            setProjects(data);
        } catch (e) {
            console.error(e);
        }
    };

    const handleCreate = async () => {
        if (!newName.trim()) return;
        await createNewProject(newName);
        setNewName('');
        setIsCreating(false);
        fetchProjects(); // Refresh list
    };

    const handleLoad = async (id: string) => {
        await loadProject(id);
        onClose();
    };

    const handleDelete = async (e: React.MouseEvent, id: string) => {
        e.stopPropagation();
        if (!confirm("Delete this project? This cannot be undone.")) return;

        await deleteProject(id);
        await fetchProjects();
    };

    const formatDate = (ts: number) => new Date(ts * 1000).toLocaleDateString() + ' ' + new Date(ts * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4" onClick={onClose}>
            <div className="bg-[#111] border border-white/10 w-full max-w-2xl rounded-2xl overflow-hidden shadow-2xl flex flex-col max-h-[80vh]" onClick={e => e.stopPropagation()}>

                {/* Header */}
                <div className="p-4 border-b border-white/5 flex items-center justify-between bg-[#151515]">
                    <h2 className="text-lg font-bold text-white flex items-center gap-2">
                        <FolderOpen size={20} className="text-milimo-500" />
                        Projects
                    </h2>
                    <div className="flex items-center gap-2">
                        {!isCreating && (
                            <button
                                onClick={() => setIsCreating(true)}
                                className="px-3 py-1.5 bg-milimo-500 rounded text-xs font-bold text-black hover:bg-milimo-400 flex items-center gap-1 transition-colors"
                            >
                                <Plus size={14} /> New Project
                            </button>
                        )}
                        <button onClick={onClose} className="p-2 hover:bg-white/5 rounded-lg text-white/50 hover:text-white transition-colors">
                            <X size={18} />
                        </button>
                    </div>
                </div>

                {/* Create Form */}
                {isCreating && (
                    <div className="p-4 border-b border-white/5 bg-[#1a1a1a]">
                        <div className="flex gap-2">
                            <input
                                autoFocus
                                type="text"
                                placeholder="Project Name..."
                                className="flex-1 bg-black/30 border border-white/10 rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-milimo-500"
                                value={newName}
                                onChange={e => setNewName(e.target.value)}
                                onKeyDown={e => e.key === 'Enter' && handleCreate()}
                            />
                            <button
                                onClick={handleCreate}
                                className="px-4 py-2 bg-white/10 hover:bg-white/20 rounded text-sm font-bold text-white transition-colors"
                            >
                                Create
                            </button>
                            <button
                                onClick={() => setIsCreating(false)}
                                className="px-4 py-2 text-sm text-white/50 hover:text-white transition-colors"
                            >
                                Cancel
                            </button>
                        </div>
                    </div>
                )}

                {/* List */}
                <div className="flex-1 overflow-y-auto p-4 space-y-2">
                    {projects.map(p => (
                        <div
                            key={p.id}
                            onClick={() => handleLoad(p.id)}
                            className={clsx(
                                "group flex items-center justify-between p-4 rounded-xl border transition-all cursor-pointer",
                                currentProject.id === p.id
                                    ? "bg-milimo-500/10 border-milimo-500/30"
                                    : "bg-white/5 border-white/5 hover:border-white/20 hover:bg-white/10"
                            )}
                        >
                            <div className="flex flex-col">
                                <span className={clsx(
                                    "font-bold text-sm",
                                    currentProject.id === p.id ? "text-milimo-400" : "text-white"
                                )}>
                                    {p.name}
                                    {currentProject.id === p.id && <span className="ml-2 text-[10px] bg-milimo-500 text-black px-1.5 py-0.5 rounded font-extrabold uppercase">Active</span>}
                                </span>
                                <span className="text-[10px] text-white/30 flex items-center gap-1 mt-1">
                                    <Clock size={10} /> {formatDate(p.updated_at)}
                                </span>
                            </div>

                            <div className="flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                                <button
                                    onClick={(e) => handleDelete(e, p.id)}
                                    className="p-2 bg-red-500/10 text-red-500 hover:bg-red-500 hover:text-white rounded transition-colors"
                                    title="Delete Project"
                                >
                                    <Trash2 size={14} />
                                </button>
                            </div>
                        </div>
                    ))}

                    {projects.length === 0 && (
                        <div className="text-center py-12 text-white/20 text-sm italic">
                            No projects found.
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};
