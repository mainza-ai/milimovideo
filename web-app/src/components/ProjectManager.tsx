import { useState, useEffect } from 'react';
import { useTimelineStore } from '../stores/timelineStore';
import { X, Plus, FolderOpen, Trash2, Clock, Film } from 'lucide-react';
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

    // Create Form State
    const [isCreating, setIsCreating] = useState(false);
    const [newName, setNewName] = useState('');
    const [resW, setResW] = useState(768);
    const [resH, setResH] = useState(512);
    const [fps, setFps] = useState(24);
    const [seed, setSeed] = useState(42);

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

        await createNewProject(newName, {
            resolutionW: resW,
            resolutionH: resH,
            fps: fps,
            seed: seed
        });

        // Reset
        setNewName('');
        setSeed(Math.floor(Math.random() * 100000));
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

    const formatDate = (ts: number) => new Date(ts * 1000).toLocaleDateString();

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4" onClick={onClose}>
            <div className="bg-[#111] border border-white/10 w-full max-w-2xl rounded-2xl overflow-hidden shadow-2xl flex flex-col max-h-[85vh] animate-in fade-in zoom-in-95 duration-200" onClick={e => e.stopPropagation()}>

                {/* Header */}
                <div className="p-5 border-b border-white/5 flex items-center justify-between bg-[#151515]">
                    <h2 className="text-xl font-bold text-white flex items-center gap-3">
                        <FolderOpen size={24} className="text-milimo-500" />
                        <div>
                            Projects
                            <span className="block text-[10px] text-white/30 font-normal uppercase tracking-wider">Manage your video projects</span>
                        </div>
                    </h2>
                    <div className="flex items-center gap-2">
                        {!isCreating && (
                            <button
                                onClick={() => setIsCreating(true)}
                                className="px-4 py-2 bg-milimo-500 rounded-lg text-xs font-bold text-black hover:bg-milimo-400 flex items-center gap-2 transition-colors shadow-lg shadow-milimo-500/10"
                            >
                                <Plus size={16} /> New Project
                            </button>
                        )}
                        <button onClick={onClose} className="p-2 hover:bg-white/5 rounded-lg text-white/50 hover:text-white transition-colors">
                            <X size={20} />
                        </button>
                    </div>
                </div>

                {/* Create Form */}
                {isCreating && (
                    <div className="p-6 border-b border-white/5 bg-[#1a1a1a] space-y-4">
                        <div className="space-y-1">
                            <label className="text-xs font-bold text-white/40 uppercase tracking-widest pl-1">Project Name</label>
                            <input
                                autoFocus
                                type="text"
                                placeholder="My Awesome Video..."
                                className="w-full bg-black/30 border border-white/10 rounded-lg px-4 py-3 text-sm text-white focus:outline-none focus:border-milimo-500 transition-colors"
                                value={newName}
                                onChange={e => setNewName(e.target.value)}
                                onKeyDown={e => e.key === 'Enter' && handleCreate()}
                            />
                        </div>

                        <div className="grid grid-cols-3 gap-4">
                            <div className="space-y-1">
                                <label className="text-xs font-bold text-white/40 uppercase tracking-widest pl-1">Resolution</label>
                                <select
                                    className="w-full bg-black/30 border border-white/10 rounded-lg px-3 py-2.5 text-xs text-white focus:outline-none focus:border-milimo-500"
                                    value={`${resW}x${resH}`}
                                    onChange={e => {
                                        const [w, h] = e.target.value.split('x').map(Number);
                                        setResW(w); setResH(h);
                                    }}
                                >
                                    <option value="1280x704">1280x704 (HD Aligned)</option>
                                    <option value="1920x1088">1920x1088 (FHD Aligned)</option>
                                    <option value="3840x2176">3840x2176 (4K UHD Aligned)</option>
                                    <option value="768x512">768x512 (Standard Landscape)</option>
                                    <option value="512x768">512x768 (Standard Portrait)</option>
                                    <option value="1024x576">1024x576 (Wide SD)</option>
                                </select>
                            </div>
                            <div className="space-y-1">
                                <label className="text-xs font-bold text-white/40 uppercase tracking-widest pl-1">FPS</label>
                                <select
                                    className="w-full bg-black/30 border border-white/10 rounded-lg px-3 py-2.5 text-xs text-white focus:outline-none focus:border-milimo-500"
                                    value={fps}
                                    onChange={e => setFps(Number(e.target.value))}
                                >
                                    <option value="24">24 fps (Cinema)</option>
                                    <option value="25">25 fps (PAL)</option>
                                    <option value="30">30 fps (NTSC)</option>
                                    <option value="50">50 fps (High PAL)</option>
                                    <option value="60">60 fps (High)</option>
                                </select>
                            </div>
                            <div className="space-y-1">
                                <label className="text-xs font-bold text-white/40 uppercase tracking-widest pl-1">Seed</label>
                                <div className="flex gap-2">
                                    <input
                                        type="number"
                                        className="w-full bg-black/30 border border-white/10 rounded-lg px-3 py-2.5 text-xs text-white focus:outline-none focus:border-milimo-500"
                                        value={seed}
                                        onChange={e => setSeed(Number(e.target.value))}
                                    />
                                    <button
                                        onClick={() => setSeed(Math.floor(Math.random() * 100000))}
                                        className="px-2 bg-white/5 border border-white/10 rounded-lg hover:bg-white/10 text-white/50"
                                    >
                                        🎲
                                    </button>
                                </div>
                            </div>
                        </div>

                        <div className="flex gap-3 pt-2">
                            <button
                                onClick={handleCreate}
                                className="flex-1 py-2.5 bg-milimo-500 hover:bg-milimo-400 rounded-lg text-sm font-bold text-black transition-colors shadow-lg shadow-milimo-500/20"
                            >
                                Create Project
                            </button>
                            <button
                                onClick={() => setIsCreating(false)}
                                className="px-6 py-2.5 bg-white/5 hover:bg-white/10 rounded-lg text-sm text-white/70 hover:text-white transition-colors"
                            >
                                Cancel
                            </button>
                        </div>
                    </div>
                )}

                {/* List */}
                <div className="flex-1 overflow-y-auto p-4 space-y-2 bg-[#0a0a0a]">
                    <div className="grid grid-cols-1 gap-2">
                        {projects.map(p => (
                            <div
                                key={p.id}
                                onClick={() => handleLoad(p.id)}
                                className={clsx(
                                    "group flex items-center justify-between p-4 rounded-xl border transition-all cursor-pointer relative overflow-hidden",
                                    currentProject.id === p.id
                                        ? "bg-milimo-900/10 border-milimo-500/50"
                                        : "bg-white/5 border-white/5 hover:border-white/20 hover:bg-white/10"
                                )}
                            >
                                {currentProject.id === p.id && (
                                    <div className="absolute left-0 top-0 bottom-0 w-1 bg-milimo-500" />
                                )}

                                <div className="flex items-center gap-4 pl-2">
                                    <div className={clsx(
                                        "w-10 h-10 rounded-lg flex items-center justify-center",
                                        currentProject.id === p.id ? "bg-milimo-500 text-black" : "bg-white/5 text-white/20"
                                    )}>
                                        <Film size={20} />
                                    </div>
                                    <div className="flex flex-col">
                                        <span className={clsx(
                                            "font-bold text-sm",
                                            currentProject.id === p.id ? "text-milimo-400" : "text-white"
                                        )}>
                                            {p.name}
                                        </span>
                                        <span className="text-[10px] text-white/30 flex items-center gap-1 mt-0.5">
                                            <Clock size={10} /> Updated {formatDate(p.updated_at)}
                                        </span>
                                    </div>
                                </div>

                                <div className="flex items-center gap-2">
                                    {currentProject.id === p.id && (
                                        <span className="mr-2 text-[9px] bg-milimo-500/20 text-milimo-400 border border-milimo-500/20 px-2 py-1 rounded font-bold uppercase tracking-wider">
                                            Active
                                        </span>
                                    )}
                                    <button
                                        onClick={(e) => handleDelete(e, p.id)}
                                        className="p-2 bg-white/5 text-white/30 hover:bg-red-500/20 hover:text-red-500 rounded-lg transition-colors opacity-0 group-hover:opacity-100"
                                        title="Delete Project"
                                    >
                                        <Trash2 size={16} />
                                    </button>
                                </div>
                            </div>
                        ))}
                    </div>

                    {projects.length === 0 && (
                        <div className="flex flex-col items-center justify-center py-20 text-white/20">
                            <FolderOpen size={48} className="mb-4 opacity-20" />
                            <p className="text-sm">No projects yet.</p>
                            <p className="text-xs opacity-50">Create one to get started.</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};
