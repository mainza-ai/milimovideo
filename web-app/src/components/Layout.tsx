import React, { useEffect, useState } from 'react';
import { MediaLibrary } from './Library/MediaLibrary';

import { InspectorPanel } from './Inspector/InspectorPanel';
import { VisualTimeline } from './Timeline/VisualTimeline';
import { StoryboardView } from './Storyboard/StoryboardView';
import { ElementsView } from './Elements/ElementsView';
import { ImagesView } from './Images/ImagesView';
import { useTimelineStore, getLastProjectId } from '../stores/timelineStore';
import { ProjectManager } from './ProjectManager';
import { Save, Command, Share, FolderOpen, Undo as UndoIcon, Redo as RedoIcon } from 'lucide-react';

export const Layout = ({ children }: { children: React.ReactNode }) => {
    const { project, saveProject, addToast, isPlaying, setIsPlaying, deleteShot, selectedShotId, loadProject, viewMode, setViewMode } = useTimelineStore();
    const [showProjects, setShowProjects] = useState(false);

    // Access temporal store (zundo)
    const temporal = (useTimelineStore as any).temporal;

    // Auto-load last opened project on mount
    useEffect(() => {
        const lastProjectId = getLastProjectId();
        if (lastProjectId) {
            loadProject(lastProjectId).catch((e) => {
                console.error("Layout: Failed to restore last project:", e);
                // Force clear storage
                localStorage.removeItem('milimo_last_project_id');
                // We rely on store to default to empty project, so no need to force reload page, 
                // just let the user see the empty state.
            });
        }
    }, []); // Run once on mount

    // Auto-save feedback or logic could go here

    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            // Ignore if typing in an input
            if ((e.target as HTMLElement).tagName === 'INPUT' || (e.target as HTMLElement).tagName === 'TEXTAREA') return;

            // Undo/Redo
            if ((e.metaKey || e.ctrlKey) && e.key === 'z') {
                e.preventDefault();
                if (e.shiftKey) {
                    temporal.getState().redo();
                } else {
                    temporal.getState().undo();
                }
                return;
            }

            if (e.code === 'Space') {
                e.preventDefault();
                setIsPlaying(!isPlaying);
            } else if ((e.metaKey || e.ctrlKey) && e.key === 's') {
                e.preventDefault();
                saveProject();
            } else if (e.key === 'Delete' || e.key === 'Backspace') {
                if (selectedShotId) deleteShot(selectedShotId);
            }
        };

        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [isPlaying, setIsPlaying, saveProject, deleteShot, selectedShotId, temporal]);

    const handleExport = async () => {
        addToast("Saving project...", "info");
        await saveProject();

        addToast("Starting Export...", "info");
        try {
            const res = await fetch(`http://localhost:8000/projects/${project.id}/render`, { method: 'POST' });
            const text = await res.text();
            let data;
            try {
                data = JSON.parse(text);
            } catch (e) {
                addToast("Export failed: Server error", "error");
                return;
            }

            if (data.status === 'completed') {
                addToast("Export Complete!", "success");
                window.open(`http://localhost:8000${data.video_url}`, '_blank');
            } else {
                addToast("Export failed", "error");
            }
        } catch (e) {
            addToast("Export failed", "error");
        }
    };

    return (
        <div className="w-screen h-screen bg-[#050505] text-white flex flex-col font-sans overflow-hidden selection:bg-milimo-500/30">
            {/* Header / Toolbar */}
            <header className="h-14 border-b border-white/5 bg-[#0a0a0a] flex items-center justify-between px-6 shrink-0 z-50">
                <div className="flex items-center gap-4">
                    <button onClick={() => setShowProjects(true)} className="hover:opacity-80 transition-opacity" title="Manage Projects">
                        <div className="w-8 h-8 rounded-lg flex items-center justify-center overflow-hidden">
                            <img src="/logo.png" alt="Milimo" className="w-full h-full object-contain" />
                        </div>
                    </button>
                    <div className="flex flex-col">
                        <span className="text-sm font-bold tracking-wide flex items-center gap-2">
                            <button
                                onClick={() => setShowProjects(true)}
                                className="hover:text-milimo-400 transition-colors flex items-center gap-2"
                            >
                                <FolderOpen size={14} className="text-white/40" />
                                Milimo Video
                            </button>
                            <span className="text-white/20">|</span>
                            <input
                                className="bg-transparent border-none focus:outline-none text-white font-bold w-32 focus:w-64 transition-all placeholder-white/30"
                                value={project.name}
                                onChange={(e) => useTimelineStore.getState().setProject({ ...project, name: e.target.value })}
                            />
                            {/* View Switcher */}
                            <div className="flex bg-white/5 p-0.5 rounded-lg border border-white/5 ml-4">
                                <button
                                    onClick={() => setViewMode('timeline')}
                                    className={`px-3 py-1 text-[10px] font-medium rounded-md transition-all ${viewMode === 'timeline' ? 'bg-white/10 text-white shadow-sm' : 'text-white/40 hover:text-white/60'}`}
                                >
                                    Timeline
                                </button>
                                <button
                                    onClick={() => setViewMode('elements')}
                                    className={`px-3 py-1 text-[10px] font-medium rounded-md transition-all ${viewMode === 'elements' ? 'bg-white/10 text-white shadow-sm' : 'text-white/40 hover:text-white/60'}`}
                                >
                                    Elements
                                </button>
                                <button
                                    onClick={() => setViewMode('storyboard')}
                                    className={`px-3 py-1 text-[10px] font-medium rounded-md transition-all ${viewMode === 'storyboard' ? 'bg-white/10 text-white shadow-sm' : 'text-white/40 hover:text-white/60'}`}
                                >
                                    Storyboard
                                </button>
                                <button
                                    onClick={() => setViewMode('images')}
                                    className={`px-3 py-1 text-[10px] font-medium rounded-md transition-all ${viewMode === 'images' ? 'bg-white/10 text-white shadow-sm' : 'text-white/40 hover:text-white/60'}`}
                                >
                                    Images
                                </button>
                            </div>
                        </span>
                        <span className="text-[10px] text-white/30 font-mono uppercase">Resolution: {project.resolutionW}x{project.resolutionH}</span>
                    </div>
                </div>


                <div className="flex items-center gap-4">
                    {/* ... (Undo/Redo, Save, Export remains same) */}
                    <div className="flex items-center gap-1 border-r border-white/10 pr-4 mr-4">
                        <button
                            onClick={() => temporal.getState().undo()}
                            className="p-1.5 text-white/50 hover:text-white hover:bg-white/10 rounded transition-colors"
                            title="Undo (Ctrl+Z)"
                        >
                            <UndoIcon size={16} />
                        </button>
                        <button
                            onClick={() => temporal.getState().redo()}
                            className="p-1.5 text-white/50 hover:text-white hover:bg-white/10 rounded transition-colors"
                            title="Redo (Ctrl+Shift+Z)"
                        >
                            <RedoIcon size={16} />
                        </button>
                    </div>

                    <div className="flex items-center gap-2 text-[10px] text-white/30 mr-4">
                        <Command size={12} /> <span className="font-mono">S</span> to Save
                    </div>
                    <button
                        onClick={() => saveProject()}
                        className="px-4 py-1.5 bg-white/5 border border-white/10 rounded-md text-xs font-semibold hover:bg-white/10 flex items-center gap-2 transition-colors"
                    >
                        <Save size={14} /> Save
                    </button>
                    <button
                        onClick={handleExport}
                        className="px-4 py-1.5 bg-milimo-500 rounded-md text-xs font-semibold text-black hover:bg-milimo-400 flex items-center gap-2 transition-colors shadow-lg shadow-milimo-500/20"
                    >
                        <Share size={14} /> Export
                    </button>
                </div>
            </header>

            {/* Workspace Grid */}
            <div className="flex-1 flex overflow-hidden">
                {/* Left: Library - Hide in Elements view? Or keep? Let's keep for consistency or hide if needing space. */}
                {/* User asked for Elements to be a main view. Let's hide sidebar in Elements or Storyboard mode for full focus? */}
                {/* Or keep generic layout. Let's keep Layout consistent for now. */}
                <MediaLibrary />

                {/* Center: Stage & Timeline */}
                <div className="flex-1 flex flex-col min-w-0 bg-[#0f0f0f]">
                    {/* Main Content (Player / Storyboard / Elements) */}
                    <div className="flex-1 overflow-hidden relative flex items-center justify-center">
                        {viewMode === 'timeline' && children}
                        {viewMode === 'storyboard' && (
                            <div className="absolute inset-0 z-10 bg-[#050505] w-full h-full">
                                <StoryboardView />
                            </div>
                        )}
                        {viewMode === 'elements' && (
                            <div className="absolute inset-0 z-10 bg-[#050505] w-full h-full">
                                <ElementsView />
                            </div>
                        )}
                        {viewMode === 'images' && (
                            <div className="absolute inset-0 z-10 bg-[#050505] w-full h-full">
                                <ImagesView />
                            </div>
                        )}
                    </div>

                    {/* Bottom: Timeline (Only in Timeline View) */}
                    {viewMode === 'timeline' && (
                        <div className="h-80 shrink-0 z-10 border-t border-white/5">
                            <VisualTimeline />
                        </div>
                    )}
                </div>

                {/* Right: Inspector */}
                <InspectorPanel />
            </div>

            {/* Modals */}
            {showProjects && <ProjectManager onClose={() => setShowProjects(false)} />}
        </div >
    );
};
