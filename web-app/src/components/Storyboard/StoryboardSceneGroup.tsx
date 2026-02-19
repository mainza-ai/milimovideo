import React, { useState } from 'react';
import { useTimelineStore } from '../../stores/timelineStore';
import { useShallow } from 'zustand/react/shallow';
import type { Shot, Scene, MatchedElement } from '../../stores/types';
import { StoryboardShotCard } from './StoryboardShotCard';
import {
    ChevronDown, ChevronRight, Plus, Sparkles,
    Film, Loader2, ImageIcon
} from 'lucide-react';

interface StoryboardSceneGroupProps {
    scene: Scene;
    shots: Shot[];
    shotOffset?: number;
}

export const StoryboardSceneGroup: React.FC<StoryboardSceneGroupProps> = ({ scene, shots, shotOffset = 0 }) => {
    const { batchGenerateShots, addShotToScene, reorderShotsInScene, updateSceneName, batchGenerateThumbnails } = useTimelineStore(useShallow(state => ({
        batchGenerateShots: state.batchGenerateShots,
        addShotToScene: state.addShotToScene,
        reorderShotsInScene: state.reorderShotsInScene,
        updateSceneName: state.updateSceneName,
        batchGenerateThumbnails: state.batchGenerateThumbnails,
    })));

    const [collapsed, setCollapsed] = useState(false);
    const [editingName, setEditingName] = useState(false);
    const [nameValue, setNameValue] = useState(scene.name);
    const [isBatchGenerating, setIsBatchGenerating] = useState(false);
    const [isGeneratingThumbnails, setIsGeneratingThumbnails] = useState(false);

    const pendingShots = shots.filter(s => !s.videoUrl && s.status !== 'generating');
    const generatingCount = shots.filter(s => s.status === 'generating' || s.isGenerating).length;
    const completedCount = shots.filter(s => s.videoUrl).length;

    const handleBatchGenerate = async (e: React.MouseEvent) => {
        e.stopPropagation();
        if (pendingShots.length === 0) return;
        setIsBatchGenerating(true);
        await batchGenerateShots(pendingShots.map(s => s.id));
        setIsBatchGenerating(false);
    };

    const handleAddShot = async (e: React.MouseEvent) => {
        e.stopPropagation();
        await addShotToScene(scene.id);
    };

    const handleSceneRenameBlur = () => {
        setEditingName(false);
        if (nameValue !== scene.name && scene.id !== '__unassigned__') {
            updateSceneName(scene.id, nameValue);
        }
    };

    const handleGenerateThumbnails = async (e: React.MouseEvent) => {
        e.stopPropagation();
        const withoutArt = shots.filter(s => !s.thumbnailUrl && !s.videoUrl);
        if (withoutArt.length === 0) return;
        setIsGeneratingThumbnails(true);
        await batchGenerateThumbnails(withoutArt.map(s => s.id));
        setIsGeneratingThumbnails(false);
    };

    // ─── Drag-and-drop ─────────────────────────────────────────

    const handleDragStart = (e: React.DragEvent, shotId: string) => {
        e.dataTransfer.setData('text/plain', shotId);
        e.dataTransfer.effectAllowed = 'move';
    };

    const handleDragOver = (e: React.DragEvent) => {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'move';
    };

    const handleDrop = async (e: React.DragEvent, targetShotId: string) => {
        e.preventDefault();
        const draggedShotId = e.dataTransfer.getData('text/plain');
        if (!draggedShotId || draggedShotId === targetShotId) return;

        const currentIds = shots.map(s => s.id);
        const fromIdx = currentIds.indexOf(draggedShotId);
        const toIdx = currentIds.indexOf(targetShotId);

        if (fromIdx === -1 || toIdx === -1) return;

        // Reorder: remove dragged and insert before target
        const newIds = [...currentIds];
        newIds.splice(fromIdx, 1);
        newIds.splice(toIdx, 0, draggedShotId);

        await reorderShotsInScene(scene.id, newIds);
    };

    return (
        <div className="bg-[#161616] rounded-xl border border-white/5 overflow-hidden mb-6">
            {/* Scene Header */}
            <div
                className="flex items-center justify-between px-4 py-3 bg-[#131313] border-b border-white/5 cursor-pointer select-none hover:bg-[#181818] transition-colors"
                onClick={() => setCollapsed(!collapsed)}
            >
                <div className="flex items-center gap-3">
                    {collapsed ? (
                        <ChevronRight size={16} className="text-white/30" />
                    ) : (
                        <ChevronDown size={16} className="text-white/30" />
                    )}

                    {editingName ? (
                        <input
                            autoFocus
                            className="bg-black/50 border border-milimo-500/30 rounded px-2 py-0.5 text-sm text-white font-semibold focus:outline-none"
                            value={nameValue}
                            onChange={(e) => setNameValue(e.target.value)}
                            onBlur={handleSceneRenameBlur}
                            onKeyDown={(e) => { if (e.key === 'Enter') handleSceneRenameBlur(); }}
                            onClick={(e) => e.stopPropagation()}
                        />
                    ) : (
                        <h4
                            className="text-sm font-semibold text-white/90 hover:text-milimo-400 transition-colors cursor-text"
                            onDoubleClick={(e) => { e.stopPropagation(); setEditingName(true); }}
                            title="Double-click to rename"
                        >
                            {scene.name}
                        </h4>
                    )}

                    {/* Status summary */}
                    <div className="flex items-center gap-2 text-[10px] font-mono text-white/30">
                        <span className="flex items-center gap-1">
                            <Film size={10} /> {shots.length} shots
                        </span>
                        {completedCount > 0 && (
                            <span className="text-emerald-400/60">{completedCount} ✓</span>
                        )}
                        {generatingCount > 0 && (
                            <span className="text-milimo-400/60 animate-pulse">{generatingCount} generating</span>
                        )}
                    </div>

                    {/* Element Cast Summary */}
                    {(() => {
                        const allMatches: MatchedElement[] = [];
                        const seen = new Set<string>();
                        for (const s of shots) {
                            for (const m of s.matchedElements || []) {
                                if (!seen.has(m.element_id)) {
                                    seen.add(m.element_id);
                                    allMatches.push(m);
                                }
                            }
                        }
                        if (allMatches.length === 0) return null;
                        const chars = allMatches.filter(m => m.element_type === 'character');
                        const locs = allMatches.filter(m => m.element_type === 'location');
                        const parts: string[] = [];
                        if (chars.length) parts.push(`👤 ${chars.map(c => c.element_name).join(', ')}`);
                        if (locs.length) parts.push(`📍 ${locs.map(l => l.element_name).join(', ')}`);
                        return (
                            <span className="text-[10px] text-white/40 ml-2">
                                {parts.join('  ·  ')}
                            </span>
                        );
                    })()}
                </div>

                <div className="flex items-center gap-2">
                    {/* Thumbnail Generation */}
                    {shots.some(s => !s.thumbnailUrl && !s.videoUrl) && (
                        <button
                            onClick={handleGenerateThumbnails}
                            disabled={isGeneratingThumbnails}
                            className="px-3 py-1.5 bg-violet-500/20 hover:bg-violet-500/30 text-violet-400 text-xs font-semibold rounded-lg flex items-center gap-1.5 transition-colors disabled:opacity-50"
                        >
                            {isGeneratingThumbnails ? (
                                <Loader2 size={12} className="animate-spin" />
                            ) : (
                                <ImageIcon size={12} />
                            )}
                            Concept Art
                        </button>
                    )}
                    {pendingShots.length > 0 && (
                        <button
                            onClick={handleBatchGenerate}
                            disabled={isBatchGenerating}
                            className="px-3 py-1.5 bg-milimo-500/20 hover:bg-milimo-500/30 text-milimo-400 text-xs font-semibold rounded-lg flex items-center gap-1.5 transition-colors disabled:opacity-50"
                        >
                            {isBatchGenerating ? (
                                <Loader2 size={12} className="animate-spin" />
                            ) : (
                                <Sparkles size={12} />
                            )}
                            Generate All ({pendingShots.length})
                        </button>
                    )}
                    <button
                        onClick={handleAddShot}
                        className="p-1.5 bg-white/5 hover:bg-white/10 rounded-lg text-white/40 hover:text-white transition-colors"
                        title="Add shot to scene"
                    >
                        <Plus size={14} />
                    </button>
                </div>
            </div>

            {/* Shots Grid */}
            {!collapsed && (
                <div className="p-4">
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                        {shots.map((shot, i) => (
                            <StoryboardShotCard
                                key={shot.id}
                                shot={shot}
                                index={i}
                                globalIndex={shotOffset + i + 1}
                                onDragStart={handleDragStart}
                                onDragOver={handleDragOver}
                                onDrop={handleDrop}
                            />
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
};
