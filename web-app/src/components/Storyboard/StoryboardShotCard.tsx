import React, { useState, useRef, useCallback } from 'react';
import { useTimelineStore } from '../../stores/timelineStore';
import { useShallow } from 'zustand/react/shallow';
import type { Shot } from '../../stores/types';
import {
    GripVertical, Trash2, Film, Image as ImageIcon,
    RotateCcw, CheckCircle, AlertCircle, Loader2
} from 'lucide-react';
import { ElementBadgeRow } from './ElementBadge';

const SHOT_TYPE_LABELS: Record<string, string> = {
    close_up: 'CU', medium: 'MS', wide: 'WS',
    establishing: 'EST', insert: 'INS', tracking: 'TRK'
};

const SHOT_TYPE_COLORS: Record<string, string> = {
    close_up: 'bg-rose-500/20 text-rose-400',
    medium: 'bg-blue-500/20 text-blue-400',
    wide: 'bg-emerald-500/20 text-emerald-400',
    establishing: 'bg-amber-500/20 text-amber-400',
    insert: 'bg-purple-500/20 text-purple-400',
    tracking: 'bg-cyan-500/20 text-cyan-400',
};

interface StoryboardShotCardProps {
    shot: Shot;
    index: number;
    globalIndex?: number;
    onDragStart?: (e: React.DragEvent, shotId: string) => void;
    onDragOver?: (e: React.DragEvent) => void;
    onDrop?: (e: React.DragEvent, shotId: string) => void;
}

export const StoryboardShotCard: React.FC<StoryboardShotCardProps> = ({
    shot, index, globalIndex, onDragStart, onDragOver, onDrop
}) => {
    const { selectShot, selectedShotId, patchShot, generateShot, cancelShotGeneration, deleteShotFromStoryboard, generateThumbnail, addConditioningToShot } = useTimelineStore(useShallow(state => ({
        selectShot: state.selectShot,
        selectedShotId: state.selectedShotId,
        patchShot: state.patchShot,
        generateShot: state.generateShot,
        cancelShotGeneration: state.cancelShotGeneration,
        deleteShotFromStoryboard: state.deleteShotFromStoryboard,
        generateThumbnail: state.generateThumbnail,
        addConditioningToShot: state.addConditioningToShot,
    })));

    const [editingField, setEditingField] = useState<'action' | 'dialogue' | 'character' | null>(null);
    const [editValue, setEditValue] = useState('');
    const [confirmDelete, setConfirmDelete] = useState(false);
    const editRef = useRef<HTMLTextAreaElement | HTMLInputElement>(null);
    const debounceRef = useRef<ReturnType<typeof setTimeout>>(undefined);

    const startEdit = useCallback((field: 'action' | 'dialogue' | 'character', value: string) => {
        setEditingField(field);
        setEditValue(value || '');
        setTimeout(() => editRef.current?.focus(), 50);
    }, []);

    const commitEdit = useCallback(() => {
        if (editingField && editValue !== (shot[editingField] ?? '')) {
            clearTimeout(debounceRef.current);
            debounceRef.current = setTimeout(() => {
                patchShot(shot.id, { [editingField]: editValue });
            }, 300);
        }
        setEditingField(null);
    }, [editingField, editValue, shot, patchShot]);

    const handleGenerate = (e: React.MouseEvent) => {
        e.stopPropagation();
        generateShot(shot.id);
    };

    const handleDelete = (e: React.MouseEvent) => {
        e.stopPropagation();
        if (confirmDelete) {
            deleteShotFromStoryboard(shot.id);
            setConfirmDelete(false);
        } else {
            setConfirmDelete(true);
            setTimeout(() => setConfirmDelete(false), 3000);
        }
    };

    const statusIcon = () => {
        switch (shot.status) {
            case 'completed': return <CheckCircle size={12} className="text-emerald-400" />;
            case 'generating': return <Loader2 size={12} className="text-milimo-400 animate-spin" />;
            case 'failed': return <AlertCircle size={12} className="text-red-400" />;
            default: return null;
        }
    };

    const handleCardDragOver = (e: React.DragEvent) => {
        e.preventDefault();
        // If it has JSON (likely asset), allow copy
        if (e.dataTransfer.types.includes('application/json')) {
            e.dataTransfer.dropEffect = 'copy';
        } else {
            // Otherwise defer to parent reorder
            onDragOver?.(e);
        }
    };

    const handleCardDrop = (e: React.DragEvent) => {
        e.preventDefault();

        // 1. Try handling Media Asset Drop
        const jsonData = e.dataTransfer.getData('application/json');
        if (jsonData) {
            try {
                const asset = JSON.parse(jsonData);
                if (asset.id && asset.type) {
                    e.stopPropagation();
                    // Add conditioning
                    addConditioningToShot(shot.id, {
                        type: asset.type === 'video' ? 'video' : 'image',
                        path: asset.url,
                        frameIndex: 0,
                        strength: 0.8
                    });
                    // Visual feedback could be added here (e.g. toast provided by store?)
                    return;
                }
            } catch (err) {
                // Not valid JSON or asset, ignore
            }
        }

        // 2. Fallback to Shot Reordering
        onDrop?.(e, shot.id);
    };

    const isSelected = shot.id === selectedShotId;
    const shotTypeKey = shot.shotType || 'medium';


    return (
        <div
            draggable
            onDragStart={(e) => onDragStart?.(e, shot.id)}
            onDragOver={handleCardDragOver}
            onDrop={handleCardDrop}
            onClick={() => selectShot(shot.id)}
            className={`relative group bg-[#1a1a1a] rounded-xl overflow-hidden border-2 transition-all cursor-pointer flex flex-col ${isSelected
                ? 'border-milimo-500 shadow-xl shadow-milimo-500/10'
                : 'border-transparent hover:border-white/20'
                }`}
        >
            {/* Header */}
            <div className="px-3 py-2 flex justify-between items-center bg-black/40 border-b border-white/5">
                <div className="flex items-center gap-2">
                    {globalIndex != null && (
                        <span className="w-5 h-5 rounded-full bg-milimo-500/20 text-milimo-400 text-[10px] font-bold flex items-center justify-center">
                            {globalIndex}
                        </span>
                    )}
                    <span className="text-xs font-mono text-white/40">SHOT {index + 1}</span>
                    {shot.shotType && (
                        <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded ${SHOT_TYPE_COLORS[shotTypeKey] || 'bg-white/10 text-white/50'}`}>
                            {SHOT_TYPE_LABELS[shotTypeKey] || shotTypeKey.toUpperCase()}
                        </span>
                    )}
                </div>
                <div className="flex items-center gap-2">
                    {statusIcon()}
                    {shot.statusMessage && (
                        <span className="text-[10px] uppercase font-bold text-milimo-400 animate-pulse">
                            {shot.statusMessage}
                        </span>
                    )}
                    <GripVertical size={14} className="text-white/20 hover:text-white cursor-grab active:cursor-grabbing" />
                </div>
            </div>

            {/* Thumbnail */}
            <div className="aspect-video bg-black relative w-full">
                {shot.videoUrl ? (
                    shot.videoUrl.match(/\.(jpg|jpeg|png|webp)$/i) ?
                        <img src={shot.videoUrl} className="w-full h-full object-cover" alt="" />
                        : <video src={shot.videoUrl} className="w-full h-full object-cover" controls={false} />
                ) : shot.thumbnailUrl ? (
                    <img src={shot.thumbnailUrl} className="w-full h-full object-cover opacity-80" alt="Concept art" />
                ) : (
                    <div className="absolute inset-0 flex flex-col items-center justify-center text-white/10 p-4 text-center">
                        <ImageIcon size={32} className="mb-2 opacity-50" />
                        <p className="text-[10px] line-clamp-3 opacity-50">{shot.action || shot.prompt}</p>
                    </div>
                )}

                <div className="absolute inset-0 bg-gradient-to-t from-black/90 via-transparent to-transparent opacity-60 pointer-events-none" />

                {/* Prompt overlay */}
                <div className="absolute bottom-2 left-3 right-3 text-xs text-white/90 line-clamp-3 font-medium pointer-events-none drop-shadow-md">
                    {shot.character && (
                        <span className="block text-[10px] font-bold text-milimo-300 uppercase mb-0.5 tracking-wider">
                            {shot.character}
                        </span>
                    )}
                    {shot.dialogue ? (
                        <span className="italic text-white">"{shot.dialogue}"</span>
                    ) : (
                        shot.action || shot.prompt
                    )}
                </div>

                {/* Progress bar and Cancelling UI */}
                {shot.isGenerating && (
                    <>
                        <div className="absolute inset-0 bg-black/40 flex flex-col items-center justify-center backdrop-blur-[2px]">
                            <button
                                onClick={(e) => {
                                    e.stopPropagation();
                                    cancelShotGeneration(shot.id);
                                }}
                                className="px-3 py-1.5 bg-red-500/20 hover:bg-red-500/40 text-red-200 text-xs rounded uppercase font-bold tracking-wider cursor-pointer border border-red-500/30 transition-colors"
                            >
                                Cancel
                            </button>
                        </div>
                        {shot.progress != null && shot.progress > 0 && (
                            <div className="absolute bottom-0 left-0 right-0 h-1 bg-black/50 z-10">
                                <div
                                    className="h-full bg-milimo-500 transition-all duration-300"
                                    style={{ width: `${shot.progress}%` }}
                                />
                            </div>
                        )}
                    </>
                )}
            </div>

            {/* Action Bar (Always Visible) */}
            <div className="flex bg-[#111] p-2 border-b border-t border-white/5 gap-2 items-center justify-between">
                <div className="flex gap-2 flex-1">
                    {!shot.videoUrl && (
                        <button
                            onClick={(e) => { e.stopPropagation(); handleGenerate(e); }}
                            className="flex-1 py-1.5 px-2 bg-milimo-500 hover:bg-milimo-400 text-black font-bold rounded flex items-center justify-center gap-1.5 text-[10px] uppercase transition-colors"
                        >
                            <Film size={12} /> {shot.thumbnailUrl ? "Generate Video" : "Video"}
                        </button>
                    )}
                    {!shot.videoUrl && !shot.thumbnailUrl && (
                        <button
                            onClick={(e) => { e.stopPropagation(); generateThumbnail(shot.id); }}
                            className="flex-1 py-1.5 px-2 bg-violet-500/20 hover:bg-violet-500/30 text-violet-300 font-bold rounded flex items-center justify-center gap-1.5 text-[10px] uppercase transition-colors"
                        >
                            <ImageIcon size={12} /> Concept Art
                        </button>
                    )}
                    {!shot.videoUrl && shot.thumbnailUrl && (
                        <button
                            className="p-1.5 bg-white/10 hover:bg-white/20 text-white font-semibold rounded transition-colors text-xs"
                            onClick={(e) => { e.stopPropagation(); generateThumbnail(shot.id, true); }}
                            title="Regenerate Concept Art"
                        >
                            <RotateCcw size={12} />
                        </button>
                    )}
                    {shot.videoUrl && (
                        <>
                            <button
                                className="flex-1 py-1.5 px-2 bg-white/5 hover:bg-white/10 text-white/80 font-bold rounded flex items-center justify-center gap-1.5 text-[10px] uppercase transition-colors"
                                onClick={(e) => { e.stopPropagation(); handleGenerate(e); }}
                            >
                                <RotateCcw size={12} /> Regen Video
                            </button>
                            <button
                                className="p-1.5 bg-white/5 hover:bg-white/10 text-white/80 rounded transition-colors"
                                onClick={(e) => { e.stopPropagation(); generateThumbnail(shot.id, true); }}
                                title="Regenerate Concept Art"
                            >
                                <ImageIcon size={12} />
                            </button>
                            <button
                                className="p-1.5 bg-red-500/10 hover:bg-red-500/20 text-red-400 rounded transition-colors"
                                onClick={(e) => {
                                    e.stopPropagation();
                                    if (confirm('Are you sure you want to remove this video?')) {
                                        patchShot(shot.id, { videoUrl: null });
                                    }
                                }}
                                title="Remove Video"
                            >
                                <Trash2 size={12} />
                            </button>
                        </>
                    )}
                </div>
                <button
                    onClick={handleDelete}
                    className={`p-1.5 rounded transition-colors ${confirmDelete
                        ? 'bg-red-500 hover:bg-red-400 text-white'
                        : 'bg-white/5 hover:bg-red-500/20 text-white/40 hover:text-red-400'
                        }`}
                    title={confirmDelete ? "Click again to confirm" : "Delete shot"}
                >
                    <Trash2 size={12} />
                </button>
            </div>

            {/* Editable fields */}
            <div className="px-3 py-2 space-y-1.5 bg-[#151515]">
                {/* Action */}
                {editingField === 'action' ? (
                    <textarea
                        ref={editRef as React.RefObject<HTMLTextAreaElement>}
                        className="w-full bg-black/50 border border-milimo-500/30 rounded px-2 py-1 text-xs text-white/80 resize-none focus:outline-none"
                        value={editValue || ''}
                        rows={2}
                        onChange={(e) => setEditValue(e.target.value)}
                        onBlur={commitEdit}
                        onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && commitEdit()}
                        onClick={(e) => e.stopPropagation()}
                    />
                ) : (
                    <p
                        className="text-xs text-white/60 line-clamp-2 cursor-text hover:text-white/80 hover:bg-white/5 rounded px-1 py-0.5 transition-colors"
                        onClick={(e) => { e.stopPropagation(); startEdit('action', shot.action || ''); }}
                        title="Click to edit"
                    >
                        {shot.action || <span className="italic text-white/20">Click to add action...</span>}
                    </p>
                )}

                {/* Dialogue */}
                {(shot.dialogue || editingField === 'dialogue') && (
                    editingField === 'dialogue' ? (
                        <textarea
                            ref={editRef as React.RefObject<HTMLTextAreaElement>}
                            className="w-full bg-black/50 border border-milimo-500/30 rounded px-2 py-1 text-xs text-white/80 italic resize-none focus:outline-none"
                            value={editValue || ''}
                            rows={2}
                            onChange={(e) => setEditValue(e.target.value)}
                            onBlur={commitEdit}
                            onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && commitEdit()}
                            onClick={(e) => e.stopPropagation()}
                        />
                    ) : (
                        <p
                            className="text-xs text-white/50 italic line-clamp-2 cursor-text hover:text-white/70 hover:bg-white/5 rounded px-1 py-0.5 transition-colors"
                            onClick={(e) => { e.stopPropagation(); startEdit('dialogue', shot.dialogue || ''); }}
                        >
                            "{shot.dialogue}"
                        </p>
                    )
                )}

                {/* Matched Elements */}
                {shot.matchedElements && shot.matchedElements.length > 0 && (
                    <div className="px-2 py-1.5 border-t border-white/5">
                        <ElementBadgeRow elements={shot.matchedElements} size="sm" maxDisplay={3} />
                    </div>
                )}
            </div>

            {/* Footer Stats */}
            <div className="px-3 py-2 bg-[#111] flex justify-between items-center text-[10px] text-white/30 font-mono mt-auto border-t border-white/5">
                <div className="flex gap-2">
                    <span className="flex items-center gap-1"><Film size={10} /> {shot.numFrames}f</span>
                </div>
                <span className="uppercase tracking-wider">{shot.width}x{shot.height}</span>
            </div>
        </div>
    );
};
