import { useState, useEffect } from 'react';
import { useTimelineStore } from '../../stores/timelineStore';
import { useShallow } from 'zustand/react/shallow';
// import type { StoryElement } from '../../stores/timelineStore';
import { Plus, Trash2, Wand2, Image as ImageIcon, X, Edit2 } from 'lucide-react';

interface ElementCardProps {
    el: any;
    generating: boolean;
    project: any;
    onGenerate: (id: string, promptOverride?: string, guidanceOverride?: number, enableAeOverride?: boolean) => void;
    onCancel: (id: string) => void;
    onDelete: (id: string) => void;
    onEdit: (el: any) => void;
    onPreview: (url: string) => void;
}

const ElementCard = ({ el, generating, project, onGenerate, onDelete, onCancel, onEdit, onPreview }: ElementCardProps) => {
    const [guidance, setGuidance] = useState(2.0);
    const [enableAe, setEnableAe] = useState(false);
    const [isHovered, setIsHovered] = useState(false);

    return (
        <div
            tabIndex={0}
            onMouseEnter={() => setIsHovered(true)}
            onMouseLeave={() => setIsHovered(false)}
            className="bg-[#111] border border-white/5 rounded-xl overflow-hidden hover:border-white/20 transition-all relative hover:shadow-lg hover:shadow-milimo-500/10 hover:-translate-y-1 flex flex-col transform-gpu"
            draggable
            onDragStart={(e) => {
                const data = {
                    id: el.id,
                    name: el.name,
                    triggerWord: el.triggerWord,
                    image_path: el.image_path,
                    type: el.type,
                    dragType: 'milimo-element'
                };
                e.dataTransfer.setData('application/json', JSON.stringify(data));
                e.dataTransfer.effectAllowed = 'copy';
            }}
        >
            {/* Visual Header */}
            <div
                className={`aspect-video bg-black/50 relative transition-colors ${isHovered ? 'bg-black/40' : ''}`}
                onClick={(e) => {
                    if (el.image_path) {
                        e.stopPropagation();
                        const src = el.image_path.startsWith('http')
                            ? el.image_path
                            : `http://localhost:8000${el.image_path.startsWith('/') ? '' : '/'}${el.image_path}`;
                        onPreview(src);
                    }
                }}
            >
                {el.image_path ? (
                    <img
                        src={el.image_path.startsWith('http') ? el.image_path : `http://localhost:8000${el.image_path.startsWith('/') ? '' : '/'}${el.image_path}`}
                        alt={el.name}
                        className="w-full h-full object-cover"
                        onError={(e) => {
                            if (el.image_path && !el.image_path.startsWith('http')) {
                                const filename = el.image_path.split('/').pop();
                                const fallbackSrc = `http://localhost:8000/projects/${project?.id}/assets/${filename}`;
                                if ((e.target as HTMLImageElement).src !== fallbackSrc) {
                                    (e.target as HTMLImageElement).src = fallbackSrc;
                                }
                            }
                        }}
                    />
                ) : (
                    <div className="w-full h-full flex flex-col items-center justify-center text-white/20 gap-2">
                        <ImageIcon size={32} />
                        <span className="text-xs font-medium">No Visual</span>
                    </div>
                )}

                {/* Loading Overlay */}
                {generating && (
                    <div className="absolute inset-0 bg-black/80 flex flex-col items-center justify-center z-20 backdrop-blur-sm">
                        <div className="w-6 h-6 border-2 border-milimo-500 border-t-transparent rounded-full animate-spin mb-2" />
                        <span className="text-[10px] text-milimo-500 font-bold uppercase tracking-wider animate-pulse mb-2">Generating...</span>
                        <button
                            onClick={(e) => { e.stopPropagation(); onCancel(el.id); }}
                            className="px-2 py-1 bg-red-500/20 hover:bg-red-500/40 text-red-500 text-[10px] font-bold rounded border border-red-500/50 transition-colors"
                        >
                            CANCEL
                        </button>
                    </div>
                )}

                {/* Overlay Actions */}
                {!generating && (
                    <div className={`absolute inset-0 bg-black/60 transition-opacity flex flex-col items-center justify-center gap-3 backdrop-blur-sm p-4 transform-gpu ${isHovered ? 'opacity-100' : 'opacity-0 pointer-events-none'}`}>
                        <button
                            onClick={(e) => { e.stopPropagation(); onGenerate(el.id, undefined, guidance, enableAe); }}
                            onPointerDown={(e) => { e.stopPropagation(); e.preventDefault(); onGenerate(el.id, undefined, guidance, enableAe); }}
                            className="p-2 px-4 bg-milimo-500 text-black rounded-lg hover:bg-milimo-400 transition-transform hover:scale-105 shadow-xl font-bold flex items-center gap-2 pointer-events-auto"
                            title="Generate Visual (High Quality)"
                        >
                            <Wand2 size={18} /> Generate Visual
                        </button>

                        {/* Quick Guidance Slider */}
                        <div
                            className="bg-black/80 rounded px-2 py-1 flex items-center gap-2 mt-2"
                            onClick={e => e.stopPropagation()}
                        >
                            <span className="text-[10px] text-white/50 uppercase font-bold">Guidance: {guidance.toFixed(1)}</span>
                            <input
                                type="range"
                                min="1.0" max="5.0" step="0.5"
                                value={guidance}
                                className="w-16 accent-milimo-500 h-1 cursor-pointer"
                                onChange={(e) => setGuidance(parseFloat(e.target.value))}
                            />
                        </div>


                        {/* NativeAE Toggle */}
                        <div
                            className="bg-black/80 rounded px-2 py-1 flex items-center gap-2 cursor-pointer border border-white/5 hover:border-white/20 transition-colors"
                            onClick={(e) => { e.stopPropagation(); setEnableAe(!enableAe); }}
                        >
                            <div className={`w-3 h-3 rounded-full ${enableAe ? 'bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.6)]' : 'bg-gray-600'}`} />
                            <span className={`text-[10px] font-bold uppercase transition-colors ${enableAe ? 'text-white' : 'text-white/40'}`}>NativeAE</span>
                        </div>
                    </div>
                )}
            </div>

            {/* Content */}
            <div className="p-5 flex-1">
                <div className="flex items-start justify-between mb-2">
                    <div>
                        <h3 className="text-xl font-bold text-white">{el.name}</h3>
                        <span className="text-xs font-mono text-milimo-500 bg-milimo-500/10 px-2 py-0.5 rounded">{el.triggerWord}</span>
                    </div>
                    <span className="text-[10px] uppercase font-bold tracking-wider text-white/30 border border-white/10 px-2 py-1 rounded-full">
                        {el.type}
                    </span>
                </div>
                <p className="text-sm text-white/60 line-clamp-3 leading-relaxed">
                    {el.description}
                </p>
            </div>

            {/* Footer Actions */}
            <div className={`px-5 pb-5 pt-0 flex justify-between items-center transition-opacity mt-auto transform-gpu ${isHovered ? 'opacity-100' : 'opacity-0 pointer-events-none'}`}>
                <button
                    onClick={(e) => {
                        e.stopPropagation();
                        e.preventDefault();
                        onEdit(el);
                    }}
                    onPointerDown={(e) => {
                        e.stopPropagation();
                        e.preventDefault();
                        onEdit(el);
                    }}
                    className="flex items-center gap-1.5 text-xs font-bold text-white/50 hover:text-white transition-colors p-2 cursor-pointer z-50 relative bg-black/50 rounded-md"
                >
                    <Edit2 size={14} /> Edit Element
                </button>
                <button
                    onClick={(e) => {
                        e.stopPropagation();
                        if (window.confirm(`Delete element "${el.name}"? This cannot be undone.`)) {
                            onDelete(el.id);
                        }
                    }}
                    className="text-red-500/50 hover:text-red-500 transition-colors p-2"
                >
                    <Trash2 size={16} />
                </button>
            </div>
        </div>
    );
};

export const ElementsView = () => {
    const {
        project, elements, fetchElements, createElement, updateElement, deleteElement,
        generateVisual, generatingElementIds, cancelElementGeneration
    } = useTimelineStore(useShallow(state => ({
        project: state.project,
        elements: state.elements,
        fetchElements: state.fetchElements,
        createElement: state.createElement,
        updateElement: state.updateElement,
        deleteElement: state.deleteElement,
        generateVisual: state.generateVisual,
        generatingElementIds: state.generatingElementIds,
        cancelElementGeneration: state.cancelElementGeneration
    })));
    // const [selectedElement, setSelectedElement] = useState<StoryElement | null>(null);
    const [isCreating, setIsCreating] = useState(false);
    const [previewImage, setPreviewImage] = useState<string | null>(null); // For Lightbox

    // Form State
    const [editingId, setEditingId] = useState<string | null>(null);
    const [name, setName] = useState('');
    const [type, setType] = useState('character');
    const [description, setDescription] = useState('');
    const [triggerWord, setTriggerWord] = useState('');

    useEffect(() => {
        if (project?.id) {
            fetchElements(project.id);
        }
    }, [project?.id]);

    const handleCreate = async () => {
        if (!project) return;
        if (editingId) {
            // Update existing element
            await updateElement(editingId, { name, type: type as any, description, triggerWord });
        } else {
            // Create new element
            await createElement(project.id, { name, type: type as any, description, triggerWord });
        }
        setIsCreating(false);
        setEditingId(null);
        resetForm();
    };

    const startEdit = (el: any) => {
        setName(el.name);
        setType(el.type);
        setDescription(el.description || '');
        setTriggerWord(el.triggerWord);
        setEditingId(el.id);
        setIsCreating(true);
    };

    const resetForm = () => {
        setName('');
        setDescription('');
        setTriggerWord('');
        setType('character');
        setEditingId(null);
    };



    return (
        <div className="absolute inset-0 bg-[#050505] p-8 overflow-y-auto">
            <div className="max-w-6xl mx-auto">
                {/* Header */}
                <div className="flex items-center justify-between mb-8">
                    <div>
                        <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-white/50">
                            Story Elements
                        </h1>
                        <p className="text-white/40 mt-1">Manage characters, locations, and props for consistent generation.</p>
                    </div>
                    <button
                        onClick={() => { resetForm(); setIsCreating(true); }}
                        className="px-4 py-2 bg-milimo-500 hover:bg-milimo-400 text-black font-bold rounded-lg flex items-center gap-2 transition-all shadow-lg shadow-milimo-500/20"
                    >
                        <Plus size={18} />
                        New Element
                    </button>
                </div>

                {/* Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {/* Create Card */}
                    {isCreating && (
                        <div className="bg-[#111] border border-milimo-500/50 rounded-xl p-6 relative group ring-2 ring-milimo-500/20">
                            <button
                                onClick={() => { setIsCreating(false); resetForm(); }}
                                className="absolute top-4 right-4 text-white/20 hover:text-white"
                            >
                                <X size={18} />
                            </button>
                            <h3 className="text-lg font-bold mb-4 flex items-center gap-2 text-milimo-400">
                                {editingId ? <><Edit2 size={18} /> Edit Element</> : <><Plus size={18} /> Create New Element</>}
                            </h3>
                            <div className="space-y-4">
                                <div>
                                    <label className="text-xs text-white/40 uppercase font-bold tracking-wider">Name</label>
                                    <input
                                        value={name} onChange={e => setName(e.target.value)}
                                        className="w-full bg-black/20 border border-white/10 rounded px-3 py-2 mt-1 text-white focus:border-milimo-500 outline-none"
                                        placeholder="e.g. Kael"
                                    />
                                </div>
                                <div className="grid grid-cols-2 gap-4">
                                    <div>
                                        <label className="text-xs text-white/40 uppercase font-bold tracking-wider">Type</label>
                                        <select
                                            value={type} onChange={e => setType(e.target.value)}
                                            className="w-full bg-black/20 border border-white/10 rounded px-3 py-2 mt-1 text-white outline-none appearance-none"
                                        >
                                            <option value="character">Character</option>
                                            <option value="location">Location</option>
                                            <option value="object">Object</option>
                                        </select>
                                    </div>
                                    <div>
                                        <label className="text-xs text-white/40 uppercase font-bold tracking-wider">Trigger</label>
                                        <input
                                            value={triggerWord} onChange={e => setTriggerWord(e.target.value)}
                                            className="w-full bg-black/20 border border-white/10 rounded px-3 py-2 mt-1 text-white/70 focus:border-milimo-500 outline-none"
                                            placeholder="@Name"
                                        />
                                    </div>
                                </div>
                                <div>
                                    <label className="text-xs text-white/40 uppercase font-bold tracking-wider">Visual Description</label>
                                    <textarea
                                        value={description} onChange={e => setDescription(e.target.value)}
                                        className="w-full bg-black/20 border border-white/10 rounded px-3 py-2 mt-1 text-white/70 focus:border-milimo-500 outline-none min-h-[100px]"
                                        placeholder="Detailed visual description..."
                                    />
                                </div>
                                <button
                                    onClick={handleCreate}
                                    className={`w-full py-2 text-black font-bold rounded transition-colors ${editingId ? 'bg-blue-400 hover:bg-blue-300' : 'bg-milimo-500 hover:bg-milimo-400'}`}
                                >
                                    {editingId ? 'Save Changes' : 'Create Element'}
                                </button>
                            </div>
                        </div>
                    )}

                    {/* Element Cards */}
                    {elements.map((el) => (
                        <ElementCard
                            key={el.id}
                            el={el}
                            generating={!!generatingElementIds[el.id]}
                            project={project}
                            onGenerate={generateVisual}
                            onDelete={deleteElement}
                            onCancel={cancelElementGeneration}
                            onEdit={startEdit}
                            onPreview={setPreviewImage}
                        />
                    ))}
                </div>
            </div>

            {/* Lightbox Preview */}
            {previewImage && (
                <div
                    className="fixed inset-0 z-[2000] bg-black/95 flex items-center justify-center p-4 cursor-zoom-out animate-in fade-in duration-200"
                    onClick={() => setPreviewImage(null)}
                >
                    <img
                        src={previewImage}
                        className="max-w-full max-h-[90vh] object-contain rounded-lg shadow-2xl ring-1 ring-white/10"
                        alt="Preview"
                    />
                    <button
                        className="absolute top-4 right-4 p-2 bg-white/10 hover:bg-white/20 text-white rounded-full transition-colors z-[2001]"
                        onClick={() => setPreviewImage(null)}
                    >
                        <X size={24} />
                    </button>
                </div>
            )}
        </div>
    );
};
