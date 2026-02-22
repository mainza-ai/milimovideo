import { useState, useEffect } from 'react';

import { Users, MapPin, Box, Plus, Trash2, X } from 'lucide-react';

// Types (Mocking backend type for now)
interface Element {
    id: string;
    name: string;
    type: 'character' | 'location' | 'object';
    description: string;
    trigger_word: string;
    image_path?: string;
}

export const ElementManager = () => {
    const [elements, setElements] = useState<Element[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [showAddForm, setShowAddForm] = useState(false);
    const [hoveredElementId, setHoveredElementId] = useState<string | null>(null);

    // Form State
    const [newName, setNewName] = useState('');
    const [newType, setNewType] = useState<'character' | 'location' | 'object'>('character');
    const [newDesc, setNewDesc] = useState('');


    const projectId = 'default_project'; // In real app, get from project.id (if persistent)
    // Note: Our project store mock currently doesn't have a real DB ID unless we sync it. 
    // We'll assume project.id is valid or use a default for dev.

    useEffect(() => {
        fetchElements();
    }, [projectId]);

    const fetchElements = async () => {
        setIsLoading(true);
        try {
            const res = await fetch(`http://localhost:8000/projects/${projectId}/elements`);
            if (res.ok) {
                const data = await res.json();
                setElements(data);
            }
        } catch (e) {
            console.error("Failed to fetch elements:", e);
        } finally {
            setIsLoading(false);
        }
    };

    const handleCreate = async () => {
        if (!newName || !newDesc) return;

        try {
            const res = await fetch(`http://localhost:8000/projects/${projectId}/elements`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    name: newName,
                    type: newType,
                    description: newDesc,
                    trigger_word: `@${newName.replace(/\s+/g, '')}`
                })
            });

            if (res.ok) {
                fetchElements();
                setShowAddForm(false);
                setNewName('');
                setNewDesc('');
            }
        } catch (e) {
            console.error("Failed to create element:", e);
        }
    };

    const handleDelete = async (id: string) => {
        try {
            const res = await fetch(`http://localhost:8000/elements/${id}`, { method: 'DELETE' });
            if (res.ok) {
                setElements(prev => prev.filter(e => e.id !== id));
            }
        } catch (e) {
            console.error("Failed to delete element:", e);
        }
    };

    const getIcon = (type: string) => {
        switch (type) {
            case 'character': return <Users size={14} />;
            case 'location': return <MapPin size={14} />;
            default: return <Box size={14} />;
        }
    };

    return (
        <div className="flex flex-col h-full bg-[#0a0a0a] border-l border-white/5">
            <div className="p-4 border-b border-white/5 flex justify-between items-center">
                <h3 className="font-bold text-sm tracking-wider text-white/80 uppercase">Story Elements</h3>
                <button
                    onClick={() => setShowAddForm(true)}
                    className="p-1.5 hover:bg-white/10 rounded-md transition-colors text-milimo-400"
                >
                    <Plus size={16} />
                </button>
            </div>

            {/* Content */}
            <div className="flex-1 overflow-y-auto p-4 space-y-3">

                {showAddForm && (
                    <div className="bg-white/5 p-3 rounded-lg border border-white/10 space-y-3 animate-in fade-in slide-in-from-top-2">
                        <div className="flex justify-between items-center mb-2">
                            <span className="text-xs font-bold text-white/60">NEW ELEMENT</span>
                            <button onClick={() => setShowAddForm(false)} className="text-white/30 hover:text-white"><X size={14} /></button>
                        </div>

                        <input
                            placeholder="Name (e.g. Hero)"
                            className="w-full bg-black/50 border border-white/10 rounded px-2 py-1.5 text-xs text-white focus:outline-none focus:border-milimo-500"
                            value={newName}
                            onChange={e => setNewName(e.target.value)}
                        />

                        <select
                            className="w-full bg-black/50 border border-white/10 rounded px-2 py-1.5 text-xs text-white focus:outline-none"
                            value={newType}
                            onChange={e => setNewType(e.target.value as any)}
                        >
                            <option value="character">Character</option>
                            <option value="location">Location</option>
                            <option value="object">Object</option>
                        </select>

                        <textarea
                            placeholder="Visual Description..."
                            className="w-full bg-black/50 border border-white/10 rounded px-2 py-1.5 text-xs text-white focus:outline-none focus:border-milimo-500 h-20"
                            value={newDesc}
                            onChange={e => setNewDesc(e.target.value)}
                        />

                        <button
                            onClick={handleCreate}
                            disabled={!newName || !newDesc}
                            className="w-full bg-milimo-500 hover:bg-milimo-400 text-black font-bold py-1.5 rounded text-xs transition-colors disabled:opacity-50"
                        >
                            Create Element
                        </button>
                    </div>
                )}

                {elements.map(el => (
                    <div
                        key={el.id}
                        className="bg-white/5 hover:bg-white/10 border border-white/5 hover:border-white/20 rounded-lg p-3 transition-all"
                        onMouseEnter={() => setHoveredElementId(el.id)}
                        onMouseLeave={() => setHoveredElementId(null)}
                    >
                        <div className="flex justify-between items-start">
                            <div className="flex items-center gap-2 mb-1">
                                <span className={`p-1 rounded bg-black/50 ${el.type === 'character' ? 'text-blue-400' : el.type === 'location' ? 'text-green-400' : 'text-yellow-400'}`}>
                                    {getIcon(el.type)}
                                </span>
                                <span className="font-bold text-sm text-white/90">{el.name}</span>
                            </div>
                            <button
                                onClick={() => handleDelete(el.id)}
                                className={`text-white/20 hover:text-red-400 transition-opacity ${hoveredElementId === el.id ? 'opacity-100' : 'opacity-0 pointer-events-none'}`}
                            >
                                <Trash2 size={12} />
                            </button>
                        </div>

                        <div className="text-[10px] font-mono text-milimo-400/80 mb-2">{el.trigger_word}</div>
                        <p className="text-xs text-white/50 line-clamp-2">{el.description}</p>
                    </div>
                ))}

                {elements.length === 0 && !isLoading && !showAddForm && (
                    <div className="text-center py-8 text-white/20 text-xs">
                        No elements yet.<br />Click + to add characters or locations.
                    </div>
                )}
            </div>
        </div>
    );
};
