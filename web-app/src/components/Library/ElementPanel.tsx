import { useEffect, useState } from 'react';
import { useTimelineStore } from '../../stores/timelineStore';
import { Search, Plus, Trash2, User, MapPin, Box } from 'lucide-react';

export const ElementPanel = () => {
    const { project, fetchElements, elements = [], createElement, deleteElement } = useTimelineStore();
    const [search, setSearch] = useState("");
    const [isCreating, setIsCreating] = useState(false);

    // Form State
    const [newName, setNewName] = useState("");
    const [newType, setNewType] = useState("character");
    const [newDesc, setNewDesc] = useState("");
    const [newTrigger, setNewTrigger] = useState("");

    useEffect(() => {
        if (project?.id) {
            // Fetch elements when project loads
            fetchElements(project.id);
        }
    }, [project?.id]);

    const handleCreate = async () => {
        if (!newName || !newDesc) return;

        // Ensure type safety
        const typeValue = newType as 'character' | 'location' | 'object';

        await createElement(project.id, {
            name: newName,
            type: typeValue,
            description: newDesc,
            triggerWord: newTrigger || `@${newName.replace(/\s+/g, '')}`
        });
        setIsCreating(false);
        setNewName("");
        setNewDesc("");
        setNewTrigger("");
    };

    const filteredElements = elements.filter((el: any) =>
        el.name.toLowerCase().includes(search.toLowerCase()) ||
        el.triggerWord.toLowerCase().includes(search.toLowerCase())
    );

    return (
        <div className="flex flex-col h-full bg-[#0a0a0a] text-white border-r border-white/5 w-80">
            {/* Header */}
            <div className="p-4 border-b border-white/5">
                <h3 className="font-bold text-sm tracking-wide mb-4">Story Elements</h3>

                {/* Search */}
                <div className="relative mb-4">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-white/30" size={14} />
                    <input
                        className="w-full bg-white/5 border border-white/10 rounded-lg pl-9 pr-3 py-2 text-xs focus:outline-none focus:border-milimo-500/50 transition-colors"
                        placeholder="Search elements..."
                        value={search}
                        onChange={(e) => setSearch(e.target.value)}
                    />
                </div>

                {/* Create Button */}
                <button
                    onClick={() => setIsCreating(true)}
                    className="w-full py-2 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg text-xs font-medium flex items-center justify-center gap-2 transition-colors"
                >
                    <Plus size={14} /> New Element
                </button>
            </div>

            {/* Creation Form */}
            {isCreating && (
                <div className="p-4 border-b border-milimo-500/20 bg-milimo-500/5">
                    <div className="space-y-3">
                        <input
                            className="w-full bg-black/40 border border-white/10 rounded px-2 py-1.5 text-xs focus:border-milimo-500/50 outline-none"
                            placeholder="Name (e.g. Hero)"
                            value={newName}
                            onChange={(e) => setNewName(e.target.value)}
                            autoFocus
                        />
                        <select
                            className="w-full bg-black/40 border border-white/10 rounded px-2 py-1.5 text-xs outline-none"
                            value={newType}
                            onChange={(e) => setNewType(e.target.value as 'character' | 'location' | 'object')}
                        >
                            <option value="character">Character</option>
                            <option value="location">Location</option>
                            <option value="object">Object</option>
                        </select>
                        <input
                            className="w-full bg-black/40 border border-white/10 rounded px-2 py-1.5 text-xs focus:border-milimo-500/50 outline-none font-mono text-milimo-400"
                            placeholder="Trigger (e.g. @Hero)"
                            value={newTrigger}
                            onChange={(e) => setNewTrigger(e.target.value)}
                        />
                        <textarea
                            className="w-full bg-black/40 border border-white/10 rounded px-2 py-1.5 text-xs focus:border-milimo-500/50 outline-none min-h-[60px]"
                            placeholder="Description (Visual appearance...)"
                            value={newDesc}
                            onChange={(e) => setNewDesc(e.target.value)}
                        />
                        <div className="flex gap-2 pt-1">
                            <button onClick={() => setIsCreating(false)} className="flex-1 py-1.5 text-xs opacity-50 hover:opacity-100">Cancel</button>
                            <button onClick={handleCreate} className="flex-1 py-1.5 bg-milimo-500 text-black text-xs font-bold rounded">Create</button>
                        </div>
                    </div>
                </div>
            )}

            {/* List */}
            <div className="flex-1 overflow-y-auto custom-scrollbar p-2 space-y-2">
                {filteredElements.map((el: any) => (
                    <div key={el.id} className="group p-3 bg-white/5 hover:bg-white/10 rounded-lg border border-transparent hover:border-white/10 transition-all">
                        <div className="flex justify-between items-start mb-1">
                            <div className="flex items-center gap-2">
                                {el.type === 'character' && <User size={12} className="text-milimo-400" />}
                                {el.type === 'location' && <MapPin size={12} className="text-blue-400" />}
                                {el.type === 'object' && <Box size={12} className="text-orange-400" />}
                                <span className="text-sm font-bold">{el.name}</span>
                            </div>
                            <button
                                onClick={() => {
                                    if (window.confirm(`Delete element "${el.name}"? This cannot be undone.`)) {
                                        deleteElement(el.id);
                                    }
                                }}
                                className="text-white/20 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-all"
                            >
                                <Trash2 size={12} />
                            </button>
                        </div>
                        <div className="text-[10px] font-mono text-white/40 mb-2">{el.triggerWord}</div>
                        <p className="text-xs text-white/60 line-clamp-2 leading-relaxed">
                            {el.description}
                        </p>
                    </div>
                ))}

                {filteredElements.length === 0 && !isCreating && (
                    <div className="text-center py-10 text-white/20 text-xs">
                        No elements found.
                    </div>
                )}
            </div>
        </div>
    );
};
