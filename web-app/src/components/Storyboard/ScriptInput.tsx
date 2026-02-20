import { useState, useCallback, useEffect } from 'react';
import { useTimelineStore, type ParsedScene } from '../../stores/timelineStore';
import { Play, Check, Edit2, Loader2, Plus, Trash2, RotateCcw, Brain } from 'lucide-react';
import { ElementBadgeRow } from './ElementBadge';

const SHOT_TYPE_OPTIONS = [
    { value: 'close_up', label: 'Close Up' },
    { value: 'medium', label: 'Medium' },
    { value: 'wide', label: 'Wide' },
    { value: 'establishing', label: 'Establishing' },
    { value: 'insert', label: 'Insert' },
    { value: 'tracking', label: 'Tracking' },
];

export const ScriptInput = () => {
    const { project, parseScript, aiParseScript, commitStoryboard } = useTimelineStore();
    const [scriptText, setScriptText] = useState('');
    const [parsedScenes, setParsedScenes] = useState<ParsedScene[] | null>(null);
    const [isParsing, setIsParsing] = useState(false);
    const [isAiParsing, setIsAiParsing] = useState(false);
    const [isCommitting, setIsCommitting] = useState(false);

    // Load persisted script
    useEffect(() => {
        if (project?.scriptContent) {
            setScriptText(project.scriptContent);
        }
    }, [project?.scriptContent]);

    const handleParse = async () => {
        if (!scriptText.trim()) return;
        setIsParsing(true);
        const scenes = await parseScript(scriptText);
        setParsedScenes(scenes);
        setIsParsing(false);
    };

    const handleCommit = async () => {
        if (!parsedScenes) return;
        setIsCommitting(true);
        await commitStoryboard(parsedScenes, scriptText);
        setIsCommitting(false);
        setParsedScenes(null);
        // Don't clear scriptText so it persists
    };

    // ─── Editable Preview Mutations ─────────────────────────────

    const updateShotField = useCallback((sceneIdx: number, shotIdx: number, field: string, value: string) => {
        if (!parsedScenes) return;
        const updated = [...parsedScenes];
        const scene = { ...updated[sceneIdx], shots: [...updated[sceneIdx].shots] };
        scene.shots[shotIdx] = { ...scene.shots[shotIdx], [field]: value };
        updated[sceneIdx] = scene;
        setParsedScenes(updated);
    }, [parsedScenes]);

    const addShotToPreview = useCallback((sceneIdx: number) => {
        if (!parsedScenes) return;
        const updated = [...parsedScenes];
        const scene = { ...updated[sceneIdx], shots: [...updated[sceneIdx].shots] };
        scene.shots.push({ action: 'A new cinematic shot...', shot_type: 'medium' });
        updated[sceneIdx] = scene;
        setParsedScenes(updated);
    }, [parsedScenes]);

    const removeShotFromPreview = useCallback((sceneIdx: number, shotIdx: number) => {
        if (!parsedScenes) return;
        const updated = [...parsedScenes];
        const scene = { ...updated[sceneIdx], shots: [...updated[sceneIdx].shots] };
        scene.shots.splice(shotIdx, 1);
        updated[sceneIdx] = scene;
        setParsedScenes(updated);
    }, [parsedScenes]);

    return (
        <div className="bg-[#1a1a1a] rounded-xl border border-white/10 p-6 mb-8">
            <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
                <Edit2 size={18} className="text-milimo-500" />
                Script / Screenplay
            </h3>

            {!parsedScenes ? (
                <div className="space-y-4">
                    <textarea
                        className="w-full h-64 bg-black/50 border border-white/10 rounded-lg p-4 text-white/80 font-mono text-sm focus:outline-none focus:border-milimo-500/50 resize-y"
                        placeholder={"INT. CYBERPUNK ALLEY - NIGHT\n\nThe neon rain falls heavily. HERO (30s) runs down the wet pavement.\n\nHERO\nI can't let them find me."}
                        value={scriptText}
                        onChange={(e) => setScriptText(e.target.value)}
                    />
                    <div className="flex justify-end gap-3">
                        <button
                            onClick={async () => {
                                if (!scriptText.trim()) return;
                                setIsAiParsing(true);
                                const scenes = await aiParseScript(scriptText);
                                setParsedScenes(scenes);
                                setIsAiParsing(false);
                            }}
                            disabled={isAiParsing || isParsing || !scriptText.trim()}
                            className="px-6 py-2 bg-milimo-500 hover:bg-milimo-400 text-black font-semibold rounded-lg flex items-center gap-2 transition-colors disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-milimo-500/20"
                        >
                            {isAiParsing ? <Loader2 size={18} className="animate-spin" /> : <Brain size={18} />}
                            AI Analyze
                        </button>
                        <button
                            onClick={handleParse}
                            disabled={isParsing || isAiParsing || !scriptText.trim()}
                            className="px-6 py-2 bg-milimo-500 hover:bg-milimo-400 text-black font-semibold rounded-lg flex items-center gap-2 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            {isParsing ? <Loader2 size={18} className="animate-spin" /> : <Play size={18} />}
                            Quick Parse
                        </button>
                    </div>
                </div>
            ) : (
                <div className="space-y-6">
                    <div className="flex justify-between items-center mb-2">
                        <h4 className="text-white/60 text-sm font-medium">
                            Preview: {parsedScenes.length} Scenes, {parsedScenes.reduce((acc, s) => acc + s.shots.length, 0)} Shots
                        </h4>
                        <div className="flex items-center gap-3">
                            <button
                                onClick={handleParse}
                                className="text-xs text-white/40 hover:text-white flex items-center gap-1"
                                title="Re-parse from original script"
                            >
                                <RotateCcw size={12} />
                                Re-parse
                            </button>
                            <button
                                onClick={() => setParsedScenes(null)}
                                className="text-xs text-white/40 hover:text-white underline"
                            >
                                Edit Script
                            </button>
                        </div>
                    </div>

                    <div className="max-h-[500px] overflow-y-auto space-y-4 pr-2 custom-scrollbar">
                        {parsedScenes.map((scene, i) => (
                            <div key={i} className="bg-black/30 rounded-lg p-4 border border-white/5">
                                <div className="text-milimo-400 font-bold font-mono text-xs uppercase mb-3 tracking-wider flex items-center justify-between">
                                    <span>{scene.name}</span>
                                    <button
                                        onClick={() => addShotToPreview(i)}
                                        className="flex items-center gap-1 text-white/30 hover:text-milimo-400 transition-colors"
                                        title="Add shot to this scene"
                                    >
                                        <Plus size={12} /> Add Shot
                                    </button>
                                </div>
                                <div className="space-y-3">
                                    {scene.shots.map((shot, j) => (
                                        <div key={j} className="flex gap-3 text-sm text-white/70 pl-2 border-l-2 border-white/10 group/shot">
                                            <span className="text-white/30 font-mono text-[10px] pt-1 shrink-0">SHOT {j + 1}</span>
                                            <div className="flex-1 space-y-1.5">
                                                {/* Character (editable) */}
                                                {shot.character && (
                                                    <input
                                                        className="text-milimo-300 font-bold block text-xs mb-1 bg-transparent border-b border-transparent hover:border-milimo-500/30 focus:border-milimo-500 focus:outline-none w-full"
                                                        value={shot.character}
                                                        onChange={(e) => updateShotField(i, j, 'character', e.target.value)}
                                                    />
                                                )}
                                                {/* Action / Dialogue (editable) */}
                                                <textarea
                                                    className={`w-full bg-transparent border border-transparent hover:border-white/10 focus:border-milimo-500/30 rounded px-1 py-0.5 text-sm resize-none focus:outline-none ${shot.dialogue ? 'italic text-white/90' : 'text-white/70'}`}
                                                    rows={2}
                                                    value={shot.dialogue || shot.action || ''}
                                                    onChange={(e) => {
                                                        const field = shot.dialogue ? 'dialogue' : 'action';
                                                        updateShotField(i, j, field, e.target.value);
                                                    }}
                                                />
                                                {/* Shot Type Selector */}
                                                <div className="flex items-center gap-2">
                                                    <select
                                                        className="text-[10px] bg-black/50 border border-white/10 rounded px-1.5 py-0.5 text-white/50 focus:outline-none focus:border-milimo-500/30 appearance-none cursor-pointer"
                                                        value={shot.shot_type || 'medium'}
                                                        onChange={(e) => updateShotField(i, j, 'shot_type', e.target.value)}
                                                    >
                                                        {SHOT_TYPE_OPTIONS.map(opt => (
                                                            <option key={opt.value} value={opt.value}>{opt.label}</option>
                                                        ))}
                                                    </select>
                                                    {/* Element Match Badges */}
                                                    {shot.matched_elements && shot.matched_elements.length > 0 && (
                                                        <ElementBadgeRow elements={shot.matched_elements} size="sm" />
                                                    )}
                                                </div>
                                            </div>
                                            {/* Delete shot */}
                                            <button
                                                onClick={() => removeShotFromPreview(i, j)}
                                                className="opacity-0 group-hover/shot:opacity-100 p-1 text-white/20 hover:text-red-400 transition-all shrink-0"
                                                title="Remove shot"
                                            >
                                                <Trash2 size={14} />
                                            </button>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        ))}
                    </div>

                    <div className="flex justify-end pt-4 border-t border-white/10">
                        <button
                            onClick={handleCommit}
                            disabled={isCommitting}
                            className="px-6 py-3 bg-white text-black hover:bg-white/90 font-bold rounded-lg flex items-center gap-2 transition-colors disabled:opacity-50"
                        >
                            {isCommitting ? <Loader2 size={18} className="animate-spin" /> : <Check size={18} />}
                            Generate Storyboard
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
};
