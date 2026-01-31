import { useState } from 'react';
import { useTimelineStore, type ParsedScene } from '../../stores/timelineStore';
import { Play, Check, Edit2, Loader2 } from 'lucide-react';

export const ScriptInput = () => {
    const { parseScript, commitStoryboard } = useTimelineStore();
    const [scriptText, setScriptText] = useState('');
    const [parsedScenes, setParsedScenes] = useState<ParsedScene[] | null>(null);
    const [isParsing, setIsParsing] = useState(false);
    const [isCommitting, setIsCommitting] = useState(false);

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
        await commitStoryboard(parsedScenes);
        setIsCommitting(false);
        setParsedScenes(null); // Reset after commit
        setScriptText('');
    };

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
                        placeholder="INT. CYBERPUNK ALLEY - NIGHT&#10;&#10;The neon rain falls heavily. HERO (30s) runs down the wet pavement.&#10;&#10;HERO&#10;I can't let them find me."
                        value={scriptText}
                        onChange={(e) => setScriptText(e.target.value)}
                    />
                    <div className="flex justify-end">
                        <button
                            onClick={handleParse}
                            disabled={isParsing || !scriptText.trim()}
                            className="px-6 py-2 bg-milimo-500 hover:bg-milimo-400 text-black font-semibold rounded-lg flex items-center gap-2 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            {isParsing ? <Loader2 size={18} className="animate-spin" /> : <Play size={18} />}
                            Analyze Script
                        </button>
                    </div>
                </div>
            ) : (
                <div className="space-y-6">
                    <div className="flex justify-between items-center mb-2">
                        <h4 className="text-white/60 text-sm font-medium">Preview: {parsedScenes.length} Scenes Found</h4>
                        <button
                            onClick={() => setParsedScenes(null)}
                            className="text-xs text-white/40 hover:text-white underline"
                        >
                            Edit Script
                        </button>
                    </div>

                    <div className="max-h-[400px] overflow-y-auto space-y-4 pr-2 custom-scrollbar">
                        {parsedScenes.map((scene, i) => (
                            <div key={i} className="bg-black/30 rounded-lg p-4 border border-white/5">
                                <div className="text-milimo-400 font-bold font-mono text-xs uppercase mb-2 tracking-wider">
                                    {scene.header}
                                </div>
                                <div className="space-y-2">
                                    {scene.shots.map((shot, j) => (
                                        <div key={j} className="flex gap-3 text-sm text-white/70 pl-2 border-l-2 border-white/10">
                                            <span className="text-white/30 font-mono text-[10px] pt-1">SHOT {j + 1}</span>
                                            <div>
                                                {shot.character && (
                                                    <span className="text-milimo-300 font-bold block text-xs mb-1">{shot.character}</span>
                                                )}
                                                <p className={shot.dialogue ? "italic text-white/90" : ""}>
                                                    {shot.dialogue || shot.action}
                                                </p>
                                            </div>
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
