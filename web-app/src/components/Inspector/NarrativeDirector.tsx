import { clsx } from 'clsx';
import { ChevronDown, ChevronRight } from 'lucide-react';
import { useState } from 'react';
import { type Shot } from '../../stores/timelineStore';

interface NarrativeDirectorProps {
    shot: Shot;
}

export const NarrativeDirector = ({ shot }: NarrativeDirectorProps) => {
    const [isCollapsed, setIsCollapsed] = useState(true);

    if (!shot.isGenerating && !shot.currentPrompt) return null;

    return (
        <div className="space-y-2">
            <button
                className="flex items-center gap-2 w-full group focus:outline-none"
                onClick={() => setIsCollapsed(!isCollapsed)}
            >
                {isCollapsed ? <ChevronRight size={12} className="text-white/40 group-hover:text-white" /> : <ChevronDown size={12} className="text-white/40 group-hover:text-white" />}
                <label className="text-[10px] uppercase tracking-widest text-milimo-400 font-bold flex items-center gap-2 cursor-pointer group-hover:text-milimo-300">
                    {shot.isGenerating && <span className="animate-pulse">🔴</span>}
                    <span>Narrative Director</span>
                </label>
            </button>

            {!isCollapsed && (
                <div className={clsx(
                    "w-full bg-milimo-900/20 border border-milimo-500/30 rounded-lg p-3 text-xs text-milimo-300 italic",
                    shot.isGenerating && "animate-pulse"
                )}>
                    {shot.currentPrompt}
                </div>
            )}
        </div>
    );
};
