import { BookOpen, Clapperboard, FlaskConical, Library, PlaySquare, Send, Settings, Sparkles } from 'lucide-react';
import { useStudioEditor, type StudioSurface } from '../state/editorStore';

const MODES: Array<{ id: StudioSurface; label: string; icon: typeof BookOpen }> = [
    { id: 'story', label: 'Story', icon: BookOpen },
    { id: 'generate', label: 'Generate', icon: Sparkles },
    { id: 'edit', label: 'Edit', icon: Clapperboard },
    { id: 'review', label: 'Review', icon: PlaySquare },
    { id: 'library', label: 'Library', icon: Library },
    { id: 'deliver', label: 'Deliver', icon: Send },
];

export function ModeRail() {
    const activeSurface = useStudioEditor((state) => state.activeSurface);
    const setSurface = useStudioEditor((state) => state.setSurface);

    return (
        <aside className="st-rail" aria-label="Studio modes">
            {MODES.map((mode) => {
                const Icon = mode.icon;
                return (
                    <button
                        key={mode.id}
                        type="button"
                        className="st-rail-button"
                        data-active={activeSurface === mode.id}
                        onClick={() => setSurface(mode.id)}
                        aria-current={activeSurface === mode.id ? 'page' : undefined}
                        title={mode.label}
                    >
                        <Icon size={17} strokeWidth={1.8} />
                        <span>{mode.label}</span>
                    </button>
                );
            })}
            <div className="st-rail-spacer" />
            <button
                type="button"
                className="st-rail-button"
                onClick={() => window.location.assign('/')}
                title="Legacy editor"
            >
                <FlaskConical size={16} strokeWidth={1.8} />
                <span>Legacy</span>
            </button>
            <button type="button" className="st-rail-button" title="Settings (coming in F0.5)">
                <Settings size={16} strokeWidth={1.8} />
                <span>Setup</span>
            </button>
        </aside>
    );
}
