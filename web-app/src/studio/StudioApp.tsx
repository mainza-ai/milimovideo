import { useEffect } from 'react';
import './studio.css';
import { ModeRail } from './components/ModeRail';
import { TopBar } from './components/TopBar';
import { StatusStrip } from './components/StatusStrip';
import { CommandDock } from './components/CommandDock';
import { Inspector } from './components/Inspector';
import { StorySurface } from './surfaces/StorySurface';
import { GenerateSurface } from './surfaces/GenerateSurface';
import { EditSurface } from './surfaces/EditSurface';
import { ReviewSurface } from './surfaces/ReviewSurface';
import { LibrarySurface } from './surfaces/LibrarySurface';
import { DeliverSurface } from './surfaces/DeliverSurface';
import { SetupSurface } from './surfaces/SetupSurface';
import { useStudioEditor, type StudioSurface } from './state/editorStore';
import { useStudioProject, useStudioBoardSync } from './data/useStudioData';
import { useTimelineStore } from '../stores/timelineStore';
import { SharedReviewPage } from './SharedReviewPage';

const SURFACES: StudioSurface[] = ['story', 'generate', 'edit', 'review', 'library', 'deliver', 'setup'];

export function StudioApp() {
    const shareMatch = typeof window !== 'undefined'
        ? window.location.pathname.match(/^\/studio\/share\/([^/]+)/)
        : null;
    if (shareMatch) {
        return <SharedReviewPage token={decodeURIComponent(shareMatch[1])} />;
    }
    return <StudioShell />;
}

function StudioShell() {
    const { activeSurface, setSurface, inspectorOpen, focusMode, commandOpen, setCommandOpen, setFocusMode } = useStudioEditor();
    const project = useStudioProject();
    useStudioBoardSync(project.id);
    const isPlaying = useTimelineStore((state) => state.isPlaying);
    const setIsPlaying = useTimelineStore((state) => state.setIsPlaying);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const temporalStore = (useTimelineStore as any).temporal;

    useEffect(() => {
        const onKeyDown = (event: KeyboardEvent) => {
            const target = event.target as HTMLElement;
            const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(target?.tagName || '') || target?.isContentEditable;
            if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
                event.preventDefault();
                setCommandOpen(!commandOpen);
                return;
            }
            if (typing) return;
            if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'z') {
                event.preventDefault();
                if (event.shiftKey) temporalStore.getState().redo();
                else temporalStore.getState().undo();
                return;
            }
            if (event.key === 'Escape') {
                setFocusMode('none');
                return;
            }
            if (event.code === 'Space') {
                event.preventDefault();
                if (activeSurface !== 'edit') setSurface('edit');
                setIsPlaying(!isPlaying);
                return;
            }
            const index = Number(event.key);
            if (index >= 1 && index <= SURFACES.length && !event.metaKey && !event.ctrlKey && !event.altKey) {
                setSurface(SURFACES[index - 1]);
            }
        };
        window.addEventListener('keydown', onKeyDown);
        return () => window.removeEventListener('keydown', onKeyDown);
    }, [activeSurface, commandOpen, isPlaying, setCommandOpen, setFocusMode, setIsPlaying, setSurface, temporalStore]);

    const surface = (() => {
        switch (activeSurface) {
            case 'story': return <StorySurface />;
            case 'generate': return <GenerateSurface />;
            case 'edit': return <EditSurface />;
            case 'review': return <ReviewSurface />;
            case 'library': return <LibrarySurface />;
            case 'deliver': return <DeliverSurface />;
            case 'setup': return <SetupSurface />;
            default: return <StorySurface />;
        }
    })();

    return (
        <div className="studio-root">
            <TopBar />
            <div className="st-body">
                <ModeRail />
                <main className="st-main">
                    <div className="st-surface" data-focus={focusMode}>
                        {surface}
                    </div>
                    <CommandDock />
                </main>
                {inspectorOpen && <Inspector />}
            </div>
            <StatusStrip />
        </div>
    );
}
