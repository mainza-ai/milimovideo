import { useEffect, useState } from 'react';
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
import { ShortsSurface } from './surfaces/ShortsSurface';
import { useStudioEditor, type StudioSurface } from './state/editorStore';
import { useStudioProject, useStudioActiveJobs, useStudioBoardSync, useStudioProjectBootstrap } from './data/useStudioData';
import { useStudioDocumentStore } from './state/documentStore';
import { SharedReviewPage } from './SharedReviewPage';
import { studioQueryClient } from './state/queryClient';
import { QueryClientProvider } from '@tanstack/react-query';
import { useShallow } from 'zustand/react/shallow';
import { CommandPalette, ShortcutSheet } from './components/CommandPalette';
import { MediaPreview } from './components/MediaPreview';
import { StudioToastLayer } from './components/StudioToastLayer';
import { ConfirmDialog } from './components/ConfirmDialog';
import { enabledStudioSurfaces, isStudioSurfaceEnabled } from './features';

const SURFACES: StudioSurface[] = enabledStudioSurfaces();

export function StudioApp() {
    const shareMatch = typeof window !== 'undefined'
        ? window.location.pathname.match(/^\/studio\/share\/([^/]+)/)
        : null;
    if (shareMatch) {
        return <SharedReviewPage token={decodeURIComponent(shareMatch[1])} />;
    }
    return (
        <QueryClientProvider client={studioQueryClient}>
            <StudioShell />
        </QueryClientProvider>
    );
}

function StudioShell() {
    const {
        activeSurface, setSurface, inspectorOpen, focusMode, commandOpen, setCommandOpen, setFocusMode, density,
    } = useStudioEditor(useShallow((state) => ({
        activeSurface: state.activeSurface,
        setSurface: state.setSurface,
        inspectorOpen: state.inspectorOpen,
        focusMode: state.focusMode,
        commandOpen: state.commandOpen,
        setCommandOpen: state.setCommandOpen,
        setFocusMode: state.setFocusMode,
        density: state.density,
    })));

    useEffect(() => {
        const closeNarrowInspector = () => {
            // Transient close only: keep the per-surface rail preference.
            if (window.innerWidth < 1180) useStudioEditor.setState({ inspectorOpen: false });
        };
        closeNarrowInspector();
        window.addEventListener('resize', closeNarrowInspector);
        return () => window.removeEventListener('resize', closeNarrowInspector);
    }, []);
    const [paletteOpen, setPaletteOpen] = useState(false);
    const [shortcutsOpen, setShortcutsOpen] = useState(false);
    const project = useStudioProject();
    useStudioProjectBootstrap();
    useStudioActiveJobs(project.id);
    useStudioBoardSync(project.id);
    const isPlaying = useStudioDocumentStore((state) => state.isPlaying);
    const setIsPlaying = useStudioDocumentStore((state) => state.setIsPlaying);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const temporalStore = (useStudioDocumentStore as any).temporal;

    useEffect(() => {
        if (!isStudioSurfaceEnabled(activeSurface) && SURFACES.length > 0) {
            setSurface(SURFACES[0]);
        }
    }, [activeSurface, setSurface]);

    useEffect(() => {
        const onKeyDown = (event: KeyboardEvent) => {
            const target = event.target as HTMLElement;
            const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(target?.tagName || '') || target?.isContentEditable;
            if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
                event.preventDefault();
                setPaletteOpen((open) => !open);
                setShortcutsOpen(false);
                return;
            }
            if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'j') {
                event.preventDefault();
                setCommandOpen(!commandOpen);
                return;
            }
            if (event.key === 'Escape' && (paletteOpen || shortcutsOpen)) {
                event.preventDefault();
                setPaletteOpen(false);
                setShortcutsOpen(false);
                return;
            }
            if (typing) return;
            if (event.key === '?') {
                event.preventDefault();
                setShortcutsOpen(true);
                return;
            }
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
    }, [
        activeSurface, commandOpen, isPlaying, paletteOpen, shortcutsOpen,
        setCommandOpen, setFocusMode, setIsPlaying, setSurface, temporalStore,
    ]);

    if (SURFACES.length === 0) {
        return (
            <div className="studio-root">
                <main className="st-main" style={{ display: 'grid', placeItems: 'center' }}>
                    <div className="st-empty">
                        <strong>Studio surfaces are disabled</strong>
                        <span>Enable a surface with VITE_STUDIO_DISABLED_SURFACES, or use the legacy editor at /.</span>
                    </div>
                </main>
            </div>
        );
    }

    const surface = (() => {
        switch (activeSurface) {
            case 'story': return <StorySurface />;
            case 'generate': return <GenerateSurface />;
            case 'edit': return <EditSurface />;
            case 'review': return <ReviewSurface />;
            case 'library': return <LibrarySurface />;
            case 'deliver': return <DeliverSurface />;
            case 'shorts': return <ShortsSurface />;
            case 'setup': return <SetupSurface />;
            default: return <StorySurface />;
        }
    })();

    return (
        <div className="studio-root" data-density={density}>
            <a className="st-skip-link" href="#studio-main">Skip to workspace</a>
            <TopBar />
            <div className="st-body">
                <ModeRail />
                <main className="st-main" id="studio-main" tabIndex={-1}>
                    <div className="st-surface" data-focus={focusMode}>
                        {surface}
                    </div>
                    <CommandDock />
                </main>
                {inspectorOpen && <Inspector />}
            </div>
            <StatusStrip />
            <StudioToastLayer />
            <ConfirmDialog />
            <MediaPreview />
            <CommandPalette
                open={paletteOpen}
                onClose={() => setPaletteOpen(false)}
                onOpenShortcuts={() => {
                    setPaletteOpen(false);
                    setShortcutsOpen(true);
                }}
            />
            <ShortcutSheet open={shortcutsOpen} onClose={() => setShortcutsOpen(false)} />
        </div>
    );
}
