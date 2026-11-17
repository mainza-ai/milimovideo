import { useEffect, useMemo, useRef, useState } from 'react';
import { Command, Search, X } from 'lucide-react';
import { useShallow } from 'zustand/react/shallow';
import { useStudioEditor, type StudioSurface } from '../state/editorStore';
import { useStudioDocumentStore } from '../state/documentStore';

interface CommandItem {
    id: string;
    label: string;
    section: string;
    hint?: string;
    keywords?: string;
    run: () => void;
}

interface Props {
    open: boolean;
    onClose: () => void;
    onOpenShortcuts: () => void;
}

export function CommandPalette({ open, onClose, onOpenShortcuts }: Props) {
    const {
        setSurface, setDensity, density, setInspectorOpen, inspectorOpen,
        setCommandOpen, commandOpen, setSpineCollapsed, spineCollapsed,
    } = useStudioEditor(useShallow((state) => ({
        setSurface: state.setSurface,
        setDensity: state.setDensity,
        density: state.density,
        setInspectorOpen: state.setInspectorOpen,
        inspectorOpen: state.inspectorOpen,
        setCommandOpen: state.setCommandOpen,
        commandOpen: state.commandOpen,
        setSpineCollapsed: state.setSpineCollapsed,
        spineCollapsed: state.spineCollapsed,
    })));
    const saveProject = useStudioDocumentStore((state) => state.saveProject);
    const [query, setQuery] = useState('');
    const [selected, setSelected] = useState(0);
    const inputRef = useRef<HTMLInputElement>(null);

    const commands = useMemo<CommandItem[]>(() => {
        const go = (surface: StudioSurface) => () => setSurface(surface);
        return [
            { id: 'story', label: 'Go to Story', section: 'Navigate', hint: '1', run: go('story') },
            { id: 'generate', label: 'Go to Generate', section: 'Navigate', hint: '2', run: go('generate') },
            { id: 'edit', label: 'Go to Edit', section: 'Navigate', hint: '3', run: go('edit') },
            { id: 'review', label: 'Go to Review', section: 'Navigate', hint: '4', run: go('review') },
            { id: 'library', label: 'Go to Library', section: 'Navigate', hint: '5', run: go('library') },
            { id: 'deliver', label: 'Go to Deliver', section: 'Navigate', hint: '6', run: go('deliver') },
            { id: 'shorts', label: 'Go to Shorts', section: 'Navigate', hint: '7', run: go('shorts') },
            { id: 'setup', label: 'Go to Setup', section: 'Navigate', hint: '8', run: go('setup') },
            { id: 'save', label: 'Save project', section: 'Project', hint: '⌘S', keywords: 'write flush', run: () => void saveProject() },
            {
                id: 'inspector',
                label: inspectorOpen ? 'Hide inspector' : 'Show inspector',
                section: 'Layout',
                run: () => setInspectorOpen(!inspectorOpen),
            },
            {
                id: 'dock',
                label: commandOpen ? 'Hide agent command dock' : 'Show agent command dock',
                section: 'Layout',
                hint: '⌘J',
                run: () => setCommandOpen(!commandOpen),
            },
            {
                id: 'spine',
                label: spineCollapsed ? 'Show story spine' : 'Hide story spine',
                section: 'Layout',
                run: () => setSpineCollapsed(!spineCollapsed),
            },
            {
                id: 'density',
                label: density === 'comfortable' ? 'Use compact density' : 'Use comfortable density',
                section: 'Preferences',
                run: () => setDensity(density === 'comfortable' ? 'compact' : 'comfortable'),
            },
            {
                id: 'shortcuts',
                label: 'Keyboard shortcuts',
                section: 'Help',
                hint: '?',
                run: onOpenShortcuts,
            },
        ];
    }, [
        commandOpen, density, inspectorOpen, onOpenShortcuts, saveProject,
        setCommandOpen, setDensity, setInspectorOpen, setSpineCollapsed, setSurface, spineCollapsed,
    ]);

    const filtered = useMemo(() => {
        const q = query.trim().toLowerCase();
        return q
            ? commands.filter((item) => `${item.label} ${item.section} ${item.keywords || ''}`.toLowerCase().includes(q))
            : commands;
    }, [commands, query]);

    useEffect(() => {
        if (!open) return;
        setQuery('');
        setSelected(0);
        window.setTimeout(() => inputRef.current?.focus(), 20);
    }, [open]);

    useEffect(() => {
        setSelected(0);
    }, [query]);

    if (!open) return null;

    const run = (item: CommandItem | undefined) => {
        if (!item) return;
        item.run();
        onClose();
    };

    return (
        <div className="st-palette-backdrop" role="presentation" onPointerDown={onClose}>
            <div
                className="st-palette"
                role="dialog"
                aria-modal="true"
                aria-label="Command palette"
                onPointerDown={(event) => event.stopPropagation()}
            >
                <div className="st-palette-search">
                    <Search size={16} />
                    <input
                        ref={inputRef}
                        className="st-input"
                        placeholder="Search commands, surfaces, and actions…"
                        aria-label="Search commands"
                        value={query}
                        onChange={(event) => setQuery(event.target.value)}
                        onKeyDown={(event) => {
                            if (event.key === 'ArrowDown') {
                                event.preventDefault();
                                setSelected((current) => Math.min(filtered.length - 1, current + 1));
                            } else if (event.key === 'ArrowUp') {
                                event.preventDefault();
                                setSelected((current) => Math.max(0, current - 1));
                            } else if (event.key === 'Enter') {
                                event.preventDefault();
                                run(filtered[selected]);
                            } else if (event.key === 'Escape') {
                                event.preventDefault();
                                onClose();
                            }
                        }}
                    />
                    <button type="button" className="st-icon-button" aria-label="Close command palette" onClick={onClose}>
                        <X size={15} />
                    </button>
                </div>
                <div className="st-palette-list" role="listbox" aria-label="Commands">
                    {filtered.length === 0 && <div className="st-empty" style={{ minHeight: 80 }}>No matching commands</div>}
                    {filtered.map((item, index) => (
                        <button
                            key={item.id}
                            type="button"
                            role="option"
                            aria-selected={index === selected}
                            className="st-palette-item"
                            data-selected={index === selected}
                            onPointerMove={() => setSelected(index)}
                            onClick={() => run(item)}
                        >
                            <span className="st-palette-section">{item.section}</span>
                            <span>{item.label}</span>
                            {item.hint && <span className="st-mono">{item.hint}</span>}
                        </button>
                    ))}
                </div>
                <div className="st-palette-foot">
                    <span><Command size={12} /> Arrow keys navigate · Enter runs · Esc closes</span>
                    <span>{filtered.length} command(s)</span>
                </div>
            </div>
        </div>
    );
}

const SHORTCUTS: Array<[string, string]> = [
    ['1–8', 'Switch Studio surfaces'],
    ['⌘K / Ctrl+K', 'Command palette'],
    ['⌘J / Ctrl+J', 'Agent command dock'],
    ['⌘S / Ctrl+S', 'Save project'],
    ['⌘Z / Ctrl+Z', 'Undo'],
    ['⇧⌘Z / Ctrl+Shift+Z', 'Redo'],
    ['Space', 'Play / pause'],
    ['B', 'Blade at playhead'],
    ['R', 'Ripple delete'],
    ['[ / ]', 'Slide selected V1 clip'],
    ['Shift-click', 'Multi-select cards'],
    ['Shift-drag', 'Marquee select board'],
    ['Esc', 'Close menus / exit focus'],
    ['?', 'This shortcut sheet'],
];

export function ShortcutSheet({ open, onClose }: { open: boolean; onClose: () => void }) {
    if (!open) return null;
    return (
        <div className="st-palette-backdrop" role="presentation" onPointerDown={onClose}>
            <div
                className="st-shortcuts"
                role="dialog"
                aria-modal="true"
                aria-label="Keyboard shortcuts"
                onPointerDown={(event) => event.stopPropagation()}
            >
                <div className="st-palette-search">
                    <strong style={{ flex: 1 }}>Keyboard shortcuts</strong>
                    <button type="button" className="st-icon-button" aria-label="Close shortcuts" onClick={onClose}>
                        <X size={15} />
                    </button>
                </div>
                <div className="st-shortcut-list">
                    {SHORTCUTS.map(([key, label]) => (
                        <div className="st-shortcut-row" key={key}>
                            <kbd>{key}</kbd>
                            <span>{label}</span>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}
