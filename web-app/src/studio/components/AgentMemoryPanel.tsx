import { useCallback, useEffect, useState } from 'react';
import { Brain, Download, Loader2, Pin, PinOff, Plus, Search, ShieldAlert, Trash2 } from 'lucide-react';
import { apiUrl } from '../../config';
import { useStudioDocumentStore } from '../state/documentStore';
import { requestConfirm } from '../state/confirmStore';

interface MemoryEntry {
    id: string;
    scope: string;
    shot_id?: string | null;
    kind: string;
    text: string;
    metadata: Record<string, unknown>;
    provenance: string;
    trust: string;
    pinned: boolean;
    quarantined: boolean;
    use_count: number;
    tokens: number;
    updated_at?: string | null;
    why: string[];
}

const KINDS = ['semantic', 'episodic', 'procedural', 'working'] as const;

/**
 * Project-scoped agent memory. Entries are local SQLite rows with provenance and
 * trust; quarantined rows (agent/imported text that looked like an instruction
 * injection) stay visible but never enter a context pack until approved.
 */
export function AgentMemoryPanel() {
    const projectId = useStudioDocumentStore((state) => state.project.id);
    const addToast = useStudioDocumentStore((state) => state.addToast);
    const [entries, setEntries] = useState<MemoryEntry[]>([]);
    const [query, setQuery] = useState('');
    const [draft, setDraft] = useState('');
    const [kind, setKind] = useState<(typeof KINDS)[number]>('semantic');
    const [pinned, setPinned] = useState(false);
    const [includeQuarantined, setIncludeQuarantined] = useState(true);
    const [loading, setLoading] = useState(false);
    const [busy, setBusy] = useState(false);

    const load = useCallback(async () => {
        if (!projectId) return;
        setLoading(true);
        try {
            const params = new URLSearchParams({
                query,
                include_quarantined: String(includeQuarantined),
                limit: '100',
            });
            const response = await fetch(apiUrl(`/projects/${projectId}/memory?${params}`));
            const body = response.ok ? await response.json() : { entries: [] };
            setEntries(Array.isArray(body.entries) ? body.entries : []);
        } catch {
            setEntries([]);
        } finally {
            setLoading(false);
        }
    }, [projectId, query, includeQuarantined]);

    useEffect(() => { void load(); }, [load]);

    const remember = async () => {
        const text = draft.trim();
        if (!text || !projectId) return;
        setBusy(true);
        try {
            const response = await fetch(apiUrl(`/projects/${projectId}/memory`), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text, kind, pinned }),
            });
            if (!response.ok) throw new Error((await response.json()).detail || 'Could not save memory');
            const entry = await response.json();
            setDraft('');
            setPinned(false);
            addToast(entry.quarantined ? 'Saved and quarantined for review' : 'Memory saved', entry.quarantined ? 'error' : 'success');
            await load();
        } catch (exc) {
            addToast(exc instanceof Error ? exc.message : 'Could not save memory', 'error');
        } finally {
            setBusy(false);
        }
    };

    const mutate = async (entry: MemoryEntry, patch: { pinned?: boolean }) => {
        if (!projectId) return;
        const response = await fetch(apiUrl(`/projects/${projectId}/memory/${entry.id}`), {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(patch),
        });
        if (!response.ok) {
            addToast('Could not update memory', 'error');
            return;
        }
        await load();
    };

    const approve = async (entry: MemoryEntry) => {
        const response = await fetch(apiUrl(`/projects/${projectId}/memory/${entry.id}/approve`), {
            method: 'POST',
        });
        if (!response.ok) {
            addToast('Could not approve memory', 'error');
            return;
        }
        addToast('Memory approved for retrieval', 'success');
        await load();
    };

    const forget = async (entry: MemoryEntry) => {
        const ok = await requestConfirm({
            title: 'Delete memory?',
            body: 'This entry will no longer be available to the agent.',
            confirmLabel: 'Delete',
            tone: 'danger',
        });
        if (!ok) return;
        const response = await fetch(apiUrl(`/projects/${projectId}/memory/${entry.id}`), {
            method: 'DELETE',
        });
        if (!response.ok) {
            addToast('Could not delete memory', 'error');
            return;
        }
        await load();
    };

    const clearAll = async () => {
        const ok = await requestConfirm({
            title: 'Clear project memory?',
            body: 'Every note, decision, and preference for this project is deleted.',
            confirmLabel: 'Clear all',
            tone: 'danger',
        });
        if (!ok) return;
        const response = await fetch(apiUrl(`/projects/${projectId}/memory`), {
            method: 'DELETE',
        });
        const body = response.ok ? await response.json() : { deleted: 0 };
        addToast(`Deleted ${body.deleted ?? 0} memory entries`, 'success');
        await load();
    };

    const exportAll = async () => {
        const response = await fetch(apiUrl(`/projects/${projectId}/memory/export`));
        if (!response.ok) {
            addToast('Export failed', 'error');
            return;
        }
        const body = await response.json();
        const blob = new Blob([JSON.stringify(body, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const anchor = document.createElement('a');
        anchor.href = url;
        anchor.download = `agent-memory-${projectId}.json`;
        anchor.click();
        URL.revokeObjectURL(url);
    };

    if (!projectId) return null;

    return (
        <section className="st-panel" aria-label="Agent memory" style={{ marginBottom: 14 }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 10 }}>
                <h3 className="st-panel-title" style={{ margin: 0 }}>
                    <Brain size={12} /> Agent memory
                </h3>
                <div className="st-button-row">
                    <button type="button" className="st-button" data-size="sm" onClick={() => void exportAll()}>
                        <Download size={11} /> Export
                    </button>
                    <button type="button" className="st-button" data-size="sm" data-variant="ghost" onClick={() => void clearAll()}>
                        <Trash2 size={11} /> Clear all
                    </button>
                </div>
            </div>
            <p className="st-value" style={{ color: 'var(--st-muted)', fontSize: 11, marginTop: 4 }}>
                Local, project-scoped facts, decisions, and preferences. Quarantined entries never reach the model until approved.
            </p>

            <div className="st-row" style={{ marginTop: 8 }}>
                <label className="st-grow">
                    <span className="st-label"><Search size={10} /> Search</span>
                    <input
                        className="st-input"
                        value={query}
                        placeholder="Find a fact, decision, or preference"
                        onChange={(event) => setQuery(event.target.value)}
                    />
                </label>
                <label style={{ alignSelf: 'end' }}>
                    <span className="st-label">Review</span>
                    <button
                        type="button"
                        className="st-button"
                        data-size="sm"
                        data-variant={includeQuarantined ? 'primary' : 'ghost'}
                        onClick={() => setIncludeQuarantined((value) => !value)}
                    >
                        <ShieldAlert size={11} /> {includeQuarantined ? 'Showing quarantined' : 'Hiding quarantined'}
                    </button>
                </label>
            </div>

            <div style={{ marginTop: 8 }}>
                <label>
                    <span className="st-label">
                        <Plus size={10} /> Remember
                    </span>
                    <textarea
                        className="st-input"
                        rows={2}
                        value={draft}
                        placeholder="The rooftops are always wet after the chase; the director rejected the dry cut."
                        onChange={(event) => setDraft(event.target.value)}
                    />
                </label>
                <div className="st-button-row" style={{ marginTop: 6, alignItems: 'center' }}>
                    <select className="st-input" value={kind} onChange={(event) => setKind(event.target.value as typeof kind)}>
                        {KINDS.map((value) => <option key={value} value={value}>{value}</option>)}
                    </select>
                    <button
                        type="button"
                        className="st-button"
                        data-size="sm"
                        data-variant={pinned ? 'primary' : 'ghost'}
                        onClick={() => setPinned((value) => !value)}
                    >
                        <Pin size={11} /> Pin
                    </button>
                    <button type="button" className="st-button" data-size="sm" disabled={busy || !draft.trim()} onClick={() => void remember()}>
                        {busy ? <Loader2 size={11} className="animate-spin" /> : <Plus size={11} />} Save
                    </button>
                </div>
            </div>

            <div className="st-link-list" style={{ marginTop: 10 }}>
                {loading && entries.length === 0 && (
                    <div className="st-value"><Loader2 size={11} className="animate-spin" /> Loading memory…</div>
                )}
                {!loading && entries.length === 0 && (
                    <div className="st-value" style={{ color: 'var(--st-faint)' }}>No memory entries yet.</div>
                )}
                {entries.map((entry) => (
                    <div
                        key={entry.id}
                        className="st-link-row"
                        style={{ alignItems: 'flex-start', gap: 8, padding: '6px 0' }}
                    >
                        <span className="st-chip" title={`${entry.provenance} · ${entry.trust}`}>{entry.kind}</span>
                        <div style={{ flex: 1, minWidth: 0 }}>
                            <div style={{ fontSize: 12 }}>{entry.text}</div>
                            <div className="st-value" style={{ fontSize: 10, color: 'var(--st-faint)', marginTop: 2 }}>
                                {entry.trust} · used {entry.use_count}× · {entry.tokens} tokens
                                {entry.quarantined ? ' · quarantined' : ''}
                                {entry.why.length ? ` · ${entry.why.join(', ')}` : ''}
                            </div>
                        </div>
                        {entry.quarantined && (
                            <button type="button" className="st-button" data-size="sm" onClick={() => void approve(entry)}>
                                Approve
                            </button>
                        )}
                        <button
                            type="button"
                            className="st-button"
                            data-size="sm"
                            data-variant="ghost"
                            title={entry.pinned ? 'Unpin' : 'Pin'}
                            onClick={() => void mutate(entry, { pinned: !entry.pinned })}
                        >
                            {entry.pinned ? <Pin size={11} /> : <PinOff size={11} />}
                        </button>
                        <button
                            type="button"
                            className="st-button"
                            data-size="sm"
                            data-variant="ghost"
                            title="Delete memory"
                            onClick={() => void forget(entry)}
                        >
                            <Trash2 size={11} />
                        </button>
                    </div>
                ))}
            </div>
        </section>
    );
}
