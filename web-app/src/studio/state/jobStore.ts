import { create } from 'zustand';
import type { ActiveJob, JobMetrics } from '../data/useStudioData';

export interface JobHistoryEntry extends ActiveJob {
    finished_at: number;
    error?: string | null;
}

interface JobState {
    activeJobs: ActiveJob[];
    history: JobHistoryEntry[];
    metrics: JobMetrics | null;
    lastUpdated: number | null;
    unread: number;
    setActiveJobs: (jobs: ActiveJob[]) => void;
    setMetrics: (metrics: JobMetrics | null) => void;
    upsertActive: (job: ActiveJob) => void;
    finish: (job: JobHistoryEntry) => void;
    clearUnread: () => void;
}

function mergeJobs(current: ActiveJob[], incoming: ActiveJob[]): ActiveJob[] {
    const byId = new Map(current.map((job) => [job.job_id, job]));
    for (const job of incoming) {
        const existing = byId.get(job.job_id);
        byId.set(job.job_id, existing ? { ...existing, ...job } : job);
    }
    return [...byId.values()].sort((a, b) => (a.position ?? 999) - (b.position ?? 999));
}

/** Isolated job/queue state; document and playback never subscribe here. */
export const useStudioJobs = create<JobState>((set) => ({
    activeJobs: [],
    history: [],
    metrics: null,
    lastUpdated: null,
    unread: 0,
    // `/active_jobs` is authoritative: replace, so a missed terminal event does
    // not leave a finished job spinning forever.
    setActiveJobs: (jobs) => set(() => ({
        activeJobs: [...jobs].sort((a, b) => (a.position ?? 999) - (b.position ?? 999)),
        lastUpdated: Date.now(),
    })),
    setMetrics: (metrics) => set({ metrics }),
    upsertActive: (job) => set((state) => ({
        activeJobs: mergeJobs(state.activeJobs, [job]),
        lastUpdated: Date.now(),
    })),
    finish: (job) => set((state) => ({
        activeJobs: state.activeJobs.filter((item) => item.job_id !== job.job_id),
        history: [job, ...state.history.filter((item) => item.job_id !== job.job_id)].slice(0, 20),
        unread: state.unread + 1,
        lastUpdated: Date.now(),
    })),
    clearUnread: () => set({ unread: 0 }),
}));
