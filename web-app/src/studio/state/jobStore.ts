import { create } from 'zustand';
import type { ActiveJob, JobMetrics } from '../data/useStudioData';

interface JobState {
    activeJobs: ActiveJob[];
    metrics: JobMetrics | null;
    lastUpdated: number | null;
    setActiveJobs: (jobs: ActiveJob[]) => void;
    setMetrics: (metrics: JobMetrics | null) => void;
}

/** Isolated job/queue state; document and playback never subscribe here. */
export const useStudioJobs = create<JobState>((set) => ({
    activeJobs: [],
    metrics: null,
    lastUpdated: null,
    setActiveJobs: (activeJobs) => set({ activeJobs, lastUpdated: Date.now() }),
    setMetrics: (metrics) => set({ metrics }),
}));
