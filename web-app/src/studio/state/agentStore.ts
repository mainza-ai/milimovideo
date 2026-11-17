import { create } from 'zustand';

export interface AgentPlanRecord {
    id: string;
    mode: string;
    summary: string;
    text: string;
    shotId: string | null;
    created: number;
}

interface AgentState {
    lastPlan: AgentPlanRecord | null;
    history: AgentPlanRecord[];
    recordPlan: (plan: Omit<AgentPlanRecord, 'id' | 'created'>) => void;
    clearHistory: () => void;
}

/** Isolated agent/plan state; document edits remain in the document store. */
export const useStudioAgent = create<AgentState>((set) => ({
    lastPlan: null,
    history: [],
    recordPlan: (plan) => set((state) => {
        const record: AgentPlanRecord = {
            ...plan,
            id: `plan_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`,
            created: Date.now(),
        };
        return {
            lastPlan: record,
            history: [record, ...state.history].slice(0, 50),
        };
    }),
    clearHistory: () => set({ lastPlan: null, history: [] }),
}));
