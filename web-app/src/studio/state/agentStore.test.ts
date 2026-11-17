import { beforeEach, describe, expect, it } from 'vitest';
import { useStudioAgent } from './agentStore';

describe('agent store', () => {
    beforeEach(() => {
        useStudioAgent.getState().clearHistory();
    });

    it('records plans without touching document state', () => {
        useStudioAgent.getState().recordPlan({
            mode: 'direct',
            summary: 'one edit',
            text: 'seed 7',
            shotId: 'shot-1',
        });
        const state = useStudioAgent.getState();
        expect(state.history).toHaveLength(1);
        expect(state.lastPlan?.shotId).toBe('shot-1');
        expect(state.lastPlan?.id.startsWith('plan_')).toBe(true);
    });
});
