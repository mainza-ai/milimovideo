import { beforeEach, describe, expect, it } from 'vitest';
import { useStudioDocumentStore } from './documentStore';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const temporal = (useStudioDocumentStore as any).temporal;

describe('document undo', () => {
    beforeEach(() => {
        temporal.getState().clear();
    });

    it('restores a plan-style shot patch in one undo step', () => {
        const store = useStudioDocumentStore.getState();
        if (store.project.shots.length === 0) {
            store.addShot({ prompt: 'original' });
        }
        temporal.getState().clear();
        const shot = useStudioDocumentStore.getState().project.shots[0];
        const before = shot.prompt;

        useStudioDocumentStore.getState().updateShot(shot.id, {
            prompt: 'approved plan text',
            seed: 1234,
            numFrames: 81,
        });
        expect(useStudioDocumentStore.getState().project.shots[0].prompt).toBe('approved plan text');

        temporal.getState().undo();
        const restored = useStudioDocumentStore.getState().project.shots[0];
        expect(restored.prompt).toBe(before);
        expect(restored.seed).toBe(shot.seed);
        expect(restored.numFrames).toBe(shot.numFrames);
    });
});
