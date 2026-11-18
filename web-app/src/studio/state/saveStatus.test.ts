import { describe, expect, it, beforeEach } from 'vitest';
import { isDocumentChange, saveStatusLabel, saveStatusTone, shotHasTake } from '../../stores/saveStatus';
import { useStudioDocumentStore } from '../state/documentStore';

describe('save status vocabulary', () => {
    it('never calls an unsaved document "Saved"', () => {
        expect(saveStatusLabel('saved')).toBe('Saved');
        expect(saveStatusLabel('dirty')).toBe('Unsaved');
        expect(saveStatusLabel('idle')).toBe('Not saved');
        expect(saveStatusLabel('saving')).toBe('Saving…');
        expect(saveStatusLabel('conflict')).toBe('Save conflict');
        expect(saveStatusLabel('error')).toBe('Save failed');
    });

    it('uses success tone only for a saved document', () => {
        expect(saveStatusTone('saved')).toBe('good');
        expect(saveStatusTone('dirty')).toBe('warn');
        expect(saveStatusTone('idle')).toBe('warn');
        expect(saveStatusTone('conflict')).toBe('bad');
        expect(saveStatusTone('error')).toBe('bad');
    });
});

describe('document dirty tracking', () => {
    beforeEach(() => {
        useStudioDocumentStore.setState({ saveStatus: 'saved' });
    });

    it('ignores job runtime fields', () => {
        expect(isDocumentChange({ progress: 42, status: 'generating', stage: 'denoising' })).toBe(false);
        const shotId = useStudioDocumentStore.getState().project.shots[0].id;
        useStudioDocumentStore.getState().updateShot(shotId, { progress: 42, status: 'generating' });
        expect(useStudioDocumentStore.getState().saveStatus).toBe('saved');
    });

    it('marks real edits dirty', () => {
        const shotId = useStudioDocumentStore.getState().project.shots[0].id;
        useStudioDocumentStore.getState().updateShot(shotId, { prompt: 'a new frame' });
        expect(useStudioDocumentStore.getState().saveStatus).toBe('dirty');
    });

    it('treats an existing take as completed media', () => {
        expect(shotHasTake({ status: 'pending', videoUrl: '/projects/p/generated/a.mp4' })).toBe(true);
        expect(shotHasTake({ status: 'completed' })).toBe(true);
        expect(shotHasTake({ status: 'pending', thumbnailUrl: undefined })).toBe(false);
    });
});
