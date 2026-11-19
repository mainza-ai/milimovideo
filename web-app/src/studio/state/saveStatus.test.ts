import { describe, expect, it, beforeEach } from 'vitest';
import {
    documentSignature, effectiveShotStatus, isDocumentChange, saveStatusLabel,
    saveStatusTone, shotHasTake,
} from '../../stores/saveStatus';
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
        const project = useStudioDocumentStore.getState().project;
        useStudioDocumentStore.setState({
            saveStatus: 'saved',
            savedBaseline: documentSignature(project),
        });
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
        expect(shotHasTake({ status: 'completed' })).toBe(false);
        expect(shotHasTake({ status: 'pending', thumbnailUrl: undefined })).toBe(false);
    });

    it('keeps a save conflict sticky through edits', () => {
        useStudioDocumentStore.setState({ saveStatus: 'conflict' });
        const shotId = useStudioDocumentStore.getState().project.shots[0].id;
        useStudioDocumentStore.getState().updateShot(shotId, { prompt: 'conflicted edit' });
        expect(useStudioDocumentStore.getState().saveStatus).toBe('conflict');
    });

    it('marks undo after a save dirty and redo saved again', () => {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const temporal = (useStudioDocumentStore as any).temporal;
        temporal.getState().clear();
        const shotId = useStudioDocumentStore.getState().project.shots[0].id;
        useStudioDocumentStore.getState().updateShot(shotId, { prompt: 'saved edit' });
        const saved = useStudioDocumentStore.getState().project;
        useStudioDocumentStore.setState({
            saveStatus: 'saved',
            savedBaseline: documentSignature(saved),
        });
        temporal.getState().undo();
        expect(useStudioDocumentStore.getState().saveStatus).toBe('dirty');
        temporal.getState().redo();
        expect(useStudioDocumentStore.getState().saveStatus).toBe('saved');
    });

    it('does not dirty the document when a job completes', () => {
        const shotId = useStudioDocumentStore.getState().project.shots[0].id;
        useStudioDocumentStore.getState().updateShot(shotId, {
            status: 'completed', progress: 100, videoUrl: '/projects/p/generated/new.mp4',
        });
        expect(useStudioDocumentStore.getState().saveStatus).toBe('saved');
    });

    it('derives card status from media as well as the status field', () => {
        expect(effectiveShotStatus({ status: 'pending', videoUrl: '/a.mp4' })).toBe('completed');
        expect(effectiveShotStatus({ status: 'completed' })).toBe('missing');
        expect(effectiveShotStatus({ status: 'failed', videoUrl: '/a.mp4' })).toBe('failed');
        expect(effectiveShotStatus({ status: 'generating' })).toBe('generating');
    });
});
