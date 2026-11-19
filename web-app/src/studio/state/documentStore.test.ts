import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
    sanitizeProjectForStorage,
    useStudioDocumentStore,
} from './documentStore';
import type { Project, Shot } from '../../stores/types';

function makeShot(id: string): Shot {
    return {
        id,
        prompt: `prompt ${id}`,
        negativePrompt: '',
        seed: 1,
        width: 1280,
        height: 768,
        numFrames: 121,
        timeline: [],
        cfgScale: 3,
        enhancePrompt: true,
        upscale: false,
        pipelineOverride: 'auto',
        trackIndex: 0,
        trimIn: 0,
        trimOut: 0,
        // Runtime fields that must never be persisted.
        progress: 42,
        statusMessage: 'Denoising',
        currentPrompt: 'live prompt',
        etaSeconds: 90,
        enhancedPromptResult: 'x'.repeat(4000),
        lastJobId: 'job-1',
        isGenerating: true,
        status: 'completed',
        videoUrl: '/uploads/shot.mp4',
    };
}

function makeProject(shots: Shot[]): Project {
    const base = useStudioDocumentStore.getState().project;
    return {
        ...base,
        id: 'project-1',
        name: 'Sanitize test',
        shots,
    } as Project;
}

describe('studio document persistence', () => {
    beforeEach(() => {
        vi.useFakeTimers();
        window.localStorage.clear();
    });

    afterEach(() => {
        vi.useRealTimers();
        window.localStorage.clear();
    });

    it('strips runtime shot fields but keeps document fields', () => {
        const project = makeProject([makeShot('shot-1')]);
        const clean = sanitizeProjectForStorage(project);
        const shot = clean.shots[0] as unknown as Record<string, unknown>;
        for (const field of [
            'progress',
            'statusMessage',
            'currentPrompt',
            'etaSeconds',
            'enhancedPromptResult',
            'lastJobId',
            'isGenerating',
        ]) {
            expect(shot[field], field).toBeUndefined();
        }
        expect(shot.status).toBe('completed');
        expect(shot.videoUrl).toBe('/uploads/shot.mp4');
        expect(shot.prompt).toBe('prompt shot-1');
    });

    it('persists a sanitized, size-bounded document after the debounce', () => {
        useStudioDocumentStore.setState({
            project: makeProject(Array.from({ length: 500 }, (_, index) => makeShot(`shot-${index}`))),
        });
        vi.advanceTimersByTime(1000);
        const raw = window.localStorage.getItem('milimo-studio-document');
        expect(raw).toBeTruthy();
        expect(raw).not.toContain('enhancedPromptResult');
        expect(raw).not.toContain('"currentPrompt"');
        // 500 runtime-free shots stay far below a localStorage-hostile size.
        expect(raw!.length).toBeLessThan(1_000_000);
    });

    it('coalesces rapid writes into one localStorage write', () => {
        const spy = vi.spyOn(window.localStorage, 'setItem');
        const project = makeProject([makeShot('shot-1')]);
        useStudioDocumentStore.setState({ project });
        useStudioDocumentStore.setState({ project: { ...project, name: 'renamed' } });
        useStudioDocumentStore.setState({ project: { ...project, name: 'renamed again' } });
        expect(spy).not.toHaveBeenCalled();
        vi.advanceTimersByTime(1000);
        expect(spy).toHaveBeenCalledTimes(1);
        expect(window.localStorage.getItem('milimo-studio-document')).toContain('renamed again');
    });
});
