import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, render } from '@testing-library/react';
import { PlaybackEngine } from './PlaybackEngine';
import { useStudioDocumentStore } from '../state/documentStore';

const audio = {
    sync: vi.fn(),
    tick: vi.fn(),
    stopAll: vi.fn(),
};

vi.mock('../../utils/GlobalAudioManager', () => ({
    GlobalAudioManager: {
        getInstance: () => audio,
    },
}));

describe('studio PlaybackEngine', () => {
    let rafSpy: ReturnType<typeof vi.spyOn>;
    let cancelSpy: ReturnType<typeof vi.spyOn>;

    beforeEach(() => {
        audio.sync.mockClear();
        audio.tick.mockClear();
        audio.stopAll.mockClear();
        rafSpy = vi.spyOn(window, 'requestAnimationFrame').mockReturnValue(1 as unknown as number);
        cancelSpy = vi.spyOn(window, 'cancelAnimationFrame').mockImplementation(() => {});
        const base = useStudioDocumentStore.getState().project;
        useStudioDocumentStore.setState({
            isPlaying: false,
            currentTime: 0,
            project: {
                ...base,
                shots: [{
                    id: 'audio-1',
                    prompt: '',
                    negativePrompt: '',
                    seed: 1,
                    width: 1280,
                    height: 720,
                    numFrames: 75,
                    timeline: [],
                    cfgScale: 1,
                    enhancePrompt: false,
                    upscale: false,
                    pipelineOverride: 'auto',
                    trackIndex: 2,
                    trimIn: 0,
                    trimOut: 0,
                    videoUrl: '/uploads/audio.mp4',
                }],
            },
        });
    });

    afterEach(() => {
        cleanup();
        rafSpy.mockRestore();
        cancelSpy.mockRestore();
        const base = useStudioDocumentStore.getState().project;
        useStudioDocumentStore.setState({
            isPlaying: false,
            currentTime: 0,
            project: {
                ...base,
                shots: [{
                    id: 'audio-1',
                    prompt: '',
                    negativePrompt: '',
                    seed: 1,
                    width: 1280,
                    height: 720,
                    numFrames: 75,
                    timeline: [],
                    cfgScale: 1,
                    enhancePrompt: false,
                    upscale: false,
                    pipelineOverride: 'auto',
                    trackIndex: 2,
                    trimIn: 0,
                    trimOut: 0,
                    videoUrl: '/uploads/audio.mp4',
                }],
            },
        });
    });

    it('does not schedule a frame while paused', () => {
        render(<PlaybackEngine />);
        expect(rafSpy).not.toHaveBeenCalled();
    });

    it('runs the clock only while playing and cancels on pause', () => {
        render(<PlaybackEngine />);
        act(() => {
            useStudioDocumentStore.setState({ isPlaying: true });
        });
        expect(rafSpy).toHaveBeenCalledTimes(1);

        act(() => {
            useStudioDocumentStore.setState({ isPlaying: false });
        });
        expect(cancelSpy).toHaveBeenCalled();
        const afterPause = rafSpy.mock.calls.length;
        act(() => {
            useStudioDocumentStore.setState({ currentTime: 2 });
        });
        // A paused seek ticks audio once but never starts another frame.
        expect(rafSpy.mock.calls.length).toBe(afterPause);
        expect(audio.tick).toHaveBeenCalled();
    });
});
