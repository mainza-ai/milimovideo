import { useCallback, useEffect, useRef } from 'react';
import { GlobalAudioManager } from '../../utils/GlobalAudioManager';
import { computeTimelineLayout, type TimelineClip } from '../../utils/timelineUtils';
import { useStudioDocumentStore, type Shot } from '../state/documentStore';

type TrackStates = ReturnType<typeof useStudioDocumentStore.getState>['trackStates'];

/**
 * Studio playback clock.
 *
 * The legacy engine ran one rAF callback forever, recomputing the audio clip
 * list, solo state, and mute predicate every frame whether or not anything was
 * playing. This version runs the rAF loop only while playing; the clip list and
 * track state are cached in refs and refreshed on document changes. A paused
 * timeline costs one tick per seek, not 60 ticks per second.
 */
export const PlaybackEngine = () => {
    const project = useStudioDocumentStore((state) => state.project);
    const shots = project.shots;
    const fps = project.fps;
    const trackStates = useStudioDocumentStore((state) => state.trackStates);
    const isPlaying = useStudioDocumentStore((state) => state.isPlaying);
    const currentTime = useStudioDocumentStore((state) => state.currentTime);

    const rafRef = useRef<number | null>(null);
    const layoutRef = useRef<{ clips: TimelineClip[]; maxDuration: number }>({
        clips: [],
        maxDuration: 0,
    });
    const audioRef = useRef<{ clips: Shot[]; trackStates: TrackStates; fps: number }>({
        clips: [],
        trackStates: {},
        fps: 25,
    });
    // Guards the paused tick from firing on unrelated renders.
    const lastTickRef = useRef<number>(-1);

    useEffect(() => {
        const clips = computeTimelineLayout(project);
        const maxDuration = clips.reduce((acc, clip) => Math.max(acc, clip.end), 0);
        layoutRef.current = { clips, maxDuration };
    }, [project]);

    useEffect(() => {
        const audioClips = shots.filter(
            (shot) => (shot.trackIndex === 2 || shot.trackIndex === 3) && shot.videoUrl,
        );
        GlobalAudioManager.getInstance().sync(audioClips);
    }, [shots]);

    useEffect(() => {
        audioRef.current = {
            clips: shots.filter((shot) => shot.trackIndex === 2 || shot.trackIndex === 3),
            trackStates,
            fps: fps || 25,
        };
    }, [shots, trackStates, fps]);

    const tickAudio = useCallback((time: number, playing: boolean) => {
        const { clips, trackStates: states, fps: rate } = audioRef.current;
        if (clips.length === 0) return;
        const hasSolo = Object.values(states).some((state) => !!state?.solo);
        const trackMuted = (clip: { trackIndex?: number }) => {
            const state = states[clip.trackIndex ?? 0];
            if (state?.solo) return false;
            if (hasSolo) return true;
            return !!state?.muted;
        };
        GlobalAudioManager.getInstance().tick(time, rate, playing, false, clips, trackMuted);
    }, []);

    useEffect(() => {
        if (!isPlaying) {
            lastTickRef.current = -1;
            return undefined;
        }

        let startTs: number | null = null;
        let base = useStudioDocumentStore.getState().currentTime;
        let written = base;

        const loop = (ts: number) => {
            if (startTs === null) startTs = ts;
            const state = useStudioDocumentStore.getState();
            // An external seek while playing re-bases the local clock instead of
            // refusing to move. Threshold is small but non-zero so our own writes
            // never look like seeks.
            if (Math.abs(state.currentTime - written) > 0.05) {
                base = state.currentTime;
                startTs = ts;
            }
            const next = base + (ts - startTs) / 1000;
            const { maxDuration } = layoutRef.current;
            if (maxDuration > 0 && next >= maxDuration + 0.1) {
                written = maxDuration;
                state.setCurrentTime(maxDuration);
                tickAudio(maxDuration, false);
                state.setIsPlaying(false);
                rafRef.current = null;
                return;
            }
            written = next;
            state.setCurrentTime(next);
            tickAudio(next, true);
            rafRef.current = requestAnimationFrame(loop);
        };

        rafRef.current = requestAnimationFrame(loop);
        return () => {
            if (rafRef.current) cancelAnimationFrame(rafRef.current);
            rafRef.current = null;
            lastTickRef.current = -1;
        };
    }, [isPlaying, tickAudio]);

    // Paused seeks and track toggles need one tick to stop/position audio, but
    // they must not keep a clock running.
    useEffect(() => {
        if (isPlaying) return;
        if (lastTickRef.current === currentTime) return;
        lastTickRef.current = currentTime;
        tickAudio(currentTime, false);
    }, [currentTime, isPlaying, shots, trackStates, tickAudio]);

    useEffect(() => () => {
        GlobalAudioManager.getInstance().stopAll();
    }, []);

    return null;
};
