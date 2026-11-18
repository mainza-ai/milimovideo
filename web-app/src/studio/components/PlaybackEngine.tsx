import { useEffect, useRef } from 'react';
import { GlobalAudioManager } from '../../utils/GlobalAudioManager';
import { computeTimelineLayout, type TimelineClip } from '../../utils/timelineUtils';
import { useStudioDocumentStore } from '../state/documentStore';

/** Studio-owned playback clock; separate from the legacy editor store. */
export const PlaybackEngine = () => {
    const project = useStudioDocumentStore(state => state.project);
    const rafRef = useRef<number | null>(null);
    const lastTimeRef = useRef<number>(0);

    const cachedLayoutRef = useRef<{ clips: TimelineClip[]; maxDuration: number }>({
        clips: [],
        maxDuration: 0,
    });

    useEffect(() => {
        const clips = computeTimelineLayout(project);
        const maxDuration = clips.reduce((acc, c) => Math.max(acc, c.end), 0);
        cachedLayoutRef.current = { clips, maxDuration };
    }, [project.shots, project.fps]);

    useEffect(() => {
        const audioClips = project.shots.filter(s => (s.trackIndex === 2 || s.trackIndex === 3) && s.videoUrl);
        GlobalAudioManager.getInstance().sync(audioClips);
    }, [project.shots]);

    useEffect(() => {
        const loop = (timestamp: number) => {
            const { isPlaying, currentTime, setCurrentTime, trackStates, project: stateProject } =
                useStudioDocumentStore.getState();

            let nextTime = currentTime;

            if (isPlaying) {
                if (lastTimeRef.current === 0) {
                    lastTimeRef.current = timestamp;
                }
                const dt = (timestamp - lastTimeRef.current) / 1000;
                lastTimeRef.current = timestamp;
                nextTime = currentTime + dt;

                const { maxDuration } = cachedLayoutRef.current;
                if (nextTime >= maxDuration + 0.1) {
                    useStudioDocumentStore.getState().setIsPlaying(false);
                }
                setCurrentTime(nextTime);
            } else {
                lastTimeRef.current = 0;
            }

            const audioClips = stateProject.shots.filter(s => s.trackIndex === 2 || s.trackIndex === 3);
            const hasSolo = Object.values(trackStates).some((state) => !!state?.solo);
            const trackMuted = (clip: { trackIndex?: number }) => {
                const state = trackStates[clip.trackIndex ?? 0];
                if (state?.solo) return false;
                if (hasSolo) return true;
                return !!state?.muted;
            };
            GlobalAudioManager.getInstance().tick(
                nextTime,
                stateProject.fps || 25,
                isPlaying,
                false,
                audioClips,
                trackMuted,
            );

            rafRef.current = requestAnimationFrame(loop);
        };

        rafRef.current = requestAnimationFrame(loop);

        return () => {
            if (rafRef.current) cancelAnimationFrame(rafRef.current);
            GlobalAudioManager.getInstance().stopAll();
        };
    }, []);

    return null;
};
