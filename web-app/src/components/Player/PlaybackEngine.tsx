import { useEffect, useRef } from 'react';
import { useTimelineStore } from '../../stores/timelineStore';
import { GlobalAudioManager } from '../../utils/GlobalAudioManager';
import { computeTimelineLayout, type TimelineClip } from '../../utils/timelineUtils';

export const PlaybackEngine = () => {
    const project = useTimelineStore(state => state.project);
    const rafRef = useRef<number | null>(null);
    const lastTimeRef = useRef<number>(0);

    // Cache the timeline layout and max duration to avoid recomputation every frame.
    // Only recompute when project.shots changes (not every rAF tick).
    const cachedLayoutRef = useRef<{ clips: TimelineClip[]; maxDuration: number }>({
        clips: [],
        maxDuration: 0,
    });

    useEffect(() => {
        const clips = computeTimelineLayout(project);
        const maxDuration = clips.reduce((acc, c) => Math.max(acc, c.end), 0);
        cachedLayoutRef.current = { clips, maxDuration };
    }, [project.shots, project.fps]);

    // 1. Sync Audio Assets with Global Manager
    useEffect(() => {
        const audioClips = project.shots.filter(s => s.trackIndex === 2 && s.videoUrl);
        GlobalAudioManager.getInstance().sync(audioClips);
    }, [project.shots]);

    // 2. Master Clock Loop
    useEffect(() => {
        const loop = (timestamp: number) => {
            const { isPlaying, currentTime, setCurrentTime, trackStates, project } = useTimelineStore.getState();

            let nextTime = currentTime;

            // Advance Time if Playing
            if (isPlaying) {
                if (lastTimeRef.current === 0) {
                    lastTimeRef.current = timestamp;
                }
                const dt = (timestamp - lastTimeRef.current) / 1000;
                lastTimeRef.current = timestamp;

                nextTime = currentTime + dt;

                // Use cached layout for end-of-project check (no allocation per frame)
                const { maxDuration } = cachedLayoutRef.current;

                if (nextTime >= maxDuration + 0.1) {
                    useTimelineStore.getState().setIsPlaying(false);
                }

                setCurrentTime(nextTime);
            } else {
                lastTimeRef.current = 0;
            }

            // Detect Mute State
            const isMuted = trackStates[2]?.muted;
            const audioClips = project.shots.filter(s => s.trackIndex === 2);

            // Delegate Audio Tick to Global Manager
            GlobalAudioManager.getInstance().tick(
                nextTime,
                project.fps || 25,
                isPlaying,
                isMuted,
                audioClips
            );

            rafRef.current = requestAnimationFrame(loop);
        };

        rafRef.current = requestAnimationFrame(loop);

        return () => {
            if (rafRef.current) cancelAnimationFrame(rafRef.current);
            // Pause all audio to avoid ghosts if this component dies.
            GlobalAudioManager.getInstance().stopAll();
        };
    }, []);

    return null; // Headless
};
