import { useEffect, useRef } from 'react';
import { useTimelineStore } from '../../stores/timelineStore';
import { GlobalAudioManager } from '../../utils/GlobalAudioManager';
import { computeTimelineLayout } from '../../utils/timelineUtils';

export const PlaybackEngine = () => {
    const project = useTimelineStore(state => state.project);
    const rafRef = useRef<number | null>(null);
    const lastTimeRef = useRef<number>(0);

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

                // Check End of Project using centralized layout logic
                // Avoid re-computing every frame if possible, but project changes rarely during playback?
                // Actually project state changes on every seek/edit.
                // Ideally this is memoized outside the loop, but loop closes over project state from getState().

                // We can compute maxDuration from the current state efficiently.
                // Or better, store it in the store? 
                // For now, let's just compute it. It's fast (hundreds of items max).
                // Check End of Project using centralized layout logic
                const clips = computeTimelineLayout(project);
                const maxDuration = clips.reduce((acc, c) => Math.max(acc, c.end), 0);

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
            // On unmount, usually we pause? 
            // Or if we are just hot-reloading?
            // Safer to pause all audio to avoid ghosts if this component dies.
            GlobalAudioManager.getInstance().stopAll();
        };
    }, []);

    return null; // Headless
};

