
export interface TimelineClip {
    id: string;
    start: number;
    duration: number;
    end: number;
    track: number;
    shot: any;
}

/**
 * Computes the absolute layout of shots on the timeline,
 * implementing the "magnetic" behavior for V1/Track 0.
 */
export const computeTimelineLayout = (project: any): TimelineClip[] => {
    if (!project || !project.shots) return [];

    const fps = project.fps || 25;
    const clips: TimelineClip[] = [];
    let v1Time = 0;

    // We process shots in the order they appear in the project.shots array.
    // For V1 (Track 0), the start time is strictly sequential based on previous V1 shots.
    // For other tracks, start time is based on shot.startFrame.

    project.shots.forEach((shot: any) => {
        const trackIndex = shot.trackIndex || 0;

        // Calculate duration based on frame count and trims
        const rawDuration = (shot.numFrames - (shot.trimIn || 0) - (shot.trimOut || 0));
        const duration = Math.max(1, rawDuration) / fps;

        let start = 0;

        if (trackIndex === 0) {
            // Magnetic V1: Starts where the previous V1 ended
            start = v1Time;
            v1Time += duration;
        } else {
            // Free placement for V2/A1
            start = Math.max(0, (shot.startFrame || 0) / fps);
        }

        clips.push({
            id: shot.id,
            start: start,
            duration: duration,
            end: start + duration,
            track: trackIndex,
            shot: shot
        });
    });

    return clips;
};

/**
 * Calculates the total duration of the project based on the computed layout.
 */
export const computeTotalDuration = (clips: TimelineClip[]): number => {
    return clips.reduce((acc, clip) => Math.max(acc, clip.end), 0);
};
