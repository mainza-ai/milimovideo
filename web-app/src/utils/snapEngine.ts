export interface SnapResult {
    snappedTime: number;
    isSnapped: boolean;
    snapLines: number[]; // Time points where lines should be drawn
}

export interface ClipData {
    id: string;
    start: number;
    duration: number;
    trackIndex: number;
}

export const SNAP_THRESHOLD_PX = 15;

export const getSnapPoint = (
    time: number,
    zoom: number,
    movingClipId: string | null,
    clips: ClipData[],
    playheadTime: number,
    thresholdPx: number = SNAP_THRESHOLD_PX
): SnapResult => {
    const thresholdSec = thresholdPx / zoom;
    let closestDist = Infinity;
    let bestSnap = time;

    const candidates = [
        0, // Timeline start
        playheadTime // Playhead
    ];

    // Add all clip start/ends (excluding the one moving)
    clips.forEach(c => {
        if (c.id === movingClipId) return;
        candidates.push(c.start);
        candidates.push(c.start + c.duration);
    });

    // Find closest candidate
    candidates.forEach(cand => {
        const dist = Math.abs(cand - time);
        if (dist < closestDist) {
            closestDist = dist;
            bestSnap = cand;
        }
    });

    if (closestDist <= thresholdSec) {
        return {
            snappedTime: bestSnap,
            isSnapped: true,
            snapLines: [bestSnap]
        };
    }

    return {
        snappedTime: time,
        isSnapped: false,
        snapLines: []
    };
};
