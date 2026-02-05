import { useTimelineStore } from '../../stores/timelineStore';

interface Props {
    projectFps?: number;
    totalDuration: number;
}

export const TimeDisplay = ({ projectFps = 25, totalDuration }: Props) => {
    const currentTime = useTimelineStore(state => state.currentTime);

    const formatTime = (seconds: number) => {
        const m = Math.floor(seconds / 60);
        const s = Math.floor(seconds % 60);
        const f = Math.floor((seconds % 1) * projectFps);
        return `${m}:${s.toString().padStart(2, '0')}:${f.toString().padStart(2, '0')}`;
    };

    return (
        <div className="text-xs font-mono text-milimo-300 ml-2">
            {formatTime(currentTime)} / {formatTime(totalDuration)}
        </div>
    );
};
