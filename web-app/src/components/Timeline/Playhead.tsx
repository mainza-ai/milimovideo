import { useTimelineStore } from '../../stores/timelineStore';

interface Props {
    zoom: number;
}

export const Playhead = ({ zoom }: Props) => {
    // Subscribe to currentTime to re-render this component on every frame update
    const currentTime = useTimelineStore(state => state.currentTime);

    return (
        <div
            className="absolute top-0 bottom-0 w-px bg-red-500 z-50 pointer-events-none"
            style={{
                left: 128 + (currentTime * zoom),
                boxShadow: '0 0 4px rgba(255, 0, 0, 0.5)'
            }}
        >
            <div className="absolute -top-1 -translate-x-1/2 text-red-500">
                <svg width="11" height="12" viewBox="0 0 11 12" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M0.5 0H10.5V6L5.5 11L0.5 6V0Z" fill="currentColor" />
                </svg>
            </div>
            {/* Optional: Timecode display following the head? */}
        </div>
    );
};
