import { useEffect, useRef } from 'react';
import WaveSurfer from 'wavesurfer.js';
import { useTimelineStore } from '../../stores/timelineStore';

interface Props {
    clip: any;
    zoom: number;
    color?: string;
}

export const AudioClip = ({ clip, zoom, color = '#a3e635' }: Props) => {
    const containerRef = useRef<HTMLDivElement>(null);
    const wavesurferRef = useRef<WaveSurfer | null>(null);
    const fps = useTimelineStore(state => state.project.fps || 25);
    const hasError = useRef(false);

    // Calculate full dimensions
    // We want the waveform to represent the FULL source audio
    // So width should be based on (numFrames / fps) * zoom
    // BUT clip.shot.numFrames might be the *trimmed* length if we aren't careful?
    // In our model:
    // shot.numFrames = trimmed duration? NO. 
    // Usually shot.numFrames is the duration of the SHOT on timeline.
    // We need the SOURCE duration.
    // Actually, `numFrames` in our store usually means "Duration on Timeline".
    // Does the shot have `sourceDuration` or `totalFrames`?
    // Let's assume for now we don't have the full source length easily available in `shot` root properties 
    // unless we look at defaults.
    // However, WaveSurfer naturally renders the WHOLE file.
    // If we set the container width to `(sourceDuration) * zoom`, WaveSurfer spreads correctly.
    // But we don't know sourceDuration until WaveSurfer loads it or we have metadata.

    // Alternative:
    // Let WaveSurfer render to a width that corresponds to the VISIBLE duration * scaling factor?
    // NO.
    // If we use `fillParent: true`, it fits the container.
    // We need the container to be the size of the FULL audio.
    // Visually: 
    // [ Hidden Left ] [ Visible Clip ] [ Hidden Right ]
    // The `AudioClip` component is inside `TimelineClip` which has `overflow: hidden`.
    // So we just need to make `AudioClip` div huge (Full Source Width) and position it relative to `trimIn`.

    // If we don't know full source duration, we can wait for WaveSurfer `ready` event to get `duration`.

    useEffect(() => {
        if (!containerRef.current || !clip.shot?.videoUrl) return;

        const initWaveSurfer = () => {
            if (!containerRef.current || wavesurferRef.current || hasError.current) return;

            const url = clip.shot.videoUrl.startsWith('http')
                ? clip.shot.videoUrl
                : `http://localhost:8000${clip.shot.videoUrl}`;

            // Detect Safari for backend selection
            const isSafari = /^((?!chrome|android).)*safari/i.test(navigator.userAgent);

            try {
                const wsOptions: any = {
                    container: containerRef.current,
                    waveColor: color,
                    progressColor: color,
                    cursorColor: 'transparent',
                    barWidth: 2,
                    barGap: 1,
                    barRadius: 2,
                    height: 48,
                    fillParent: true,
                    interact: false,
                    url: url,
                    fetchParams: {
                        mode: 'cors',
                    },
                };

                // Safari: Use MediaElement backend for better compatibility
                if (isSafari) {
                    wsOptions.backend = 'MediaElement';
                    wsOptions.mediaControls = false;
                }

                wavesurferRef.current = WaveSurfer.create(wsOptions);

                wavesurferRef.current.on('error', (err) => {
                    console.warn("WaveSurfer error", err);
                    hasError.current = true;
                });

                // When ready, we can check duration and set width of container
                wavesurferRef.current.on('ready', (duration) => {
                    if (containerRef.current) {
                        const fullWidth = duration * zoom;
                        containerRef.current.style.width = `${fullWidth}px`;
                    }
                });
            } catch (e) {
                console.error("Failed to create WaveSurfer", e);
                hasError.current = true;
            }
        };

        requestAnimationFrame(initWaveSurfer);

        return () => {
            wavesurferRef.current?.destroy();
            wavesurferRef.current = null;
        };
    }, [clip.id, clip.shot?.videoUrl, color]);


    // Update width and position whenever zoom or trim changes
    // We need access to the WS instance to know the duration though?
    // Or we assume 10s default?
    // Better: use WS duration if available.

    // To make this responsive to Zoom without re-init WS:
    // We update the DIV style. WS observes size changes if we didn't disable it.
    // WaveSurfer 7 automatically redraws on resize by default? yes.

    useEffect(() => {
        const ws = wavesurferRef.current;
        const div = containerRef.current;
        if (!ws || !div) return;

        const duration = ws.getDuration();
        if (duration > 0) {
            const width = duration * zoom;
            div.style.width = `${width}px`;
        }
    }, [zoom]); // Update width when zoom changes

    // Interaction with Parent:
    // Parent `TimelineClip` is WIDTH = (DurationOnTimeline) * Zoom.
    // Parent has `overflow: hidden`.
    // This Component (AudioClip) is child.
    // We need to shift this component LEFT by `trimIn`.

    const trimInSeconds = (clip.shot.trimIn || 0) / fps;
    const leftOffset = -1 * trimInSeconds * zoom;

    return (
        <div
            className="h-full relative overflow-hidden"
            style={{ width: '100%', height: '100%' }}
        >
            <div
                ref={containerRef}
                className="h-full opacity-80 pointer-events-none absolute top-0"
                style={{
                    left: `${leftOffset}px`,
                    // Width is set dynamic via JS or we can try to estimate?
                    // Initial width 100% is wrong if we rely on WS.
                    // Let's rely on the useEffect above to set width once duration is known.
                    // But we can set a min-width?
                    minWidth: '100%'
                }}
            />
        </div>
    );
};
