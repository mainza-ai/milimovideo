import { useEffect, useRef, useState } from 'react';
import { getAssetUrl } from '../../config';

const peakCache = new Map<string, number[]>();

async function loadPeaks(url: string): Promise<number[]> {
    const cached = peakCache.get(url);
    if (cached) return cached;

    const response = await fetch(getAssetUrl(url) || url);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const buffer = await response.arrayBuffer();
    const AudioContextCtor = window.AudioContext
        || (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!AudioContextCtor) throw new Error('AudioContext unavailable');
    const context = new AudioContextCtor();
    try {
        const audio = await context.decodeAudioData(buffer.slice(0));
        const samples = audio.getChannelData(0);
        const buckets = 600;
        const size = Math.max(1, Math.floor(samples.length / buckets));
        const peaks: number[] = [];
        for (let bucket = 0; bucket < buckets; bucket += 1) {
            let peak = 0;
            const start = bucket * size;
            for (let index = start; index < start + size && index < samples.length; index += 1) {
                peak = Math.max(peak, Math.abs(samples[index]));
            }
            peaks.push(peak);
        }
        peakCache.set(url, peaks);
        return peaks;
    } finally {
        void context.close();
    }
}

export function Waveform({ url, height = 26 }: { url: string; height?: number }) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [peaks, setPeaks] = useState<number[] | null>(peakCache.get(url) || null);

    useEffect(() => {
        let alive = true;
        if (peaks) return () => { alive = false; };
        loadPeaks(url)
            .then((data) => { if (alive) setPeaks(data); })
            .catch(() => { if (alive) setPeaks(null); });
        return () => { alive = false; };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [url]);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas || !peaks) return;
        const width = canvas.clientWidth || 240;
        const dpr = window.devicePixelRatio || 1;
        canvas.width = Math.max(1, Math.round(width * dpr));
        canvas.height = Math.max(1, Math.round(height * dpr));
        const context = canvas.getContext('2d');
        if (!context) return;
        context.scale(dpr, dpr);
        context.clearRect(0, 0, width, height);
        context.fillStyle = 'rgba(61, 220, 151, 0.45)';
        const step = width / peaks.length;
        for (let index = 0; index < peaks.length; index += 1) {
            const amplitude = Math.max(0.04, peaks[index]) * (height / 2 - 1);
            context.fillRect(index * step, height / 2 - amplitude, Math.max(1, step), amplitude * 2);
        }
    }, [height, peaks]);

    return (
        <canvas
            ref={canvasRef}
            className="st-waveform"
            style={{ width: '100%', height }}
            aria-hidden="true"
        />
    );
}
