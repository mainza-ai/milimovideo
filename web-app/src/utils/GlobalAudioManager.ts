export class GlobalAudioManager {
    private static instance: GlobalAudioManager;
    private audioPool: Map<string, HTMLAudioElement> = new Map();

    private constructor() { }

    public static getInstance(): GlobalAudioManager {
        if (!GlobalAudioManager.instance) {
            GlobalAudioManager.instance = new GlobalAudioManager();
            // Attach to window for debugging or strict single instance check
            (window as any).__MilimoAudioManager = GlobalAudioManager.instance;
        }
        return GlobalAudioManager.instance;
    }

    /**
     * Syncs the audio pool with the current list of clips.
     * Removes unused audio elements (strict pause & cleanup).
     * Creates new audio elements for new clips.
     */
    public sync(clips: any[]) {
        const currentIds = new Set(clips.map(c => c.id));

        // 1. Cleanup removed
        for (const [id, audio] of this.audioPool) {
            if (!currentIds.has(id)) {
                console.log(`[Audio] Cleaning up ${id}`);
                audio.pause();
                audio.src = ''; // Detach source
                audio.remove(); // Remove from DOM if attached (usually not)
                this.audioPool.delete(id);
            }
        }

        // 2. Add/Update
        clips.forEach(clip => {
            if (!clip.videoUrl) return; // Audio clip must have URL

            let audio = this.audioPool.get(clip.id);
            const url = clip.videoUrl.startsWith('http') ? clip.videoUrl : `http://localhost:8000${clip.videoUrl}`;

            if (!audio) {
                console.log(`[Audio] Creating new audio for ${clip.id}`);
                audio = new Audio(url);
                audio.preload = 'auto';
                audio.loop = false; // Strictly no looping
                this.audioPool.set(clip.id, audio);
            } else if (audio.src !== url && !audio.src.endsWith(url)) {
                // URL changed
                console.log(`[Audio] Updating URL for ${clip.id}`);
                audio.pause();
                audio.src = url;
                audio.loop = false;
            }
        });
    }

    /**
     * Ticks the audio engine for a defined time window.
     * @param currentTime The global timeline time
     * @param fps Project FPS
     * @param isPlaying Whether the timeline is playing
     * @param isMuted Whether the audio track is muted
     */
    public tick(currentTime: number, fps: number, isPlaying: boolean, isMuted: boolean, clips: any[]) {
        clips.forEach(clip => {
            const audio = this.audioPool.get(clip.id);
            if (!audio) return;

            const startTime = (clip.startFrame || 0) / fps;
            const duration = clip.numFrames / fps;
            const endTime = startTime + duration;
            const trimInTime = (clip.trimIn || 0) / fps;

            audio.volume = isMuted ? 0 : 1;

            if (currentTime >= startTime && currentTime < endTime) {
                // Inside Clip
                const targetAudioTime = (currentTime - startTime) + trimInTime;

                // Sync drift
                if (Math.abs(audio.currentTime - targetAudioTime) > 0.15) {
                    if (Number.isFinite(audio.duration)) {
                        audio.currentTime = targetAudioTime;
                    }
                }

                if (isPlaying && audio.paused) {
                    audio.play().catch(e => {
                        if (e.name !== 'AbortError') console.warn("Audio play failed", e);
                    });
                } else if (!isPlaying && !audio.paused) {
                    audio.pause();
                }
            } else {
                // Outside Clip
                if (!audio.paused) {
                    audio.pause();
                }
            }
        });
    }

    /**
     * Stop all audio immediately.
     */
    public stopAll() {
        this.audioPool.forEach(a => {
            a.pause();
        });
    }

    /**
     * Destroy everything.
     */
    public cleanup() {
        this.stopAll();
        this.audioPool.forEach(a => {
            a.src = '';
            a.remove();
        });
        this.audioPool.clear();
    }
}
