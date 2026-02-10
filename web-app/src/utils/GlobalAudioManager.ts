/**
 * GlobalAudioManager — Web Audio API Implementation
 * 
 * Uses AudioContext + AudioBufferSourceNode instead of HTMLAudioElement
 * for reliable cross-browser playback, especially Safari which has
 * strict autoplay policies and unreliable HTMLAudioElement.currentTime seeking.
 */

interface AudioEntry {
    buffer: AudioBuffer | null;
    sourceNode: AudioBufferSourceNode | null;
    gainNode: GainNode | null;
    isPlaying: boolean;
    startedAtContextTime: number; // AudioContext.currentTime when playback started
    startedAtOffset: number;     // Offset into the buffer when playback started
    url: string;
    loading: boolean;
}

export class GlobalAudioManager {
    private static instance: GlobalAudioManager;
    private audioContext: AudioContext | null = null;
    private entries: Map<string, AudioEntry> = new Map();
    private contextResumed: boolean = false;

    private constructor() { }

    public static getInstance(): GlobalAudioManager {
        if (!GlobalAudioManager.instance) {
            GlobalAudioManager.instance = new GlobalAudioManager();
            (window as any).__MilimoAudioManager = GlobalAudioManager.instance;
        }
        return GlobalAudioManager.instance;
    }

    /**
     * Lazily initialize AudioContext. Must be called after a user gesture on Safari.
     */
    private ensureContext(): AudioContext {
        if (!this.audioContext) {
            this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
        }
        return this.audioContext;
    }

    /**
     * Resume AudioContext if suspended (Safari suspends until user interaction).
     */
    private async resumeIfNeeded(): Promise<void> {
        const ctx = this.ensureContext();
        if (ctx.state === 'suspended') {
            try {
                await ctx.resume();
                this.contextResumed = true;
            } catch (e) {
                console.warn('[Audio] Failed to resume AudioContext', e);
            }
        }
    }

    /**
     * Fetch and decode an audio file into an AudioBuffer.
     */
    private async loadBuffer(url: string): Promise<AudioBuffer | null> {
        const ctx = this.ensureContext();
        try {
            const response = await fetch(url, { mode: 'cors' });
            const arrayBuffer = await response.arrayBuffer();
            const audioBuffer = await ctx.decodeAudioData(arrayBuffer);
            return audioBuffer;
        } catch (e) {
            console.warn(`[Audio] Failed to load/decode ${url}`, e);
            return null;
        }
    }

    /**
     * Syncs the audio pool with the current list of clips.
     * Removes unused entries. Creates new entries for new clips.
     */
    public sync(clips: any[]) {
        const currentIds = new Set(clips.map(c => c.id));

        // 1. Cleanup removed
        for (const [id, entry] of this.entries) {
            if (!currentIds.has(id)) {
                console.log(`[Audio] Cleaning up ${id}`);
                this.stopEntry(entry);
                this.entries.delete(id);
            }
        }

        // 2. Add/Update
        clips.forEach(clip => {
            if (!clip.videoUrl) return;

            const url = clip.videoUrl.startsWith('http') ? clip.videoUrl : `http://localhost:8000${clip.videoUrl}`;

            let entry = this.entries.get(clip.id);

            if (!entry) {
                // New clip
                entry = {
                    buffer: null,
                    sourceNode: null,
                    gainNode: null,
                    isPlaying: false,
                    startedAtContextTime: 0,
                    startedAtOffset: 0,
                    url,
                    loading: false,
                };
                this.entries.set(clip.id, entry);

                // Start loading buffer
                entry.loading = true;
                this.loadBuffer(url).then(buf => {
                    // Check entry still exists (might have been removed during load)
                    const current = this.entries.get(clip.id);
                    if (current && current.url === url) {
                        current.buffer = buf;
                        current.loading = false;
                    }
                });
            } else if (entry.url !== url) {
                // URL changed — reload
                console.log(`[Audio] Updating URL for ${clip.id}`);
                this.stopEntry(entry);
                entry.url = url;
                entry.buffer = null;
                entry.loading = true;
                this.loadBuffer(url).then(buf => {
                    const current = this.entries.get(clip.id);
                    if (current && current.url === url) {
                        current.buffer = buf;
                        current.loading = false;
                    }
                });
            }
        });
    }

    /**
     * Start playback of an entry at a specific offset.
     */
    private startEntry(entry: AudioEntry, offset: number) {
        if (!entry.buffer || entry.isPlaying) return;

        const ctx = this.ensureContext();

        // Create fresh source node (they are single-use in Web Audio API)
        const sourceNode = ctx.createBufferSource();
        sourceNode.buffer = entry.buffer;

        const gainNode = ctx.createGain();
        sourceNode.connect(gainNode);
        gainNode.connect(ctx.destination);

        // Clamp offset to valid range
        const safeOffset = Math.max(0, Math.min(offset, entry.buffer.duration));

        sourceNode.start(0, safeOffset);

        entry.sourceNode = sourceNode;
        entry.gainNode = gainNode;
        entry.isPlaying = true;
        entry.startedAtContextTime = ctx.currentTime;
        entry.startedAtOffset = safeOffset;

        // Auto-cleanup when playback ends
        sourceNode.onended = () => {
            entry.isPlaying = false;
            entry.sourceNode = null;
            entry.gainNode = null;
        };
    }

    /**
     * Stop playback of an entry.
     */
    private stopEntry(entry: AudioEntry) {
        if (entry.sourceNode) {
            try {
                entry.sourceNode.stop();
            } catch (e) {
                // May already be stopped
            }
            entry.sourceNode.disconnect();
            entry.sourceNode = null;
        }
        if (entry.gainNode) {
            entry.gainNode.disconnect();
            entry.gainNode = null;
        }
        entry.isPlaying = false;
    }

    /**
     * Get the current playback position within the audio buffer.
     */
    private getEntryCurrentTime(entry: AudioEntry): number {
        if (!entry.isPlaying || !this.audioContext) return 0;
        return entry.startedAtOffset + (this.audioContext.currentTime - entry.startedAtContextTime);
    }

    /**
     * Ticks the audio engine for a defined time window.
     */
    public tick(currentTime: number, fps: number, isPlaying: boolean, isMuted: boolean, clips: any[]) {
        // Resume context on first play attempt (user gesture requirement)
        if (isPlaying && !this.contextResumed) {
            this.resumeIfNeeded();
        }

        clips.forEach(clip => {
            const entry = this.entries.get(clip.id);
            if (!entry || !entry.buffer) return;

            const startTime = (clip.startFrame || 0) / fps;
            const duration = clip.numFrames / fps;
            const endTime = startTime + duration;
            const trimInTime = (clip.trimIn || 0) / fps;

            // Set volume
            if (entry.gainNode) {
                entry.gainNode.gain.value = isMuted ? 0 : 1;
            }

            if (currentTime >= startTime && currentTime < endTime) {
                // Inside Clip
                const targetAudioTime = (currentTime - startTime) + trimInTime;

                if (isPlaying) {
                    if (!entry.isPlaying) {
                        // Start playback at correct offset
                        this.startEntry(entry, targetAudioTime);
                    } else {
                        // Check drift (0.3s tolerance — relaxed for Safari)
                        const actualTime = this.getEntryCurrentTime(entry);
                        if (Math.abs(actualTime - targetAudioTime) > 0.3) {
                            // Re-sync: Stop and restart at correct position
                            this.stopEntry(entry);
                            this.startEntry(entry, targetAudioTime);
                        }
                    }
                } else {
                    // Paused
                    if (entry.isPlaying) {
                        this.stopEntry(entry);
                    }
                }
            } else {
                // Outside Clip
                if (entry.isPlaying) {
                    this.stopEntry(entry);
                }
            }
        });
    }

    /**
     * Stop all audio immediately.
     */
    public stopAll() {
        this.entries.forEach(entry => this.stopEntry(entry));
    }

    /**
     * Destroy everything.
     */
    public cleanup() {
        this.stopAll();
        this.entries.clear();
        if (this.audioContext) {
            this.audioContext.close().catch(() => { });
            this.audioContext = null;
        }
        this.contextResumed = false;
    }
}
