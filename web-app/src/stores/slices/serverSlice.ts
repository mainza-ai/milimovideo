import type { StateCreator } from 'zustand';
import type { TimelineState, ServerSlice } from '../types';
import { getAssetUrl } from '../../config';

export const createServerSlice: StateCreator<TimelineState, [], [], ServerSlice> = (_set, get) => ({
    handleServerEvent: (type: string, data: any) => {
        const { updateShot, triggerAssetRefresh, project, addToast } = get();

        if (type === 'progress') {
            const shot = project.shots.find(s => s.lastJobId === data.job_id);
            if (shot) {
                const updates: any = {
                    progress: data.progress,
                    isGenerating: true
                };
                if (data.enhanced_prompt) {
                    updates.enhancedPromptResult = data.enhanced_prompt;
                }
                if (data.message) updates.statusMessage = data.message;
                if (data.status) updates.statusMessage = data.status === 'processing' ? (data.message || 'Processing...') : data.status;
                updateShot(shot.id, updates);
            }
        } else if (type === 'complete') {
            // Match by lastJobId first, then fallback to shot_id for thumbnails
            let shot = project.shots.find(s => s.lastJobId === data.job_id);
            if (!shot && data.shot_id) {
                shot = project.shots.find(s => s.id === data.shot_id);
            }
            if (shot) {
                const isThumbnail = data.type === 'thumbnail';
                const isInpaint = data.type === 'inpaint';
                const updates: any = {
                    isGenerating: false,
                    progress: 100,
                    statusMessage: "Complete"
                };
                // Thumbnail events only update thumbnailUrl, not videoUrl
                if (isThumbnail) {
                    if (data.url) updates.thumbnailUrl = getAssetUrl(data.url);
                } else {
                    if (data.url) updates.videoUrl = getAssetUrl(data.url);
                    if (data.thumbnail_url) updates.thumbnailUrl = getAssetUrl(data.thumbnail_url);
                }
                if (data.actual_frames) {
                    updates.numFrames = data.actual_frames;
                }
                updateShot(shot.id, updates);
                triggerAssetRefresh();
                addToast(
                    isThumbnail ? "Concept Art Ready" : isInpaint ? "In-Painting Complete" : "Generation Complete",
                    "success"
                );
            }
        } else if (type === 'error') {
            const shot = project.shots.find(s => s.lastJobId === data.job_id);
            if (shot) {
                updateShot(shot.id, {
                    isGenerating: false,
                    statusMessage: "Failed",
                    progress: 0
                });
                addToast(`Generation Failed: ${data.message}`, "error");
            }

            // ── Render/Export Events ──────────────────────────────────────
        } else if (type === 'render_progress') {
            addToast(`Export: ${data.message} (${data.progress}%)`, "info");

        } else if (type === 'render_complete') {
            const downloadUrl = `http://localhost:8000${data.video_url}`;
            addToast("Export Complete! Opening video...", "success");
            window.open(downloadUrl, '_blank');

        } else if (type === 'render_failed') {
            addToast(`Export Failed: ${data.error}`, "error");
        }
    },
});
