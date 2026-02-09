import { useTimelineStore } from '../stores/timelineStore';
import { getAssetUrl } from '../config';

/**
 * Syncs the status of a specific job ONCE.
 * Used on page load to recover state if the user refreshed during generation.
 * Real-time updates are handled by SSE.
 */
export const pollJobStatus = async (jobId: string, shotId: string) => {
    console.log(`[Sync] Checking status for job ${jobId}...`);
    try {
        const res = await fetch(`http://localhost:8000/status/${jobId}`);
        if (!res.ok) return;

        const statusData = await res.json();
        const { updateShot, triggerAssetRefresh } = useTimelineStore.getState();

        if (statusData.status === 'completed') {
            const updates: any = {
                isGenerating: false,
                videoUrl: getAssetUrl(statusData.video_url),
                progress: 100,
                enhancedPromptResult: statusData.enhanced_prompt
            };
            if (statusData.thumbnail_url) updates.thumbnailUrl = getAssetUrl(statusData.thumbnail_url);
            if (statusData.actual_frames) updates.numFrames = statusData.actual_frames;

            updateShot(shotId, updates);
            triggerAssetRefresh();

        } else if (statusData.status === 'failed') {
            updateShot(shotId, { isGenerating: false, progress: 0 });
            console.error(`[Sync] Job ${jobId} failed: ${statusData.error}`);
        } else {
            // Still processing: Ensure UI shows "Generating" but DO NOT start a poll loop.
            // SSE will pick up the next event.
            // We just set the initial progress from the sync.
            const updates: any = {
                isGenerating: true,
                progress: statusData.progress || 0
            };
            if (statusData.status_message) updates.statusMessage = statusData.status_message;
            if (statusData.eta_seconds) updates.etaSeconds = statusData.eta_seconds;
            if (statusData.enhanced_prompt) updates.enhancedPromptResult = statusData.enhanced_prompt;

            updateShot(shotId, updates);
        }
    } catch (e) {
        console.error("Sync error", e);
    }
};
