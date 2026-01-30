import { useTimelineStore } from '../stores/timelineStore';

export const pollJobStatus = async (jobId: string, shotId: string) => {
    const poll = async () => {
        try {
            const res = await fetch(`http://localhost:8000/status/${jobId}`);
            if (!res.ok) throw new Error("Status fetch failed");

            const statusData = await res.json();
            const { updateShot, triggerAssetRefresh } = useTimelineStore.getState();

            if (statusData.status === 'completed') {
                const updates: any = {
                    isGenerating: false,
                    videoUrl: statusData.video_url,
                    progress: 100,
                    enhancedPromptResult: statusData.enhanced_prompt,
                    thumbnailUrl: statusData.thumbnail_url
                };

                if (statusData.actual_frames) {
                    updates.numFrames = statusData.actual_frames;
                }

                updateShot(shotId, updates);
                triggerAssetRefresh();

            } else if (statusData.status === 'failed') {
                updateShot(shotId, { isGenerating: false, progress: 0 });
                // We show alert in UI? Or just log? 
                // Store doesn't have "error" field shown in UI.
                // InspectorPanel showed alert. 
                // We can add error field to updates if we want UI to show it.
                // For now, console log.
                console.error(`Generation Failed: ${statusData.error || 'Unknown error'}`);

            } else {
                // Still processing
                // Still processing
                if (statusData.progress !== undefined) {
                    const { project } = useTimelineStore.getState();
                    const currentShot = project.shots.find(s => s.id === shotId);
                    const currentProgress = currentShot?.progress || 0;

                    // Prevent flickering to 0% if we are already progressed
                    // Only allow 0 if current is also low, or if we really want to support restart.
                    // For now, simple monotonicity check: if new < old, ignore unless new is 0 (maybe restart) AND old is 100? 
                    // Actually, getting 0 after 20% is the bug.
                    // So if new < current, ignore it.
                    if (statusData.progress >= currentProgress) {
                        const updates: any = { progress: statusData.progress };
                        if (statusData.enhanced_prompt) {
                            updates.enhancedPromptResult = statusData.enhanced_prompt;
                        }
                        if (statusData.status_message) {
                            updates.statusMessage = statusData.status_message;
                        }
                        if (statusData.current_prompt) {
                            updates.currentPrompt = statusData.current_prompt;
                        }
                        if (statusData.eta_seconds !== undefined && statusData.eta_seconds !== null) {
                            updates.etaSeconds = statusData.eta_seconds;
                        }
                        updateShot(shotId, updates);
                    }
                }

                // Continue polling
                setTimeout(poll, 1000);
            }
        } catch (e) {
            console.error("Polling error", e);
            setTimeout(poll, 2000); // Retry slower
        }
    };
    poll();
};
