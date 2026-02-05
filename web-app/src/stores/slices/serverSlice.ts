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
                if (data.message) updates.statusMessage = data.message;
                if (data.status) updates.statusMessage = data.status === 'processing' ? (data.message || 'Processing...') : data.status;
                updateShot(shot.id, updates);
            }
        } else if (type === 'complete') {
            const shot = project.shots.find(s => s.lastJobId === data.job_id);
            if (shot) {
                updateShot(shot.id, {
                    isGenerating: false,
                    progress: 100,
                    videoUrl: getAssetUrl(data.url),
                    thumbnailUrl: getAssetUrl(data.thumbnail_url),
                    numFrames: data.actual_frames || shot.numFrames,
                    statusMessage: "Complete"
                });
                triggerAssetRefresh();
                addToast("Generation Complete", "success");
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
        }
    },
});
