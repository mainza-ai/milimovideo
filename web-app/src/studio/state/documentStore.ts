import { useShallow } from 'zustand/react/shallow';
import { useTimelineStore } from '../../stores/timelineStore';

/**
 * Document-store boundary for the studio.
 *
 * The legacy timeline store is still authoritative during the strangler
 * migration. Studio code reads and writes through this adapter so the eventual
 * swap to a dedicated document store is a single-file change.
 */
export function useStudioDocument() {
    return useTimelineStore(useShallow((state) => ({
        project: state.project,
        selectedShotId: state.selectedShotId,
        updateShot: state.updateShot,
        patchShot: state.patchShot,
        addShot: state.addShot,
        deleteShot: state.deleteShot,
        splitShot: state.splitShot,
        reorderShots: state.reorderShots,
        setProject: state.setProject,
        saveProject: state.saveProject,
        saveStatus: state.saveStatus,
        saveError: state.saveError,
    })));
}

export const studioDocument = {
    get project() {
        return useTimelineStore.getState().project;
    },
    updateShot: (...args: Parameters<ReturnType<typeof useTimelineStore.getState>['updateShot']>) =>
        useTimelineStore.getState().updateShot(...args),
    saveProject: () => useTimelineStore.getState().saveProject(),
};
