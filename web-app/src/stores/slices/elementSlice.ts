import type { StateCreator } from 'zustand';
import type { TimelineState, ElementSlice, StoryElement, ParsedScene } from '../types';

export const createElementSlice: StateCreator<TimelineState, [], [], ElementSlice> = (set, get) => ({
    elements: [] as StoryElement[],
    generatingElementIds: {},

    fetchElements: async (projectId) => {
        try {
            // The backend returns snake_case, frontend uses camelCase for triggerWord
            const res = await fetch(`http://localhost:8000/projects/${projectId}/elements`);
            const data = await res.json();
            const elements = data.map((e: any) => ({
                ...e,
                triggerWord: e.trigger_word // map backend to frontend
            }));
            set({ elements });
        } catch (e) {
            console.error("Failed to fetch elements", e);
        }
    },

    createElement: async (projectId, elementData) => {
        try {
            const payload = {
                name: elementData.name,
                type: elementData.type,
                description: elementData.description,
                trigger_word: elementData.triggerWord // backend expects snake_case
            };
            const res = await fetch(`http://localhost:8000/projects/${projectId}/elements`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            if (res.ok) {
                // Refresh list
                get().fetchElements(projectId);
                get().addToast("Element created", "success");
            }
        } catch (e) {
            get().addToast("Failed to create element", "error");
        }
    },

    deleteElement: async (elementId) => {
        try {
            const res = await fetch(`http://localhost:8000/elements/${elementId}`, { method: 'DELETE' });
            if (res.ok) {
                const { elements } = get();
                set({ elements: elements.filter(e => e.id !== elementId) });
                get().addToast("Element deleted", "info");
            }
        } catch (e) { console.error(e); }
    },

    generateVisual: async (elementId, promptOverride, guidanceOverride = 2.0, enableAeOverride = false) => {
        const { addToast, fetchElements, project } = get();

        addToast("Queuing Visual Generation...", "info");
        try {
            const res = await fetch(`http://localhost:8000/elements/${elementId}/visualize`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prompt_override: promptOverride,
                    guidance_scale: guidanceOverride,
                    enable_ae: enableAeOverride
                })
            });

            if (!res.ok) throw new Error("Generation failed");

            const data = await res.json();
            const jobId = data.job_id;

            // Set loading state with Job ID
            set(state => ({
                generatingElementIds: { ...state.generatingElementIds, [elementId]: jobId }
            }));
            addToast("Generation Queued", "success");

            // Start Polling 
            const poll = async () => {
                try {
                    const statusRes = await fetch(`http://localhost:8000/status/${jobId}`);
                    if (!statusRes.ok) return; // Retry?
                    const status = await statusRes.json();

                    if (status.status === 'completed') {
                        addToast("Visual Generated!", "success");
                        await fetchElements(project.id);
                        // Clear loading
                        set(state => {
                            const newMap = { ...state.generatingElementIds };
                            delete newMap[elementId];
                            return { generatingElementIds: newMap };
                        });
                    } else if (status.status === 'failed' || status.status === 'cancelled') {
                        addToast(`Generation ${status.status}: ${status.status_message || ''}`, "error");
                        set(state => {
                            const newMap = { ...state.generatingElementIds };
                            delete newMap[elementId];
                            return { generatingElementIds: newMap };
                        });
                    } else {
                        // Still running
                        setTimeout(poll, 1000);
                    }
                } catch (e) {
                    console.error("Poll error", e);
                    // Don't clear immediately on poll error, retry mostly?
                    setTimeout(poll, 2000);
                }
            };
            poll();

        } catch (e) {
            console.error(e);
            addToast("Failed to queue visual", "error");
            set(state => {
                const newMap = { ...state.generatingElementIds };
                delete newMap[elementId];
                return { generatingElementIds: newMap };
            });
        }
    },

    cancelElementGeneration: async (elementId) => {
        const { generatingElementIds, addToast } = get();
        const jobId = generatingElementIds[elementId];
        if (!jobId || jobId === "true") return; // Can't cancel if no ID

        try {
            await fetch(`http://localhost:8000/jobs/${jobId}/cancel`, { method: 'POST' });
            addToast("Cancelling generation...", "info");
            // The poller will catch the 'cancelled' status and clean up
        } catch (e) {
            console.error(e);
            addToast("Failed to cancel", "error");
        }
    },

    parseScript: async (text: string) => {
        try {
            const project_id = get().project.id;
            const res = await fetch(`http://localhost:8000/projects/${project_id}/script/parse`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ script_text: text })
            });
            const data = await res.json();
            return data.scenes as ParsedScene[];
        } catch (e) {
            console.error("Parse failed", e);
            get().addToast("Failed to parse script", "error");
            return [];
        }
    },

    commitStoryboard: async (scenes: ParsedScene[], scriptText?: string) => {
        const { project, addToast, loadProject } = get();
        try {
            // Map ParsedScene format to commit payload, including matched_elements
            const payload = scenes.map((scene: any) => ({
                name: scene.name,
                content: scene.content,
                shots: (scene.shots || []).map((shot: any) => ({
                    action: shot.action,
                    dialogue: shot.dialogue,
                    character: shot.character,
                    shot_type: shot.shot_type,
                    matched_elements: shot.matched_elements || undefined,
                })),
            }));
            const res = await fetch(`http://localhost:8000/projects/${project.id}/storyboard/commit`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    scenes: payload,
                    script_text: scriptText
                })
            });
            if (!res.ok) throw new Error("Commit failed");

            addToast("Storyboard saved!", "success");
            await loadProject(project.id);
        } catch (e) {
            console.error("Commit failed", e);
            addToast("Failed to commit storyboard", "error");
        }
    },

    aiParseScript: async (text: string) => {
        try {
            const project_id = get().project.id;
            const res = await fetch(`http://localhost:8000/projects/${project_id}/storyboard/ai-parse`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ script_text: text })
            });
            const data = await res.json();
            if (data.mode === 'fallback') {
                get().addToast("AI unavailable — used standard parser", "info");
            }
            return data.scenes as ParsedScene[];
        } catch (e) {
            console.error("AI parse failed", e);
            get().addToast("AI parse failed", "error");
            return [];
        }
    },

    updateSceneName: async (sceneId: string, name: string) => {
        const { project } = get();
        try {
            const res = await fetch(`http://localhost:8000/projects/${project.id}/storyboard/scenes/${sceneId}`, {
                method: 'PATCH',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name })
            });
            if (!res.ok) throw new Error("Scene rename failed");
        } catch (e) {
            console.error("Scene rename failed", e);
            get().addToast("Failed to rename scene", "error");
        }
    },
});
