import { useEffect, useState } from 'react';
import { useTimelineStore } from '../../stores/timelineStore';
import { useShallow } from 'zustand/react/shallow';
import { Loader2, Zap, Image as ImageIcon, Wand2 } from 'lucide-react';

interface GeneratedImage {
    id: string;
    url: string;
    path: string;
    filename: string;
    width: number;
    height: number;
    created_at?: string;
    meta_json?: string;
}

export const ImagesView = () => {
    const { project, addToast, elements, fetchElements, triggerAssetRefresh } = useTimelineStore(useShallow(state => ({
        project: state.project,
        addToast: state.addToast,
        elements: state.elements,
        fetchElements: state.fetchElements,
        triggerAssetRefresh: state.triggerAssetRefresh
    })));
    const [images, setImages] = useState<GeneratedImage[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [selectedImage, setSelectedImage] = useState<GeneratedImage | null>(null);

    // Form State
    const [prompt, setPrompt] = useState("");
    const [negativePrompt, setNegativePrompt] = useState("");
    const [width, setWidth] = useState(1024);
    const [height, setHeight] = useState(1024);
    const [steps, setSteps] = useState(25);
    const [guidance, setGuidance] = useState(2.0);
    const [seed, setSeed] = useState<string>(""); // user input as string, send as int or null
    const [selectedReferences, setSelectedReferences] = useState<string[]>([]); // IP-Adapter element IDs
    const [enableAE, setEnableAE] = useState(false);
    const [enableTrueCFG, setEnableTrueCFG] = useState(false);
    const [progress, setProgress] = useState(0);
    const [statusMsg, setStatusMsg] = useState("");
    const [currentJobId, setCurrentJobId] = useState<string | null>(null);

    const handleCancel = async () => {
        if (!currentJobId) return;
        try {
            await fetch(`http://localhost:8000/jobs/${currentJobId}/cancel`, { method: 'POST' });
            addToast("Cancelling...", "info");
            // Optimistic update
            setIsLoading(false);
            setStatusMsg("Cancelled");
            setCurrentJobId(null);
        } catch (e) {
            console.error("Cancel failed", e);
        }
    };

    useEffect(() => {
        if (project.id) {
            fetchImages();
            fetchElements(project.id);
            checkActiveJobs();
        }
    }, [project.id]);

    const checkActiveJobs = async () => {
        try {
            const res = await fetch(`http://localhost:8000/projects/${project.id}/active_jobs`);
            if (res.ok) {
                const jobs = await res.json();
                // Find first active image job
                const activeJob = jobs.find((j: any) => j.type === 'image' && (j.status === 'processing' || j.status === 'pending'));
                if (activeJob) {
                    console.log("Resuming active job:", activeJob.job_id);
                    startPolling(activeJob.job_id);
                }
            }
        } catch (e) {
            console.error("Failed to check active jobs", e);
        }
    };

    const startPolling = (jobId: string) => {
        setCurrentJobId(jobId);
        setIsLoading(true);
        // If we don't have status message yet, use a default
        // But the first poll will fix it soon.

        const poll = async () => {
            const statusRes = await fetch(`http://localhost:8000/status/${jobId}`);
            if (statusRes.ok) {
                const job = await statusRes.json();
                if (job.status === 'completed') {
                    setIsLoading(false);
                    setCurrentJobId(null);
                    addToast("Image Ready!", "success");
                    triggerAssetRefresh();

                    // Refresh images
                    const res = await fetch(`http://localhost:8000/projects/${project.id}/images`);
                    if (res.ok) {
                        const data = await res.json();
                        setImages(data);
                        const newImage = data.find((img: any) => img.id === job.asset_id);
                        if (newImage) setSelectedImage(newImage);
                    }
                    return;
                } else if (job.status === 'failed' || job.status === 'cancelled') {
                    setIsLoading(false);
                    setCurrentJobId(null);
                    addToast(job.status === 'failed' ? `Failed: ${job.error_message}` : "Cancelled", job.status === 'failed' ? "error" : "info");
                    return;
                } else {
                    if (job.progress !== undefined) setProgress(job.progress);
                    if (job.status_message) setStatusMsg(job.status_message);
                    setTimeout(poll, 1000);
                }
            } else {
                // Job might be gone from memory if server restart, check DB logic handling in status endpoint handles it?
                // The status endpoint checks DB too. If 404, stop polling.
                if (statusRes.status === 404) {
                    setIsLoading(false);
                    setCurrentJobId(null);
                } else {
                    setTimeout(poll, 1000);
                }
            }
        };
        poll();
    };

    const fetchImages = async () => {
        try {
            // Re-using assets endpoint but filtering client-side or we could make a specific one.
            // Actually, we can just use the project assets if we tag them correctly. 
            // The server saves them as type="image" in the Asset table.

            // Let's assume we can fetch project assets. 
            // Currently server has GET /projects/{id} which returns structure, but maybe not raw asset list.
            // Let's try GET /projects/{id}/assets if it exists, otherwise relying on list_dir logic or similar.

            // Wait, server.py doesn't have a specific "list all images" endpoint other than generic asset management.
            // Let's implement a direct fetch for now using existing generic endpoints or assumes we can list generated/images.

            // Actually, let's just make a quick auxiliary fetch using the same pattern as "Elements" 
            // but targeting the specific folder or filtering.
            // For now, let's rely on a manual scan since we don't have a dedicated "get all project images" API explicitly documented 
            // other than what's in project structure.

            // TEMPORARY: Generic asset fetch or we assume we can list the directory via a new endpoint?
            // Let's use the generic "get project" which usually returns everything if designed well, 
            // but Project model only links "Shots".

            // It seems we missed an endpoint to LIST generated images in the plan.
            // I'll implement a fallback here to just show what we generate in session 
            // OR ideally add a GET /projects/{id}/images endpoint.

            // For now, I will add a simple endpoint fetch here assuming I'll add it to server.py next 
            // OR use the existing "Elements" as a pattern.

            // Let's try to fetch from a new endpoint I'll add: GET /projects/{id}/images
            const res = await fetch(`http://localhost:8000/projects/${project.id}/images`);
            if (res.ok) {
                const data = await res.json();
                setImages(data);
            }
        } catch (e) {
            console.error("Failed to fetch images", e);
        }
    };

    const handleGenerate = async () => {
        setIsLoading(true);
        setProgress(0);
        setStatusMsg("Initializing...");
        try {
            const payload = {
                project_id: project.id,
                prompt,
                negative_prompt: negativePrompt,
                width,
                height,
                num_inference_steps: steps,
                guidance_scale: guidance,
                seed: seed ? parseInt(seed) : null,
                reference_images: selectedReferences,
                enable_ae: enableAE,
                enable_true_cfg: enableTrueCFG
            };

            // 1. Trigger Async Job
            const res = await fetch('http://localhost:8000/generate/image', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!res.ok) throw new Error("Generation request failed");
            const { job_id } = await res.json();

            setCurrentJobId(job_id);
            addToast("Generation Started...", "info");
            startPolling(job_id);

        } catch (e) {
            console.error(e);
            addToast("Generation Failed", "error");
            setIsLoading(false);
        }
    };

    const handleReset = () => {
        setPrompt("");
        setNegativePrompt("");
        setWidth(1024);
        setHeight(1024);
        setSteps(25);
        setGuidance(2.0);
        setSeed("");
        setSeed("");
        setSelectedReferences([]);
        setEnableAE(false);
        setEnableTrueCFG(false);
        setSelectedImage(null);
        addToast("Reset to defaults", "info");
    };

    return (
        <div className="flex h-full w-full bg-[#050505] text-white">

            {/* Left: Gallery */}
            <div className="flex-1 p-6 overflow-y-auto">
                <h2 className="text-xl font-bold mb-6 flex items-center gap-2">
                    <ImageIcon className="text-milimo-400" /> Image Gallery
                </h2>

                {images.length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-64 text-white/20 border-2 border-dashed border-white/5 rounded-2xl">
                        <ImageIcon size={48} className="mb-4 opacity-50" />
                        <p>No images generated yet</p>
                    </div>
                ) : (
                    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                        {images.map(img => (
                            <div
                                key={img.id}
                                draggable
                                onDragStart={(e) => {
                                    const data = {
                                        id: img.id,
                                        url: `http://localhost:8000${img.url}`,
                                        path: img.path,
                                        filename: img.filename,
                                        type: 'image',
                                        thumbnail: `http://localhost:8000${img.url}`
                                    };
                                    e.dataTransfer.setData('application/json', JSON.stringify(data));
                                    e.dataTransfer.effectAllowed = 'copy';
                                }}
                                onClick={() => {
                                    setSelectedImage(img);
                                    if (img.meta_json) {
                                        try {
                                            const meta = JSON.parse(img.meta_json);
                                            if (meta.prompt) setPrompt(meta.prompt);
                                            if (meta.negative_prompt) setNegativePrompt(meta.negative_prompt);
                                            if (meta.width) setWidth(meta.width);
                                            if (meta.height) setHeight(meta.height);
                                            if (meta.seed) setSeed(meta.seed.toString());
                                            if (meta.steps) setSteps(meta.steps);
                                            if (meta.guidance) setGuidance(meta.guidance);
                                            if (meta.seed) setSeed(meta.seed.toString());
                                            if (meta.steps) setSteps(meta.steps);
                                            if (meta.guidance) setGuidance(meta.guidance);
                                            if (meta.reference_elements && meta.reference_elements.length > 0) {
                                                setSelectedReferences(meta.reference_elements || []);
                                            }
                                            // Optional: Restore toggle state if saved in meta (future improvement)
                                        } catch (e) {
                                            console.error("Failed to parse image metadata", e);
                                        }
                                    }
                                }}
                                className={`aspect-square rounded-xl overflow-hidden cursor-pointer border-2 transition-all group relative ${selectedImage?.id === img.id ? 'border-milimo-500 shadow-lg shadow-milimo-500/20' : 'border-transparent hover:border-white/20'}`}
                            >
                                <img src={`http://localhost:8000${img.url}`} className="w-full h-full object-cover" loading="lazy" />
                                <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                                    <span className="text-xs font-mono">{img.width}x{img.height}</span>
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>

            {/* Center: Preview (Overlay or Modal? Actually let's make Center the Preview and Left the Grid... wait layout above was 3 cols?) 
               Actually reusing the Layout: 
               Layout has "Center" as children. 
               This component is rendered inside "Center".
               So this component controls the full center area.
               I'll split IT into 2 columns: Gallery (Left) and Inspector (Right).
               Wait, usually Inspector is global right. 
               But InspectorPanel is specific to timeline shots.
               So I should probably render my own inspector logic here or hijack the global one.
               Hijacking global is complex. I'll render a dedicated "Generation Panel" on the right side of THIS view.
            */}

            {/* Right: Inspector / Controls */}
            <div className="w-80 border-l border-white/5 bg-[#0a0a0a] p-6 flex flex-col gap-6 overflow-y-auto shrink-0">

                {/* Preview Selected Large */}
                {selectedImage && (
                    <div className="aspect-square w-full bg-black rounded-lg overflow-hidden border border-white/10 relative">
                        <img src={`http://localhost:8000${selectedImage.url}`} className="w-full h-full object-contain" />
                        <div className="absolute top-2 right-2">
                            <a href={`http://localhost:8000${selectedImage.url}`} target="_blank" download className="p-2 bg-black/50 rounded-md hover:bg-white/20 text-white">
                                <ImageIcon size={14} />
                            </a>
                        </div>
                    </div>
                )}

                <div className="space-y-4">
                    <div className="flex items-center justify-between">
                        <h3 className="font-bold text-white/90 flex items-center gap-2">
                            <Zap size={16} className="text-yellow-400" /> Image Generation
                        </h3>
                        <button
                            onClick={handleReset}
                            className="text-xs text-white/40 hover:text-white uppercase font-bold tracking-wider transition-colors"
                        >
                            Reset
                        </button>
                    </div>

                    {/* Prompt */}
                    <div className="space-y-2">
                        <label className="text-xs font-semibold text-white/50 uppercase">Prompt</label>
                        <textarea
                            className="w-full bg-black/20 border border-white/10 rounded-lg p-3 text-sm focus:border-milimo-500 focus:outline-none min-h-[100px] resize-y"
                            placeholder="Describe your image..."
                            value={prompt}
                            onChange={e => setPrompt(e.target.value)}
                        />
                    </div>

                    {/* Negative Prompt */}
                    <div className="space-y-2">
                        <div className="flex justify-between items-center">
                            <label className="text-xs font-semibold text-white/50 uppercase">Negative Prompt</label>
                            <button
                                onClick={() => setNegativePrompt("low quality, ugly, blurry, bad anatomy, text, watermark, deformed, disfigured")}
                                className="text-[10px] bg-white/5 hover:bg-white/10 px-2 py-0.5 rounded text-milimo-400 flex items-center gap-1 transition-colors"
                                title="Auto-fill recommended negative prompt"
                            >
                                <Wand2 size={10} /> Auto
                            </button>
                        </div>
                        <textarea
                            className="w-full bg-black/20 border border-white/10 rounded-lg p-3 text-sm focus:border-milimo-500 focus:outline-none min-h-[60px] resize-y"
                            placeholder="Things to avoid..."
                            value={negativePrompt}
                            onChange={e => setNegativePrompt(e.target.value)}
                        />
                    </div>

                    {/* Image Ref (IP-Adapter) */}
                    <div className="space-y-2">
                        <label className="text-xs font-semibold text-white/50 uppercase">Insert Element Trigger</label>
                        <select
                            className="w-full bg-black/20 border border-white/10 rounded-lg p-2 text-xs focus:border-milimo-500 outline-none"
                            value=""
                            onChange={e => {
                                const val = e.target.value;
                                if (!val) return;
                                const el = elements.find(e => e.id === val);
                                if (el && el.triggerWord) {
                                    setPrompt(prev => {
                                        if (prev.includes(el.triggerWord)) return prev;
                                        return prev + (prev.length > 0 ? " " : "") + el.triggerWord;
                                    });
                                    // Also add element ID for IP-Adapter visual conditioning
                                    setSelectedReferences(prev => {
                                        if (prev.includes(el.id)) return prev;
                                        return [...prev, el.id];
                                    });
                                    addToast(`Added ${el.triggerWord}`, "info");
                                }
                            }}
                        >
                            <option value="">Select to Insert...</option>
                            {elements.map(el => (
                                <option key={el.id} value={el.id}>{el.name} ({el.triggerWord})</option>
                            ))}
                        </select>
                        <p className="text-[10px] text-white/30">Select an element to add its trigger to the prompt.</p>
                    </div>

                    {/* Dimensions */}
                    <div className="grid grid-cols-2 gap-4">
                        <div className="space-y-1">
                            <label className="text-[10px] uppercase text-white/40">Width</label>
                            <input type="number" step={16} value={width} onChange={e => setWidth(parseInt(e.target.value))} className="w-full bg-white/5 border border-white/10 rounded p-1.5 text-xs" />
                        </div>
                        <div className="space-y-1">
                            <label className="text-[10px] uppercase text-white/40">Height</label>
                            <input type="number" step={16} value={height} onChange={e => setHeight(parseInt(e.target.value))} className="w-full bg-white/5 border border-white/10 rounded p-1.5 text-xs" />
                        </div>
                    </div>

                    {/* Advanced */}
                    <div className="grid grid-cols-2 gap-4">
                        <div className="space-y-1">
                            <label className="text-[10px] uppercase text-white/40">Steps</label>
                            <input type="number" value={steps} onChange={e => setSteps(parseInt(e.target.value))} className="w-full bg-white/5 border border-white/10 rounded p-1.5 text-xs" />
                        </div>
                        <div className="space-y-1">
                            <label className="text-[10px] uppercase text-white/40">Guidance</label>
                            <input type="number" step={0.1} value={guidance} onChange={e => setGuidance(parseFloat(e.target.value))} className="w-full bg-white/5 border border-white/10 rounded p-1.5 text-xs" />
                        </div>
                    </div>

                    {/* Toggles */}
                    <div className="space-y-3 pt-2 border-t border-white/5">
                        <label className="flex items-center gap-2 cursor-pointer group">
                            <input
                                type="checkbox"
                                checked={enableAE}
                                onChange={e => setEnableAE(e.target.checked)}
                                className="w-4 h-4 rounded border-white/20 bg-white/5 checked:bg-milimo-500 text-black focus:ring-milimo-500/50"
                            />
                            <div className="flex flex-col">
                                <span className="text-xs font-semibold text-white/80 group-hover:text-white">Use Native AE</span>
                                <span className="text-[10px] text-white/30">Better quality (ae.safetensors)</span>
                            </div>
                        </label>

                        <label className="flex items-center gap-2 cursor-pointer group">
                            <input
                                type="checkbox"
                                checked={enableTrueCFG}
                                onChange={e => setEnableTrueCFG(e.target.checked)}
                                className="w-4 h-4 rounded border-white/20 bg-white/5 checked:bg-milimo-500 text-black focus:ring-milimo-500/50"
                            />
                            <div className="flex flex-col">
                                <span className="text-xs font-semibold text-white/80 group-hover:text-white">Enable True CFG (Slower)</span>
                                <span className="text-[10px] text-white/30">Supports Negatives. Slider = CFG Scale.</span>
                            </div>
                        </label>
                    </div>

                    <div className="space-y-1">
                        <label className="text-[10px] uppercase text-white/40">Seed (Optional)</label>
                        <input type="number" placeholder="Random" value={seed} onChange={e => setSeed(e.target.value)} className="w-full bg-white/5 border border-white/10 rounded p-1.5 text-xs" />
                    </div>

                    {/* Button */}
                    <button
                        onClick={handleGenerate}
                        disabled={isLoading || !prompt}
                        className={`w-full h-12 rounded-xl font-bold text-sm tracking-wide uppercase transition-all flex items-center justify-center gap-2 overflow-hidden relative ${isLoading ? 'bg-white/5 text-white/50 cursor-not-allowed' : 'bg-milimo-500 text-black hover:bg-milimo-400 shadow-lg shadow-milimo-500/20'}`}
                    >
                        {isLoading ? (
                            <div className="absolute inset-0 flex items-center justify-center w-full h-full">
                                {/* Progress Bar Background */}
                                <div className="absolute left-0 top-0 bottom-0 bg-milimo-500/20 h-full transition-all duration-500" style={{ width: `${progress}%` }}></div>
                                {/* Content */}
                                <div className="z-10 flex items-center justify-between w-full px-4">
                                    <div className="flex items-center gap-2">
                                        <Loader2 className="animate-spin w-4 h-4" />
                                        <span>{statusMsg || `Generating ${progress}%`}</span>
                                    </div>
                                    {/* Cancel Button */}
                                    {currentJobId && (
                                        <div
                                            onClick={(e) => { e.stopPropagation(); handleCancel(); }}
                                            className="p-1 px-2 bg-red-500/20 hover:bg-red-500/40 text-red-200 text-xs rounded uppercase font-bold tracking-wider cursor-pointer"
                                        >
                                            Cancel
                                        </div>
                                    )}
                                </div>
                            </div>
                        ) : (
                            <><Zap size={18} fill="currentColor" /> Generate Image</>
                        )}
                    </button>

                </div>
            </div>
        </div>
    );
};
