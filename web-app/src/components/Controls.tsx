import { MediaUploader } from './MediaUploader';
import { Toggle } from './Toggle';
import { Sparkles, Video, Image as ImageIcon, Type, ArrowLeftRight } from 'lucide-react';
import clsx from 'clsx';

export type GenerationMode = 'txt2vid' | 'img2vid' | 'vid2vid' | 'keyframe';

interface ControlsProps {
    mode: GenerationMode;
    setMode: (mode: GenerationMode) => void;

    prompt: string;
    setPrompt: (value: string) => void;
    negPrompt: string;
    setNegPrompt: (value: string) => void;
    seed: number;
    setSeed: (value: number) => void;

    // Media Inputs
    inputImage: File | null;
    setInputImage: (file: File | null) => void;
    inputVideo: File | null;
    setInputVideo: (file: File | null) => void;
    startImage: File | null;
    setStartImage: (file: File | null) => void;
    endImage: File | null;
    setEndImage: (file: File | null) => void;

    // Advanced Settings
    strength: number;
    setStrength: (val: number) => void;
    numFrames: number;
    setNumFrames: (val: number) => void;
    width: number;
    setWidth: (val: number) => void;
    height: number;
    setHeight: (val: number) => void;
    enhancePrompt: boolean;
    setEnhancePrompt: (val: boolean) => void;
    continuationPrompt: string;
    setContinuationPrompt: (val: string) => void;
    autoContinue: boolean;
    setAutoContinue: (val: boolean) => void;

    onGenerate: () => void;
    onStop: () => void;
    isGenerating: boolean;
}

export function Controls({
    mode, setMode,
    prompt, setPrompt,
    negPrompt, setNegPrompt,
    seed, setSeed,
    inputImage, setInputImage,
    inputVideo, setInputVideo,
    startImage, setStartImage,
    endImage, setEndImage,
    strength, setStrength,
    numFrames, setNumFrames,
    width, setWidth,
    height, setHeight,
    enhancePrompt, setEnhancePrompt,
    continuationPrompt, setContinuationPrompt,
    autoContinue, setAutoContinue,
    onGenerate, onStop, isGenerating
}: ControlsProps) {

    const tabs = [
        { id: 'txt2vid' as const, label: 'Text', icon: Type },
        { id: 'img2vid' as const, label: 'Image', icon: ImageIcon },
        { id: 'vid2vid' as const, label: 'Video', icon: Video },
        { id: 'keyframe' as const, label: 'Interp', icon: ArrowLeftRight },
    ];

    return (
        <div className="flex flex-col gap-6 pb-20">

            {/* Mode Tabs */}
            <div className="flex p-1 bg-black/40 backdrop-blur-md rounded-xl border border-white/5">
                {tabs.map((tab) => {
                    const Icon = tab.icon;
                    const isActive = mode === tab.id;
                    return (
                        <button
                            key={tab.id}
                            onClick={() => setMode(tab.id)}
                            className={clsx(
                                "flex-1 flex items-center justify-center gap-2 py-2.5 rounded-lg text-xs font-bold uppercase tracking-wider transition-all duration-300",
                                isActive
                                    ? "bg-milimo-600/90 text-white shadow-lg shadow-milimo-500/20"
                                    : "text-gray-400 hover:text-white hover:bg-white/5"
                            )}
                        >
                            <Icon size={14} />
                            <span className="hidden sm:inline">{tab.label}</span>
                        </button>
                    )
                })}
            </div>

            {/* Main Content Scrollable Area (if needed, but parent handles scroll) */}
            <div className="space-y-6">

                {/* Dynamic Media Inputs */}
                {mode === 'img2vid' && (
                    <MediaUploader
                        label="Input Image"
                        accept="image/*"
                        selectedFile={inputImage}
                        onFileSelect={setInputImage}
                    />
                )}

                {mode === 'vid2vid' && (
                    <MediaUploader
                        label="Input Video"
                        accept="video/*"
                        selectedFile={inputVideo}
                        onFileSelect={setInputVideo}
                    />
                )}

                {mode === 'vid2vid' && (
                    <div className="space-y-2">
                        <div className="flex justify-between items-center px-1">
                            <label className="text-xs font-bold text-gray-400 uppercase tracking-widest">Strength</label>
                            <span className="text-xs font-mono text-milimo-300">{strength.toFixed(2)}</span>
                        </div>
                        <input
                            type="range"
                            min="0.1"
                            max="1.0"
                            step="0.05"
                            value={strength}
                            onChange={(e) => setStrength(parseFloat(e.target.value))}
                            className="w-full"
                        />
                    </div>
                )}

                {mode === 'keyframe' && (
                    <div className="grid grid-cols-2 gap-4">
                        <MediaUploader
                            label="Start Frame"
                            accept="image/*"
                            selectedFile={startImage}
                            onFileSelect={setStartImage}
                        />
                        <MediaUploader
                            label="End Frame"
                            accept="image/*"
                            selectedFile={endImage}
                            onFileSelect={setEndImage}
                        />
                    </div>
                )}

                {/* Prompt Section */}
                <div className="space-y-2">
                    <label className="text-xs font-bold text-gray-400 uppercase tracking-widest pl-1">Prompt</label>
                    <textarea
                        className="glass-input h-32 resize-none"
                        placeholder={mode === 'vid2vid' ? "Describe the style or changes..." : "Describe your video in detail..."}
                        value={prompt}
                        onChange={(e) => setPrompt(e.target.value)}
                    />
                    <div className="flex items-center justify-between px-1">
                        <span className="text-[10px] text-gray-500 font-medium">200 word limit</span>
                        <button
                            onClick={() => setEnhancePrompt(!enhancePrompt)}
                            className={clsx(
                                "text-[10px] font-bold transition-colors uppercase tracking-wider flex items-center gap-1",
                                enhancePrompt ? "text-milimo-300" : "text-gray-500 hover:text-gray-300"
                            )}
                        >
                            <Sparkles size={12} className={enhancePrompt ? "text-milimo-400" : ""} />
                            Enhance {enhancePrompt ? 'On' : 'Off'}
                        </button>
                    </div>
                </div>

                {numFrames > 125 && (
                    <div className="space-y-2 border-l-2 border-amber-500/50 pl-2 ml-1">
                        <div className="flex justify-between items-center pr-1">
                            <label className="text-xs font-bold text-amber-500 uppercase tracking-widest pl-1">Continuation Prompt</label>
                            <Toggle checked={autoContinue} onChange={setAutoContinue} label="Smart Continue" />
                        </div>

                        {autoContinue ? (
                            <div className="h-20 glass-input flex items-center justify-center text-xs text-milimo-300 italic border-amber-500/20">
                                <Sparkles size={14} className="mr-2 animate-pulse" />
                                Using Gemma 3 to generate continuation...
                            </div>
                        ) : (
                            <textarea
                                className="glass-input h-20 resize-none border-amber-500/20 focus:border-amber-500/50"
                                placeholder="Leave empty to use main prompt..."
                                value={continuationPrompt}
                                onChange={(e) => setContinuationPrompt(e.target.value)}
                            />
                        )}
                        {!autoContinue && <p className="text-[10px] text-gray-500 pl-1">Optional. Guides the video for chunks after the first one.</p>}
                    </div>
                )}

                <div className="space-y-2">
                    <label className="text-xs font-bold text-gray-400 uppercase tracking-widest pl-1">Negative Prompt</label>
                    <textarea
                        className="glass-input h-20 resize-none"
                        placeholder="low quality, artifacts, blurry..."
                        value={negPrompt}
                        onChange={(e) => setNegPrompt(e.target.value)}
                    />
                </div>

                {/* Settings Grid */}
                <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                        <label className="text-xs font-bold text-gray-400 uppercase tracking-widest pl-1">Seed</label>
                        <div className="flex gap-2">
                            <input
                                type="number"
                                className="glass-input h-10 py-2"
                                value={seed}
                                onChange={(e) => setSeed(parseInt(e.target.value))}
                            />
                            <button
                                className="w-10 h-10 flex items-center justify-center bg-white/5 rounded-xl border border-white/10 hover:bg-white/10 hover:border-white/20 transition-all text-gray-400 hover:text-white"
                                onClick={() => setSeed(Math.floor(Math.random() * 1000000))}
                                title="Randomize"
                            >
                                🎲
                            </button>
                        </div>
                    </div>

                    <div className="space-y-2">
                        <label className="text-xs font-bold text-gray-400 uppercase tracking-widest pl-1">Ratio</label>
                        <select
                            className="glass-input h-10 py-2 pr-8 appearance-none"
                            value={`${width}x${height}`}
                            onChange={(e) => {
                                const [w, h] = e.target.value.split('x').map(Number);
                                setWidth(w);
                                setHeight(h);
                            }}
                        >
                            <option value="1280x704">1280x704 (HD)</option>
                            <option value="1920x1088">1920x1088 (FHD)</option>
                            <option value="3840x2176">3840x2176 (4K)</option>
                            <option value="768x512">768x512 (Landscape)</option>
                            <option value="512x768">512x768 (Portrait)</option>
                            <option value="512x512">512x512 (Square)</option>
                        </select>
                    </div>

                    <div className="space-y-2 col-span-2 sm:col-span-1">
                        <label className="text-xs font-bold text-gray-400 uppercase tracking-widest pl-1">Duration</label>
                        <select
                            className="glass-input h-10 py-2 pr-8 appearance-none"
                            value={numFrames}
                            onChange={(e) => setNumFrames(parseInt(e.target.value))}
                        >
                            <option value={81}>3.2s (81 frames) - Fast</option>
                            <option value={121}>4.8s (121 frames) - Standard</option>
                            <option value={161}>6.4s (161 frames) - Max Native</option>
                            <option value={241}>9.6s (~241 frames) - Chained</option>
                            <option value={500}>20s (~500 frames) - Chained</option>
                            <option value={1500}>60s (~1500 frames) - Chained</option>
                        </select>
                        {numFrames > 161 && (
                            <div className="text-[10px] text-amber-400 flex items-center gap-1.5 px-1 bg-amber-500/10 py-1 rounded">
                                <span>⚠️</span>
                                <span>Multi-stage generation. Takes longer.</span>
                            </div>
                        )}
                    </div>
                </div>

                {/* Generate / Stop Button */}
                <div className="pt-4 sticky bottom-0 z-20">
                    {isGenerating ? (
                        <button
                            onClick={onStop}
                            className="w-full py-4 rounded-xl font-bold text-sm uppercase tracking-widest shadow-lg transition-all duration-300 relative overflow-hidden group bg-red-600 hover:bg-red-500 text-white shadow-red-500/20 hover:shadow-red-500/40 hover:-translate-y-0.5"
                        >
                            <span className="relative z-10 flex items-center justify-center gap-2">
                                <span className="animate-spin">⚡</span> Generating... (Click to Stop)
                            </span>
                        </button>
                    ) : (
                        <button
                            onClick={onGenerate}
                            className={clsx(
                                "w-full py-4 rounded-xl font-bold text-sm uppercase tracking-widest shadow-lg transition-all duration-300 relative overflow-hidden group",
                                'bg-milimo-600 hover:bg-milimo-500 text-white shadow-milimo-500/20 hover:shadow-milimo-500/40 hover:-translate-y-0.5'
                            )}
                        >
                            <span className="relative z-10 flex items-center justify-center gap-2">
                                <span className="text-lg">▶</span>
                                Generate {mode === 'keyframe' ? 'Interpolation' : (mode === 'img2vid' ? 'From Image' : 'Video')}
                            </span>
                            <div className="absolute inset-0 bg-gradient-to-r from-cyan-400 to-blue-400 opacity-0 group-hover:opacity-20 transition-opacity duration-300" />
                        </button>
                    )}
                </div>
            </div>
        </div>
    );
}
