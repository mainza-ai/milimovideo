interface VideoPlayerProps {
    videoUrl: string | null;
}

export function VideoPlayer({ videoUrl }: VideoPlayerProps) {
    return (
        <div className="w-full aspect-video bg-black/40 rounded-3xl border border-white/10 shadow-2xl relative overflow-hidden group transition-all duration-500 hover:border-milimo-500/30 hover:shadow-milimo-500/20">
            {videoUrl ? (
                <video src={videoUrl} controls loop className="w-full h-full object-contain" />
            ) : (
                <div className="absolute inset-0 flex items-center justify-center">
                    <div className="text-gray-600 font-light flex flex-col items-center gap-4 animate-float">
                        <div className="p-8 rounded-full bg-white/5 border border-white/5 shadow-inner backdrop-blur-sm">
                            <svg className="w-12 h-12 opacity-30 text-milimo-300" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
                        </div>
                        <span className="text-sm opacity-50 tracking-widest uppercase text-milimo-200/60 font-semibold">Ready to Generate</span>
                    </div>
                </div>
            )}

            {/* Controls Overlay - Only show if no video playing? Or overlay? For now remove custom overlay if native controls are used or keep it simple */}
            {!videoUrl && (
                <div className="absolute bottom-0 left-0 right-0 p-6 bg-gradient-to-t from-black/80 to-transparent opacity-0 group-hover:opacity-100 transition-opacity">
                    <div className="w-full h-1 bg-white/20 rounded-full mb-4 cursor-pointer overflow-hidden">
                        <div className="w-1/3 h-full bg-milimo-500" />
                    </div>
                    <div className="flex justify-between items-center">
                        <button className="text-white hover:text-milimo-400">Play</button>
                        <span className="text-xs font-mono text-gray-400">00:00 / 00:04</span>
                    </div>
                </div>
            )}
        </div>
    );
}
