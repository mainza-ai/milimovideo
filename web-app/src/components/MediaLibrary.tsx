

export function MediaLibrary() {
    const dummyItems = Array(6).fill(null).map((_, i) => i);

    return (
        <div className="w-full">
            <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">Recent Generations</h3>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                {dummyItems.map((_, i) => (
                    <div key={i} className="aspect-video bg-white/5 rounded-lg border border-white/5 hover:border-white/20 transition-all cursor-pointer group relative overflow-hidden">
                        <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity bg-black/40">
                            <span className="text-xs font-bold">Play</span>
                        </div>
                        {/* Placeholder for thumbnail */}
                        <div className="w-full h-full bg-gradient-to-br from-gray-800 to-black opacity-50" />
                    </div>
                ))}
            </div>
        </div>
    );
}
