import { Upload, X, FileVideo, FileImage } from 'lucide-react';
import { useCallback, useEffect, useState } from 'react';
import clsx from 'clsx';

interface MediaUploaderProps {
    accept: 'image/*' | 'video/*' | 'image/*,video/*';
    label: string;
    onFileSelect: (file: File | null) => void;
    selectedFile: File | null;
    className?: string;
}

export function MediaUploader({ accept, label, onFileSelect, selectedFile, className }: MediaUploaderProps) {
    const [isDragging, setIsDragging] = useState(false);
    const [previewUrl, setPreviewUrl] = useState<string | null>(null);

    useEffect(() => {
        if (selectedFile) {
            const url = URL.createObjectURL(selectedFile);
            setPreviewUrl(url);
            return () => URL.revokeObjectURL(url);
        } else {
            setPreviewUrl(null);
        }
    }, [selectedFile]);

    const handleDragOver = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(true);
    }, []);

    const handleDragLeave = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);
    }, []);

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);

        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            const file = e.dataTransfer.files[0];
            if (accept.includes(file.type.split('/')[0])) {
                onFileSelect(file);
            } else {
                alert(`Invalid file type. Please upload a ${accept.includes('video') ? 'video' : 'image'}.`);
            }
        }
    }, [accept, onFileSelect]);

    const handleChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            onFileSelect(e.target.files[0]);
        }
    }, [onFileSelect]);

    const clearFile = useCallback((e: React.MouseEvent) => {
        e.stopPropagation();
        onFileSelect(null);
    }, [onFileSelect]);

    return (
        <div className={clsx("flex flex-col gap-2", className)}>
            <label className="text-xs font-bold text-gray-400 uppercase tracking-widest pl-1">{label}</label>

            <div
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => document.getElementById(`file-upload-${label}`)?.click()}
                className={clsx(
                    "relative group cursor-pointer rounded-xl border-2 border-dashed transition-all duration-300 h-32 flex items-center justify-center overflow-hidden",
                    isDragging ? "border-milimo-400 bg-milimo-400/10" : "border-white/10 hover:border-white/20 hover:bg-white/5",
                    previewUrl ? "bg-black/40" : ""
                )}
            >
                <input
                    id={`file-upload-${label}`}
                    type="file"
                    accept={accept}
                    className="hidden"
                    onChange={handleChange}
                />

                {previewUrl ? (
                    <>
                        {selectedFile?.type.startsWith('video') ? (
                            <video src={previewUrl} className="w-full h-full object-cover opacity-60" muted loop autoPlay playsInline />
                        ) : (
                            <img src={previewUrl} alt="Preview" className="w-full h-full object-cover opacity-60" />
                        )}

                        <div className="absolute inset-0 flex items-center justify-center p-4">
                            <div className="bg-black/60 backdrop-blur-sm rounded-lg px-3 py-1 text-xs text-white flex items-center gap-2">
                                {selectedFile?.type.startsWith('video') ? <FileVideo size={14} /> : <FileImage size={14} />}
                                <span className="truncate max-w-[120px]">{selectedFile?.name}</span>
                            </div>
                        </div>

                        <button
                            onClick={clearFile}
                            className="absolute top-2 right-2 p-1.5 bg-black/60 hover:bg-red-500/80 text-white rounded-full transition-colors backdrop-blur-sm"
                        >
                            <X size={14} />
                        </button>
                    </>
                ) : (
                    <div className="flex flex-col items-center gap-3 text-gray-500 group-hover:text-gray-300 transition-colors pointer-events-none">
                        <div className="p-3 bg-white/5 rounded-full ring-1 ring-white/10 group-hover:ring-white/20 transition-all">
                            <Upload size={20} />
                        </div>
                        <span className="text-xs font-medium">Drag & Drop or Click to Upload</span>
                    </div>
                )}
            </div>
        </div>
    );
}
