import { AlertCircle, CheckCircle2, Info, X } from 'lucide-react';
import { useStudioDocumentStore } from '../state/documentStore';

const ICONS = {
    success: CheckCircle2,
    error: AlertCircle,
    info: Info,
} as const;

/**
 * Studio-wide toast layer. The store is shared by every surface; this is the
 * single mounted renderer, so any addToast call becomes visible.
 */
export function StudioToastLayer() {
    const toasts = useStudioDocumentStore((state) => state.toasts);
    const removeToast = useStudioDocumentStore((state) => state.removeToast);

    return (
        <div className="st-toast-layer" aria-live="polite" aria-atomic="false" data-testid="studio-toasts">
            {toasts.slice(-5).map((toast) => {
                const Icon = ICONS[toast.type] || Info;
                return (
                    <div
                        key={toast.id}
                        className="st-toast"
                        data-tone={toast.type}
                        role={toast.type === 'error' ? 'alert' : 'status'}
                    >
                        <Icon size={14} aria-hidden="true" />
                        <span className="st-toast-message">{toast.message}</span>
                        <button
                            type="button"
                            className="st-toast-close"
                            aria-label="Dismiss notification"
                            onClick={() => removeToast(toast.id)}
                        >
                            <X size={12} />
                        </button>
                    </div>
                );
            })}
        </div>
    );
}
