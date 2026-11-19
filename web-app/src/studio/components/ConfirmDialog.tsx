import { useEffect, useRef } from 'react';
import { TriangleAlert, X } from 'lucide-react';
import { useConfirmStore } from '../state/confirmStore';

/** One styled confirmation surface for destructive/irreversible actions. */
export function ConfirmDialog() {
    const current = useConfirmStore((state) => state.current);
    const close = useConfirmStore((state) => state.close);
    const confirmRef = useRef<HTMLButtonElement>(null);
    const cancelRef = useRef<HTMLButtonElement>(null);

    useEffect(() => {
        if (!current) return undefined;
        const previous = document.activeElement as HTMLElement | null;
        (current.options.tone === 'danger' ? cancelRef.current : confirmRef.current)?.focus();
        const onKeyDown = (event: KeyboardEvent) => {
            if (event.key === 'Escape') {
                event.preventDefault();
                event.stopPropagation();
                close(false);
            }
            if (event.key === 'Tab') {
                const first = confirmRef.current;
                const last = cancelRef.current;
                if (!first || !last) return;
                if (event.shiftKey && document.activeElement === first) {
                    event.preventDefault();
                    last.focus();
                } else if (!event.shiftKey && document.activeElement === last) {
                    event.preventDefault();
                    first.focus();
                }
            }
        };
        document.addEventListener('keydown', onKeyDown, true);
        return () => {
            document.removeEventListener('keydown', onKeyDown, true);
            previous?.focus?.();
        };
    }, [close, current]);

    if (!current) return null;
    const { options } = current;
    const danger = options.tone === 'danger';
    return (
        <div className="st-dialog-backdrop" role="presentation" onMouseDown={() => close(false)}>
            <div
                className="st-dialog"
                role="alertdialog"
                aria-modal="true"
                aria-labelledby="st-confirm-title"
                aria-describedby={options.body ? 'st-confirm-body' : undefined}
                onMouseDown={(event) => event.stopPropagation()}
            >
                <div className="st-dialog-head">
                    <span className="st-dialog-icon" data-tone={danger ? 'danger' : undefined}>
                        <TriangleAlert size={16} />
                    </span>
                    <h2 id="st-confirm-title">{options.title}</h2>
                    <button
                        type="button"
                        className="st-icon-button"
                        aria-label="Close dialog"
                        onClick={() => close(false)}
                    >
                        <X size={14} />
                    </button>
                </div>
                {options.body && (
                    <p id="st-confirm-body" className="st-dialog-body">{options.body}</p>
                )}
                <div className="st-dialog-actions">
                    <button
                        ref={cancelRef}
                        type="button"
                        className="st-button"
                        onClick={() => close(false)}
                    >
                        {options.cancelLabel || 'Cancel'}
                    </button>
                    <button
                        ref={confirmRef}
                        type="button"
                        className="st-button"
                        data-variant={danger ? 'danger' : 'primary'}
                        onClick={() => close(true)}
                    >
                        {options.confirmLabel || 'Continue'}
                    </button>
                </div>
            </div>
        </div>
    );
}
