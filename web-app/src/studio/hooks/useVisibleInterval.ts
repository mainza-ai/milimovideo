import { useEffect, useRef } from 'react';

/**
 * Polling helper for studio surfaces.
 *
 * Chrome throttles timers in hidden tabs, but each studio hook still holds its
 * own interval and some of them refresh eagerly when the tab returns. This hook
 * centralizes the policy: one interval per resource, skipped entirely while the
 * document is hidden, and an immediate refresh when the tab becomes visible so
 * stale state is reconciled once instead of drifting.
 */
export function useVisibleInterval(
    callback: () => void,
    intervalMs: number,
    options: { enabled?: boolean; refreshOnVisible?: boolean } = {},
) {
    const { enabled = true, refreshOnVisible = true } = options;
    const callbackRef = useRef(callback);

    useEffect(() => {
        callbackRef.current = callback;
    }, [callback]);

    useEffect(() => {
        if (!enabled || intervalMs <= 0) return undefined;
        const isHidden = () => typeof document !== 'undefined' && document.hidden;
        const tick = () => {
            if (isHidden()) return;
            callbackRef.current();
        };
        const timer = window.setInterval(tick, intervalMs);
        const onVisible = () => {
            if (!document.hidden && refreshOnVisible) callbackRef.current();
        };
        document.addEventListener('visibilitychange', onVisible);
        return () => {
            window.clearInterval(timer);
            document.removeEventListener('visibilitychange', onVisible);
        };
    }, [enabled, intervalMs, refreshOnVisible]);
}

export function isDocumentHidden(): boolean {
    return typeof document !== 'undefined' && document.hidden;
}
