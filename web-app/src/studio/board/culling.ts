import type { BoardPosition, Camera } from '../state/editorStore';

export interface CullableCard {
    position: BoardPosition;
}

/**
 * Cards that intersect the viewport plus overscan. Pure so it can be tested
 * without mounting 500 DOM nodes.
 */
export function cullCards<T extends CullableCard>(
    cards: T[],
    camera: Camera,
    viewport: { width: number; height: number },
    overscanPx = 260,
): T[] {
    const zoom = Math.max(0.05, camera.zoom);
    const pad = overscanPx / zoom;
    const left = -camera.x / zoom - pad;
    const top = -camera.y / zoom - pad;
    const right = left + viewport.width / zoom + pad * 2;
    const bottom = top + viewport.height / zoom + pad * 2;

    return cards.filter(({ position }) => (
        position.x + position.width >= left
        && position.x <= right
        && position.y + position.height >= top
        && position.y <= bottom
    ));
}
