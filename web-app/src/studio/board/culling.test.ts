import { describe, expect, it } from 'vitest';
import { cullCards } from './culling';

function card(index: number) {
    return {
        id: `shot-${index}`,
        position: {
            x: (index % 4) * 344,
            y: Math.floor(index / 4) * 280,
            width: 300,
            height: 238,
        },
    };
}

describe('board culling', () => {
    it('mounts a small slice of a 500-card board at detail zoom', () => {
        const cards = Array.from({ length: 500 }, (_, index) => card(index));
        const visible = cullCards(
            cards,
            { x: 0, y: 0, zoom: 1 },
            { width: 1440, height: 900 },
        );
        expect(visible.length).toBeGreaterThan(0);
        expect(visible.length).toBeLessThan(60);
    });

    it('keeps every card when zoomed far out', () => {
        const cards = Array.from({ length: 500 }, (_, index) => card(index));
        const visible = cullCards(
            cards,
            { x: 0, y: 0, zoom: 0.05 },
            { width: 1440, height: 2000 },
        );
        expect(visible).toHaveLength(500);
    });

    it('keeps a card that starts off-screen but overlaps the overscan', () => {
        const cards = [
            { id: 'a', position: { x: -320, y: 0, width: 300, height: 238 } },
            { id: 'b', position: { x: -2000, y: 0, width: 300, height: 238 } },
        ];
        const visible = cullCards(cards, { x: 0, y: 0, zoom: 1 }, { width: 900, height: 600 });
        expect(visible.map((item) => item.id)).toEqual(['a']);
    });
});
