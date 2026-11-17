import { describe, expect, it } from 'vitest';
import { rankSearch } from './localSearch';

describe('local search', () => {
    const records = [
        { kind: 'shot', text: 'Neon city rain at night', item: { id: 's1' } },
        { kind: 'shot', text: 'Quiet forest morning', item: { id: 's2' } },
        { kind: 'asset', text: 'neon-sign.png', item: { id: 'a1' } },
        { kind: 'element', text: '@detective rain coat', item: { id: 'e1' } },
    ];

    it('ranks full token matches above partial text', () => {
        const results = rankSearch('neon rain', records);
        expect(results[0].item.id).toBe('s1');
        expect(results.map((result) => result.item.id)).toContain('a1');
        expect(results.map((result) => result.item.id)).toContain('e1');
    });

    it('supports prefix matching', () => {
        const results = rankSearch('fore', records);
        expect(results[0].item.id).toBe('s2');
    });

    it('returns nothing for empty or absent terms', () => {
        expect(rankSearch('', records)).toEqual([]);
        expect(rankSearch('spaceship', records)).toEqual([]);
    });
});
