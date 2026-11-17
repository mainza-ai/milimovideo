import { describe, expect, it } from 'vitest';
import { ALL_STUDIO_SURFACES, parseStudioFeatureFlags } from './features';

describe('studio feature flags', () => {
    it('enables every surface by default', () => {
        const flags = parseStudioFeatureFlags({});
        expect(flags.enabled).toBe(true);
        expect(flags.disabledSurfaces).toEqual([]);
        expect(ALL_STUDIO_SURFACES).toHaveLength(8);
    });

    it('parses a disabled surface list and ignores unknown names', () => {
        const flags = parseStudioFeatureFlags({
            VITE_STUDIO_DISABLED_SURFACES: 'shorts, edit, nope, SHORTS',
        });
        expect(flags.disabledSurfaces).toEqual(['shorts', 'edit']);
    });

    it('supports a global rollback', () => {
        expect(parseStudioFeatureFlags({ VITE_STUDIO_ENABLED: 'false' }).enabled).toBe(false);
        expect(parseStudioFeatureFlags({ VITE_STUDIO_ENABLED: ' FALSE ' }).enabled).toBe(false);
        expect(parseStudioFeatureFlags({ VITE_STUDIO_ENABLED: 'true' }).enabled).toBe(true);
    });
});
