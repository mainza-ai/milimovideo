import type { StudioSurface } from './state/editorStore';

export const ALL_STUDIO_SURFACES: StudioSurface[] = [
    'story', 'generate', 'edit', 'review', 'library', 'deliver', 'shorts', 'setup',
];

export interface StudioFeatureFlags {
    enabled: boolean;
    disabledSurfaces: StudioSurface[];
}

interface StudioEnv {
    VITE_STUDIO_ENABLED?: string;
    VITE_STUDIO_DISABLED_SURFACES?: string;
}

function isSurface(value: string): value is StudioSurface {
    return (ALL_STUDIO_SURFACES as string[]).includes(value);
}

/**
 * Parse build-time studio flags. The default is fully enabled so existing
 * installs are unaffected. Disable the shell for a global rollback, or disable
 * individual surfaces while leaving a surface's data intact.
 */
export function parseStudioFeatureFlags(env: StudioEnv = {}): StudioFeatureFlags {
    const enabled = (env.VITE_STUDIO_ENABLED ?? 'true').trim().toLowerCase() !== 'false';
    const disabledSurfaces = (env.VITE_STUDIO_DISABLED_SURFACES ?? '')
        .split(',')
        .map((item) => item.trim().toLowerCase())
        .filter(isSurface);
    return {
        enabled,
        disabledSurfaces: [...new Set(disabledSurfaces)],
    };
}

const env = (import.meta as ImportMeta & { env: StudioEnv }).env || {};

export const STUDIO_FLAGS = parseStudioFeatureFlags(env);

export function isStudioSurfaceEnabled(surface: StudioSurface): boolean {
    return STUDIO_FLAGS.enabled && !STUDIO_FLAGS.disabledSurfaces.includes(surface);
}

export function enabledStudioSurfaces(): StudioSurface[] {
    return ALL_STUDIO_SURFACES.filter(isStudioSurfaceEnabled);
}
