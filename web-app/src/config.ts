export const API_BASE_URL = "http://localhost:8000";

/**
 * transform a path (relative or absolute) into a fully qualified URL pointing to the backend.
 * Handles:
 * - Full URLs (returns as is)
 * - Relative URL paths (e.g. /uploads/image.png -> http://localhost:8000/uploads/image.png)
 * - undefined/null (returns undefined)
 */
export const getAssetUrl = (path: string | undefined | null): string | undefined => {
    if (!path) return undefined;
    if (path.startsWith('http')) return path;

    // Ensure path starts with / if not present
    const cleanPath = path.startsWith('/') ? path : `/${path}`;
    return `${API_BASE_URL}${cleanPath}`;
};
