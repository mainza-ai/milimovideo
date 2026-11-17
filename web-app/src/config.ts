const envBase = (import.meta.env.VITE_API_URL as string | undefined)?.trim();
const hostname = typeof window !== "undefined" ? window.location.hostname : "localhost";
const devBase = `http://${hostname || "localhost"}:8000`;

// Production builds served by the backend use same-origin. Vite dev and Tauri
// point at the backend on this host. Never hardcode localhost: a LAN browser
// must talk to the machine running the studio.
export const API_BASE_URL = (
    envBase ||
    (import.meta.env.DEV ? devBase : (typeof window !== "undefined" ? window.location.origin : devBase))
).replace(/\/$/, "");

export const API_TOKEN = ((import.meta.env.VITE_API_TOKEN as string | undefined) || "").trim();

export const apiUrl = (path: string): string => {
    if (/^https?:\/\//i.test(path)) return path;
    return `${API_BASE_URL}${path.startsWith("/") ? path : `/${path}`}`;
};

export const eventUrl = (path: string): string => {
    const url = apiUrl(path);
    return API_TOKEN
        ? `${url}${url.includes("?") ? "&" : "?"}token=${encodeURIComponent(API_TOKEN)}`
        : url;
};

export const authHeaders = (headers: Record<string, string> = {}): Record<string, string> => (
    API_TOKEN ? { ...headers, Authorization: `Bearer ${API_TOKEN}` } : headers
);

export const apiFetch = (path: string, init: RequestInit = {}): Promise<Response> =>
    fetch(apiUrl(path), {
        ...init,
        headers: authHeaders((init.headers as Record<string, string>) || {}),
    });

/**
 * transform a path (relative or absolute) into a fully qualified URL pointing to the backend.
 * Handles:
 * - Full URLs (returns as is)
 * - Relative URL paths (e.g. /uploads/image.png -> http://localhost:8000/uploads/image.png)
 * - undefined/null (returns undefined)
 */
export const getAssetUrl = (path: string | undefined | null): string | undefined => {
    if (!path) return undefined;

    let relative = path;
    if (/^https?:\/\//i.test(path)) {
        try {
            const parsed = new URL(path);
            // Legacy rows stored absolute localhost URLs. Route those through the
            // current API origin so LAN/Tauri clients and bearer tokens work.
            const local = ['localhost', '127.0.0.1', '[::1]'].includes(parsed.hostname)
                || parsed.origin === API_BASE_URL;
            if (!local) return path;
            relative = `${parsed.pathname}${parsed.search}`;
        } catch {
            return path;
        }
    }

    // Ensure path starts with / if not present
    const cleanPath = relative.startsWith('/') ? relative : `/${relative}`;
    const url = `${API_BASE_URL}${cleanPath}`;
    if (!API_TOKEN) return url;
    return `${url}${url.includes('?') ? '&' : '?'}token=${encodeURIComponent(API_TOKEN)}`;
};
