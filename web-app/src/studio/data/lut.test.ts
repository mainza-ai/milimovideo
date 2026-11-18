import { afterEach, describe, expect, it, vi } from 'vitest';
import { uploadProjectLut } from './lut';

function json(data: unknown, status = 200): Response {
    return new Response(JSON.stringify(data), {
        status,
        headers: { 'Content-Type': 'application/json' },
    });
}

describe('uploadProjectLut', () => {
    afterEach(() => vi.unstubAllGlobals());

    it('posts the LUT as a project lut asset', async () => {
        const calls: Array<{ url: string; init?: RequestInit }> = [];
        vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
            calls.push({ url: typeof input === 'string' ? input : String(input), init });
            return json({
                url: '/projects/p1/assets/lut/swap.cube',
                type: 'lut',
                filename: 'swap.cube',
                asset_id: 'asset-1',
            });
        }));

        const file = new File(['LUT_3D_SIZE 2'], 'swap.cube', { type: 'text/plain' });
        const result = await uploadProjectLut('p1', file);

        expect(result.url).toBe('/projects/p1/assets/lut/swap.cube');
        const call = calls[0];
        expect(call.url).toContain('/upload');
        const body = call.init?.body as FormData;
        expect(body.get('project_id')).toBe('p1');
        expect(body.get('type')).toBe('lut');
        expect((body.get('file') as File).name).toBe('swap.cube');
    });

    it('surfaces backend errors', async () => {
        vi.stubGlobal('fetch', vi.fn(async () => json({ detail: 'Unsupported file type' }, 415)));
        const file = new File(['x'], 'bad.txt', { type: 'text/plain' });
        await expect(uploadProjectLut('p1', file)).rejects.toThrow('Unsupported file type');
    });
});
