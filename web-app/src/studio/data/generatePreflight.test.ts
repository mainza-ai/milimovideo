import { afterEach, describe, expect, it, vi } from 'vitest';
import { renderHook } from '@testing-library/react';
import { useStudioGenerate, type MemoryDowngrade } from './useStudioData';
import { DEFAULT_PROJECT } from '../../stores/slices/projectSlice';
import type { Shot } from '../../stores/types';

function json(data: unknown, status = 200): Response {
    return new Response(JSON.stringify(data), {
        status,
        headers: { 'Content-Type': 'application/json' },
    });
}

const shot = DEFAULT_PROJECT.shots[0] as Shot;

describe('useStudioGenerate memory preflight', () => {
    afterEach(() => vi.restoreAllMocks());

    it('offers the safe preset instead of posting a job that cannot fit', async () => {
        const downgrade: MemoryDowngrade = {
            params: { num_frames: 93, upscale: false },
            label: '93 frames, upscale off',
            estimate: { total_gb: 88.4 },
        };
        const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
            const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
            if (url.includes('/system/memory/preflight')) {
                return json({
                    ok: false,
                    reason: 'Needs ~105.8 GB but only 104.0 GB of RAM is free',
                    suggestions: ['Reduce frames or resolution'],
                    downgrade,
                });
            }
            return json({ job_id: 'should-not-post' });
        });
        vi.stubGlobal('fetch', fetchMock);

        const onDowngrade = vi.fn();
        const { result } = renderHook(() => useStudioGenerate({ onDowngrade }));
        const jobId = await result.current(shot);

        expect(jobId).toBeNull();
        expect(onDowngrade).toHaveBeenCalledWith(
            shot,
            expect.objectContaining({ label: '93 frames, upscale off' }),
            expect.stringContaining('105.8 GB'),
        );
        const posted = fetchMock.mock.calls
            .map((call) => String(call[0]))
            .filter((url) => url.includes('/generate') || url.includes('/shots/'));
        expect(posted).toEqual([]);
    });

    it('queues the job when the refusal is transient', async () => {
        const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
            const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
            if (url.includes('/system/memory/preflight')) {
                return json({
                    ok: false,
                    retryable: true,
                    reason: 'Needs 40 GB but a job is running',
                    downgrade: null,
                });
            }
            return json({ job_id: 'job-wait' });
        });
        vi.stubGlobal('fetch', fetchMock);

        const { result } = renderHook(() => useStudioGenerate({}));
        const jobId = await result.current(shot);

        expect(jobId).toBe('job-wait');
        expect(fetchMock.mock.calls.some((call) => String(call[0]).includes('/generate/advanced'))).toBe(true);
    });

    it('posts the job when the preflight admits it', async () => {
        const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
            const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
            if (url.includes('/system/memory/preflight')) {
                return json({ ok: true, warnings: [], will_unload: [] });
            }
            return json({ job_id: 'job-ok' });
        });
        vi.stubGlobal('fetch', fetchMock);

        const { result } = renderHook(() => useStudioGenerate({}));
        const jobId = await result.current(shot);

        expect(jobId).toBe('job-ok');
        expect(fetchMock.mock.calls.some((call) => String(call[0]).includes('/generate/advanced'))).toBe(true);
    });
});
