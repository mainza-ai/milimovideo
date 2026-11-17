import { describe, expect, it } from 'vitest';
import { HealthSchema, PlanResponseSchema, ReviewPayloadSchema, SharePayloadSchema } from './contracts';

describe('API contracts', () => {
    it('accepts the real health shape', () => {
        const parsed = HealthSchema.safeParse({
            status: 'ok',
            device: 'mps',
            gpu: { busy: false, queued: 0 },
            slots: { active_slots: [], conflicts: { video: ['image'] } },
            disk_free_bytes: 123,
            auth_required: false,
        });
        expect(parsed.success).toBe(true);
    });

    it('rejects a plan missing its mode', () => {
        const parsed = PlanResponseSchema.safeParse({ summary: 'x', steps: [] });
        expect(parsed.success).toBe(false);
    });

    it('defaults plan warnings and accepts a generate estimate', () => {
        const parsed = PlanResponseSchema.safeParse({
            mode: 'generate',
            summary: 'one take',
            steps: [{ kind: 'generate', label: 'Generate' }],
            estimate: { model: 'auto', gpu_seconds: 12, vram_mb: 16000 },
        });
        expect(parsed.success).toBe(true);
        if (parsed.success) expect(parsed.data.warnings).toEqual([]);
    });

    it('validates review and share payloads', () => {
        const summary = { comments: 0, open: 0, approved: 0, changes_requested: 0 };
        expect(ReviewPayloadSchema.safeParse({
            comments: [], approvals: [], summary,
        }).success).toBe(true);

        expect(SharePayloadSchema.safeParse({
            shared: true,
            project: { id: 'p', name: 'P', resolution_w: 1, resolution_h: 1, fps: 25 },
            shots: [],
            comments: [],
            approvals: [],
            summary,
        }).success).toBe(true);

        expect(SharePayloadSchema.safeParse({
            shared: false,
            project: { id: 'p', name: 'P', resolution_w: 1, resolution_h: 1, fps: 25 },
            shots: [], comments: [], approvals: [], summary,
        }).success).toBe(false);
    });
});
