import { z } from 'zod';

/** Runtime contracts for endpoints the studio consumes directly. */

export const HealthSchema = z.object({
    status: z.string(),
    device: z.string(),
    gpu: z.object({ busy: z.boolean(), queued: z.number() }),
    slots: z.object({ active_slots: z.array(z.string()) }).passthrough(),
    disk_free_bytes: z.number().nullable(),
    auth_required: z.boolean(),
});

export const PlanStepSchema = z.object({
    kind: z.string(),
    label: z.string(),
    target: z.string().nullable().optional(),
    before: z.unknown().optional(),
    after: z.unknown().optional(),
    detail: z.string().nullable().optional(),
});

export const PlanResponseSchema = z.object({
    mode: z.enum(['ask', 'direct', 'generate', 'find']),
    summary: z.string(),
    steps: z.array(PlanStepSchema),
    estimate: z.object({
        model: z.string().nullable().optional(),
        gpu_seconds: z.number().nullable().optional(),
        vram_mb: z.number().nullable().optional(),
        note: z.string().nullable().optional(),
    }).nullable().optional(),
    warnings: z.array(z.string()).default([]),
    shot_id: z.string().nullable().optional(),
});

export const ReviewCommentSchema = z.object({
    id: z.string(),
    project_id: z.string(),
    shot_id: z.string().nullable(),
    kind: z.enum(['comment', 'approval']),
    frame: z.number(),
    body: z.string(),
    author: z.string(),
    status: z.enum(['open', 'resolved', 'approved', 'changes_requested']),
    created_at: z.string().nullable(),
    updated_at: z.string().nullable(),
});

export const ReviewPayloadSchema = z.object({
    comments: z.array(ReviewCommentSchema),
    approvals: z.array(ReviewCommentSchema),
    summary: z.object({
        comments: z.number(),
        open: z.number(),
        approved: z.number(),
        changes_requested: z.number(),
    }),
});

export const SharedShotSchema = z.object({
    id: z.string(),
    sequence_index: z.number().nullable(),
    action: z.string().nullable().optional(),
    prompt: z.string().nullable().optional(),
    dialogue: z.string().nullable().optional(),
    status: z.string().nullable().optional(),
    num_frames: z.number().nullable().optional(),
    video_url: z.string().nullable().optional(),
    thumbnail_url: z.string().nullable().optional(),
});

export const SharedCommentSchema = ReviewCommentSchema.omit({ project_id: true });

export const SharePayloadSchema = z.object({
    shared: z.literal(true),
    project: z.object({
        id: z.string(),
        name: z.string(),
        resolution_w: z.number(),
        resolution_h: z.number(),
        fps: z.number(),
    }),
    shots: z.array(SharedShotSchema),
    comments: z.array(SharedCommentSchema),
    approvals: z.array(SharedCommentSchema),
    summary: z.object({
        comments: z.number(),
        open: z.number(),
        approved: z.number(),
        changes_requested: z.number(),
    }),
});

export type PlanResponse = z.infer<typeof PlanResponseSchema>;
export type ReviewPayload = z.infer<typeof ReviewPayloadSchema>;
export type SharePayload = z.infer<typeof SharePayloadSchema>;
