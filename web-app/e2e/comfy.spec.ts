import { expect, test } from '@playwright/test';
import { stubStudioApi } from './stubs';

function json(data: unknown) {
    return { status: 200, contentType: 'application/json', body: JSON.stringify(data) };
}

const SHOT = {
    id: 'comfy-shot', project_id: 'project-test', index: 0, sequence_index: 0,
    prompt: 'a rainy street', negative_prompt: '', seed: 42, width: 512, height: 320,
    num_frames: 25, num_inference_steps: 8, cfg_scale: 2, pipeline_override: 'auto',
    track_index: 0, start_frame: 0, trim_in: 0, trim_out: 0, timeline: [],
    status: 'pending', sequenceIndex: 0, numFrames: 25, cfgScale: 2,
};

test.beforeEach(async ({ page }) => {
    await stubStudioApi(page);
    await page.route('**/projects', (route) => route.fulfill(json([{
        id: 'project-test', name: 'Audit Test', fps: 25, seed: 42,
        resolution_w: 768, resolution_h: 512, revision: 0,
        shots: [SHOT], scenes: [],
    }])));
    await page.route('**/projects/*/active_jobs', (route) => route.fulfill(json([])));
    await page.route('**/comfy/health', (route) => route.fulfill(json({
        enabled: true, reachable: true, compatible: true, version: '0.45.0',
        url: 'http://127.0.0.1:8001', queue_running: 0, queue_pending: 0,
    })));
    await page.route('**/comfy/settings', (route) => route.fulfill(json({
        revision: 0, url: 'http://127.0.0.1:8001', enabled: true,
        declared_gb: 2, has_token: false, hard_disabled: false,
    })));
    await page.route('**/comfy/catalog', (route) => route.fulfill(json({
        node_count: 2,
        nodes: [
            { class_type: 'EmptyImage', display_name: 'Empty Image', category: 'image', output_node: false, outputs: ['IMAGE'] },
            { class_type: 'SaveImage', display_name: 'Save Image', category: 'image', output_node: true, outputs: [] },
        ],
        templates: { test: ['Smoke'] },
    })));
    await page.route('**/comfy/schema**', (route) => route.fulfill(json({
        classes: {
            EmptyImage: { input: { required: { width: ['INT'], height: ['INT'] } }, output: ['IMAGE'] },
            SaveImage: { input: { required: { images: ['IMAGE'] } }, output: [], output_node: true },
        },
    })));
    await page.route('**/comfy/models**', (route) => route.fulfill(json({ folder: 'checkpoints', models: [] })));
    await page.route('**/projects/*/comfy/workflows', (route) => route.fulfill(json([])));
    await page.route('**/comfy/validate', (route) => route.fulfill(json({
        ok: true, live: true, nodes: 2, node_count: 2, output_nodes: ['2'],
        width: 640, height: 360, warnings: [], errors: [],
    })));
    await page.route('**/comfy/run', (route) => route.fulfill(json({
        job_id: 'job_comfy_e2e', status: 'queued', validation: { output_nodes: ['2'], warnings: [] },
        comfy_url: 'http://127.0.0.1:8001',
    })));
});

test('ComfyUI panel imports, validates, and queues a workflow', async ({ page }) => {
    await page.goto('/studio');
    await page.getByRole('button', { name: 'Generate', exact: true }).first().click();
    await expect(page.getByText('ComfyUI workflow')).toBeVisible();
    await page.getByText('ComfyUI workflow').click();
    await page.getByRole('button', { name: /Smoke test/ }).click();
    await page.getByRole('button', { name: /^Validate$/ }).click();
    await expect(page.getByText('live preflight')).toBeVisible();
    await page.getByRole('button', { name: /^Run$/ }).click();
    await expect(page.getByText('ComfyUI workflow queued')).toBeVisible();
});
