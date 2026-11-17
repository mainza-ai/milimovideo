import type { Page } from '@playwright/test';

function json(data: unknown) {
    return { status: 200, contentType: 'application/json', body: JSON.stringify(data) };
}

/** API stubs for shell-only tests; no Python backend required. */
export async function stubStudioApi(page: Page) {
    await page.route('**/health', (route) => route.fulfill(json({
        status: 'ok',
        device: 'mps',
        gpu: { busy: false, queued: 0 },
        slots: { active_slots: [] },
        disk_free_bytes: 1,
        auth_required: false,
    })));
    await page.route('**/projects', (route) => route.fulfill(json([
        { id: 'project-test', name: 'Audit Test', shots: [], scenes: [] },
    ])));
    await page.route('**/projects/*/board', (route) => route.fulfill(json({
        layout: { nodes: [], areas: [], links: [] },
        revision: 0,
    })));
    await page.route('**/projects/*/view', (route) => route.fulfill(json({ view: {} })));
    await page.route('**/projects/*/review', (route) => route.fulfill(json({
        comments: [],
        approvals: [],
        summary: { comments: 0, open: 0, approved: 0, changes_requested: 0 },
    })));
    await page.route('**/projects/*/elements', (route) => route.fulfill(json([])));
    await page.route('**/projects/*/takes', (route) => route.fulfill(json({ takes: [], count: 0 })));
    await page.route('**/projects/*/renders', (route) => route.fulfill(json([])));
    await page.route('**/models', (route) => route.fulfill(json({ models: [], count: 0 })));
    await page.route('**/models/pipelines', (route) => route.fulfill(json({})));
    await page.route('**/uploads', (route) => route.fulfill(json([])));
}
