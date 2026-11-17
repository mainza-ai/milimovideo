import { expect, test } from '@playwright/test';

const API = process.env.MILIMO_E2E_API || 'http://127.0.0.1:8010';

function shot(prefix: string, index: number) {
    return {
        id: `${prefix}-perf-shot-${index}`,
        prompt: `Shot ${index}`,
        negative_prompt: '',
        seed: 42,
        width: 768,
        height: 512,
        fps: 25,
        num_frames: 121,
        cfg_scale: 2,
        enhance_prompt: true,
        upscale: false,
        pipeline_override: 'auto',
        track_index: 0,
        start_frame: 0,
        trim_in: 0,
        trim_out: 0,
        timeline: [],
        status: 'pending',
    };
}

test('500-card board mounts only the viewport slice', async ({ page, request }) => {
    const created = await request.post(`${API}/projects`, {
        data: { name: 'Perf Board', resolution_w: 768, resolution_h: 512, fps: 25, seed: 42 },
    });
    const project = await created.json();
    const saved = await request.put(`${API}/projects/${project.id}`, {
        data: {
            id: project.id,
            name: 'Perf Board',
            fps: 25,
            seed: 42,
            resolution_w: 768,
            resolution_h: 512,
            revision: project.revision ?? 0,
            scenes: [],
            shots: Array.from({ length: 500 }, (_, index) => shot(project.id, index)),
        },
    });
    expect(saved.ok()).toBeTruthy();

    await page.goto('/studio');
    await expect(page.getByLabel('Project name')).toHaveValue('Perf Board', { timeout: 30000 });
    await expect(page.getByText('Shot 1', { exact: true }).first()).toBeVisible({ timeout: 30000 });

    const mounted = await page.locator('.st-shot-card').count();
    console.log(`500-card board mounted cards: ${mounted}`);
    expect(mounted).toBeGreaterThan(0);
    expect(mounted).toBeLessThan(40);

    // Headless Chromium throttles rAF, so FPS here is informational rather than
    // a gate. A pan must still complete without unmounting the board.
    const idleFps = await page.evaluate(() => new Promise<number>((resolve) => {
        let frames = 0;
        const start = performance.now();
        const tick = (time: number) => {
            frames += 1;
            if (time - start < 500) requestAnimationFrame(tick);
            else resolve((frames / (time - start)) * 1000);
        };
        requestAnimationFrame(tick);
    }));
    console.log(`headless rAF baseline: ${idleFps.toFixed(1)} fps`);

    await page.mouse.move(1100, 720);
    await page.mouse.down();
    const panFps = await page.evaluate(() => new Promise<number>((resolve) => {
        const viewport = document.querySelector('.st-board-viewport');
        let frames = 0;
        const start = performance.now();
        const tick = () => {
            frames += 1;
            viewport?.dispatchEvent(new PointerEvent('pointermove', {
                bubbles: true,
                pointerId: 1,
                buttons: 1,
                clientX: 1100 - frames * 6,
                clientY: 720 - frames * 2,
            }));
            if (frames < 60) requestAnimationFrame(tick);
            else resolve((frames / (performance.now() - start)) * 1000);
        };
        requestAnimationFrame(tick);
    }));
    await page.mouse.up();
    console.log(`500-card pan: ${panFps.toFixed(1)} fps`);
    expect(panFps).toBeGreaterThan(20);

    expect(await page.locator('.st-shot-card').count()).toBeGreaterThan(0);
});

test('200-clip timeline scroll stays interactive', async ({ page, request }) => {
    const created = await request.post(`${API}/projects`, {
        data: { name: 'Perf Timeline', resolution_w: 768, resolution_h: 512, fps: 25, seed: 42 },
    });
    const project = await created.json();
    const saved = await request.put(`${API}/projects/${project.id}`, {
        data: {
            id: project.id,
            name: 'Perf Timeline',
            fps: 25,
            seed: 42,
            resolution_w: 768,
            resolution_h: 512,
            revision: project.revision ?? 0,
            scenes: [],
            shots: Array.from({ length: 200 }, (_, index) => ({
                ...shot(project.id, index),
                num_frames: 25,
                status: 'completed',
            })),
        },
    });
    expect(saved.ok()).toBeTruthy();

    await page.goto('/studio');
    await expect(page.getByLabel('Project name')).toHaveValue('Perf Timeline', { timeout: 30000 });
    await page.keyboard.press('3');
    await expect(page.locator('.st-timeline-scroll')).toBeVisible();

    const mounted = await page.locator('.st-timeline-clip').count();
    console.log(`200-clip timeline mounted clips: ${mounted}`);
    expect(mounted).toBeLessThan(120);

    const fps = await page.evaluate(() => new Promise<number>((resolve) => {
        const scroll = document.querySelector('.st-timeline-scroll') as HTMLElement | null;
        let frames = 0;
        const start = performance.now();
        const tick = () => {
            frames += 1;
            if (scroll) scroll.scrollLeft = frames * 24;
            if (frames < 60) requestAnimationFrame(tick);
            else resolve((frames / (performance.now() - start)) * 1000);
        };
        requestAnimationFrame(tick);
    }));
    console.log(`200-clip timeline scroll: ${fps.toFixed(1)} fps`);
    expect(fps).toBeGreaterThan(20);
});
