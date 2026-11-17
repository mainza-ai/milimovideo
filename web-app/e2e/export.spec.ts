import { execFileSync } from 'node:child_process';
import { mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { expect, test } from '@playwright/test';

const API = process.env.MILIMO_E2E_API || 'http://127.0.0.1:8010';
const PROJECTS = process.env.MILIMO_E2E_PROJECTS || '/tmp/milimo-e2e-projects';

test.describe.configure({ timeout: 120_000 });

test('export the computed cut and wait for the render output', async ({ page, request }) => {
    let hasFfmpeg = true;
    try {
        execFileSync('ffmpeg', ['-version'], { stdio: 'ignore' });
    } catch {
        hasFfmpeg = false;
    }
    test.skip(!PROJECTS || !hasFfmpeg, 'ffmpeg or E2E projects dir not available');

    const created = await request.post(`${API}/projects`, {
        data: { name: 'E2E Export', resolution_w: 320, resolution_h: 180, fps: 25, seed: 42 },
    });
    expect(created.ok()).toBeTruthy();
    const project = await created.json();
    const pid = project.id as string;

    const generated = join(PROJECTS, pid, 'generated');
    mkdirSync(generated, { recursive: true });
    execFileSync('ffmpeg', [
        '-y', '-f', 'lavfi', '-i', 'color=c=blue:s=320x180:r=25:d=1',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', join(generated, 'clip.mp4'),
    ], { stdio: 'ignore' });

    const saved = await request.put(`${API}/projects/${pid}`, {
        data: {
            id: pid,
            name: 'E2E Export',
            fps: 25,
            seed: 42,
            resolution_w: 320,
            resolution_h: 180,
            revision: project.revision ?? 0,
            scenes: [],
            shots: [{
                id: `export-shot-${pid}`,
                prompt: 'blue clip',
                negative_prompt: '',
                seed: 42,
                width: 320,
                height: 180,
                fps: 25,
                num_frames: 25,
                cfg_scale: 1.0,
                enhance_prompt: false,
                upscale: false,
                pipeline_override: 'auto',
                track_index: 0,
                start_frame: 0,
                trim_in: 0,
                trim_out: 0,
                timeline: [],
                video_url: `/projects/${pid}/generated/clip.mp4`,
                status: 'completed',
            }],
        },
    });
    expect(saved.ok()).toBeTruthy();

    await page.goto('/studio');
    await expect(page.getByLabel('Project name')).toHaveValue('E2E Export', { timeout: 15000 });

    await page.getByRole('button', { name: 'Deliver', exact: true }).first().click();
    await page.getByRole('button', { name: 'Create export' }).click();

    await expect(page.getByRole('link', { name: /open/ })).toBeVisible({ timeout: 90_000 });
    await expect(page.getByText('completed').first()).toBeVisible();
});
