import { expect, test } from '@playwright/test';

const API = 'http://127.0.0.1:8010';

test('reviewer @Agent note becomes an approved prompt patch', async ({ page, request }) => {
    const created = await request.post(`${API}/projects`, {
        data: { name: 'Agent Review', resolution_w: 768, resolution_h: 512, fps: 25, seed: 42 },
    });
    expect(created.ok()).toBeTruthy();
    const project = await created.json();
    const pid = project.id as string;
    const shotId = `agent-shot-${pid}`;

    const saved = await request.put(`${API}/projects/${pid}`, {
        data: {
            id: pid,
            name: 'Agent Review',
            fps: 25,
            seed: 42,
            resolution_w: 768,
            resolution_h: 512,
            revision: project.revision ?? 0,
            scenes: [],
            shots: [{
                id: shotId,
                prompt: 'a noisy street',
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
            }],
        },
    });
    expect(saved.ok()).toBeTruthy();

    const share = await request.post(`${API}/projects/${pid}/share`);
    expect(share.ok()).toBeTruthy();
    const { token } = await share.json();

    await page.goto(`/studio/share/${token}`);
    await expect(page.getByText('Agent Review')).toBeVisible();
    await page.getByLabel('Comment at frame 0').fill('@Agent prompt: a quiet street at dawn');
    await page.getByRole('button', { name: /Add comment/ }).click();
    await expect(page.getByText(/quiet street/)).toBeVisible();
    await page.getByRole('button', { name: /Approve/ }).click();
    await expect(page.getByText('approved').first()).toBeVisible();

    await page.goto('/studio');
    await expect(page.getByLabel('Project name')).toHaveValue('Agent Review', { timeout: 15000 });
    await page.getByRole('button', { name: 'Review', exact: true }).first().click();
    await page.getByRole('button', { name: /Plan fix in command dock/ }).click();

    const input = page.locator('.st-command-input');
    await expect(input).toHaveValue(/quiet street/);
    await page.getByRole('button', { name: /^Plan$/ }).click();
    await expect(page.getByText('Plan preview')).toBeVisible();
    await page.getByRole('button', { name: /Approve & apply/ }).click();
    await expect(page.locator('#studio-prompt')).toHaveValue('a quiet street at dawn');
});
