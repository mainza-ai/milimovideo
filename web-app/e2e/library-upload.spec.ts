import { expect, test } from '@playwright/test';

const API = process.env.MILIMO_E2E_API || 'http://127.0.0.1:8010';
const PNG = Buffer.from(
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=',
    'base64',
);

test('upload media from the studio library', async ({ page, request }) => {
    const created = await request.post(`${API}/projects`, {
        data: { name: 'Library Upload', resolution_w: 768, resolution_h: 512, fps: 25, seed: 42 },
    });
    expect(created.ok()).toBeTruthy();

    await page.goto('/studio');
    await expect(page.getByLabel('Project name')).toHaveValue('Library Upload', { timeout: 15000 });
    await page.getByRole('button', { name: 'Library', exact: true }).first().click();
    await expect(page.getByRole('button', { name: /Upload media/ })).toBeVisible();

    await page.getByLabel('Upload media').setInputFiles({
        name: 'studio-pixel.png',
        mimeType: 'image/png',
        buffer: PNG,
    });

    await expect(page.getByText(/studio-pixel|Uploaded/).first()).toBeVisible({ timeout: 15000 });
});
