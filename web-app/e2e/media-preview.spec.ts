import { expect, test } from '@playwright/test';

const API = process.env.MILIMO_E2E_API || 'http://127.0.0.1:8010';
const PNG = Buffer.from(
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=',
    'base64',
);

test('library media opens in the global preview and closes with Escape', async ({ page, request }) => {
    const created = await request.post(`${API}/projects`, {
        data: { name: 'Media Preview', resolution_w: 768, resolution_h: 512, fps: 25, seed: 42 },
    });
    expect(created.ok()).toBeTruthy();
    const project = await created.json();

    const upload = await request.post(`${API}/upload`, {
        multipart: {
            file: { name: 'preview-pixel.png', mimeType: 'image/png', buffer: PNG },
            project_id: project.id,
            type: 'general',
        },
    });
    expect(upload.ok()).toBeTruthy();
    const asset = await upload.json();
    const filename = asset.url.split('/').pop();

    await page.goto('/studio');
    await expect(page.getByLabel('Project name')).toHaveValue('Media Preview', { timeout: 15000 });
    await page.getByRole('button', { name: 'Library', exact: true }).first().click();
    await page.getByRole('button', { name: new RegExp(`Preview ${filename}`) }).click();

    const dialog = page.getByRole('dialog', { name: filename });
    await expect(dialog).toBeVisible();
    await expect(dialog.locator('img')).toBeVisible();
    await page.keyboard.press('Escape');
    await expect(dialog).toBeHidden();
});
