import { expect, test } from '@playwright/test';

const API = 'http://127.0.0.1:8010';

test('create a shot, save, clear local state, and reload from the server', async ({ page, request }) => {
    const created = await request.post(`${API}/projects`, {
        data: { name: 'E2E Film', resolution_w: 768, resolution_h: 512, fps: 25, seed: 42 },
    });
    expect(created.ok()).toBeTruthy();

    await page.goto('/studio');
    await expect(page.getByLabel('Project name')).toHaveValue('E2E Film', { timeout: 15000 });

    await page.getByRole('button', { name: 'Add shot' }).click();
    await expect(page.getByText('New shot').first()).toBeVisible();

    const prompt = page.locator('#studio-prompt');
    await expect(prompt).toBeVisible();
    await prompt.fill('A lighthouse in a storm');

    await page.getByRole('button', { name: /Unsaved|Saved/ }).click();
    await expect(page.getByRole('button', { name: /Saved/ })).toBeVisible({ timeout: 15000 });

    // Force a server load instead of passing via persisted local state.
    await page.evaluate(() => localStorage.clear());
    await page.reload();
    await expect(page.getByLabel('Project name')).toHaveValue('E2E Film', { timeout: 15000 });
    await expect(page.getByText('A lighthouse in a storm').first()).toBeVisible({ timeout: 15000 });
});
