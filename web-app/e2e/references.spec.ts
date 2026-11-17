import { expect, test } from '@playwright/test';

const API = process.env.MILIMO_E2E_API || 'http://127.0.0.1:8010';

test('create, edit, and delete a master reference in the studio library', async ({ page, request }) => {
    const created = await request.post(`${API}/projects`, {
        data: { name: 'Library References', resolution_w: 768, resolution_h: 512, fps: 25, seed: 42 },
    });
    expect(created.ok()).toBeTruthy();

    await page.goto('/studio');
    await expect(page.getByLabel('Project name')).toHaveValue('Library References', { timeout: 15000 });
    await page.getByRole('button', { name: 'Library', exact: true }).first().click();
    await page.getByRole('button', { name: /Master references/ }).click();

    await page.getByRole('button', { name: 'New reference' }).click();
    await page.getByLabel('Name', { exact: true }).fill('Mara');
    await page.getByLabel('Trigger word').fill('mara');
    await page.getByLabel('Reference description').fill('Lead pilot in a red jacket');
    await page.getByRole('button', { name: 'Create reference' }).click();

    await expect(page.getByText('Mara', { exact: true })).toBeVisible({ timeout: 15000 });
    await expect(page.getByText(/Lead pilot in a red jacket/)).toBeVisible();

    await page.getByRole('button', { name: 'Edit Mara' }).click();
    await page.getByLabel('Reference description').fill('Lead pilot, red jacket, scar');
    await page.getByRole('button', { name: 'Save reference' }).click();
    await expect(page.getByText(/red jacket, scar/)).toBeVisible({ timeout: 15000 });

    page.once('dialog', (dialog) => void dialog.accept());
    await page.getByRole('button', { name: 'Delete Mara' }).click();
    await expect(page.getByText('Mara', { exact: true })).toHaveCount(0, { timeout: 15000 });
});
