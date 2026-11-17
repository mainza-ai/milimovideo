import { expect, test } from '@playwright/test';
import { stubStudioApi } from './stubs';

test.beforeEach(async ({ page }) => {
    await stubStudioApi(page);
});

test('studio shell, board, review and setup render', async ({ page }) => {
    await page.goto('/studio');

    await expect(page.getByText('Story spine')).toBeVisible();
    await expect(page.getByText('Milimo', { exact: true })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Area' })).toBeVisible();
    await expect(page.getByText(/Master references/)).toBeVisible();

    await page.getByRole('button', { name: 'Switch project' }).click();
    await expect(page.getByText('Audit Test')).toBeVisible();

    await page.getByRole('button', { name: 'Review', exact: true }).first().click();
    await expect(page.getByText(/Comment at frame/)).toBeVisible();
    await expect(page.getByRole('button', { name: /Approve shot/ })).toBeVisible();

    await page.getByRole('button', { name: 'Setup', exact: true }).click();
    await expect(page.getByRole('heading', { name: 'Memory slots' })).toBeVisible();
    await expect(page.getByRole('heading', { name: 'Models' })).toBeVisible();
});
