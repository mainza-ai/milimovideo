import { expect, test } from '@playwright/test';
import { stubStudioApi } from './stubs';

test.beforeEach(async ({ page }) => {
    await stubStudioApi(page);
    await page.setViewportSize({ width: 1440, height: 900 });
});

test('studio shell visual baseline', async ({ page }) => {
    await page.goto('/studio');
    await expect(page.getByText('Story spine')).toBeVisible();
    await expect(page.locator('.st-shot-card').first()).toBeVisible();
    await page.locator('.st-board-toolbar').getByRole('button', { name: /More/ }).click();
    await page.getByRole('menuitem', { name: /minimap/i }).click();
    await page.waitForTimeout(250);
    await expect(page).toHaveScreenshot('studio-shell.png', {
        animations: 'disabled',
        caret: 'hide',
        maxDiffPixelRatio: 0.2,
    });
});
