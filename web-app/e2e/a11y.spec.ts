import AxeBuilder from '@axe-core/playwright';
import { expect, test } from '@playwright/test';
import { stubStudioApi } from './stubs';

test.beforeEach(async ({ page }) => {
    await stubStudioApi(page);
});

test('studio shell has no critical accessibility violations', async ({ page }) => {
    await page.goto('/studio');
    await expect(page.getByText('Story spine')).toBeVisible();

    const results = await new AxeBuilder({ page })
        .withTags(['wcag2a', 'wcag2aa'])
        .analyze();

    const critical = results.violations.filter((violation) => violation.impact === 'critical');
    expect(critical, JSON.stringify(critical, null, 2)).toEqual([]);
});

test('keyboard reaches the workspace and mode surfaces', async ({ page }) => {
    await page.goto('/studio');
    await expect(page.getByText('Story spine')).toBeVisible();

    await page.keyboard.press('2');
    await expect(page.getByRole('heading', { name: /Shot/ })).toBeVisible();
    await page.keyboard.press('8');
    await expect(page.getByRole('heading', { name: 'Memory slots' })).toBeVisible();
});
