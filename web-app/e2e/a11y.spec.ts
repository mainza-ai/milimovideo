import AxeBuilder from '@axe-core/playwright';
import { expect, test } from '@playwright/test';
import { stubStudioApi } from './stubs';

test.beforeEach(async ({ page }) => {
    await stubStudioApi(page);
});

test('studio shell has no serious or critical accessibility violations', async ({ page }) => {
    await page.goto('/studio');
    await expect(page.getByText('Story spine')).toBeVisible();

    const results = await new AxeBuilder({ page })
        .withTags(['wcag2a', 'wcag2aa'])
        .analyze();

    const blocking = results.violations.filter(
        (violation) => violation.impact === 'critical' || violation.impact === 'serious',
    );
    expect(blocking, JSON.stringify(blocking, null, 2)).toEqual([]);
});

test('keyboard reaches the workspace and mode surfaces', async ({ page }) => {
    await page.goto('/studio');
    await expect(page.getByText('Story spine')).toBeVisible();

    await page.keyboard.press('2');
    await expect(page.getByRole('heading', { name: /Shot/ })).toBeVisible();
    await page.keyboard.press('8');
    await expect(page.getByRole('heading', { name: 'Memory slots' })).toBeVisible();
});

test('every studio surface is free of serious accessibility violations', async ({ page }) => {
    await page.goto('/studio');
    await expect(page.getByText('Story spine')).toBeVisible();

    for (const surface of ['Generate', 'Edit', 'Review', 'Library', 'Deliver', 'Shorts', 'Setup']) {
        await page.getByRole('button', { name: surface, exact: true }).first().click();
        await page.waitForTimeout(150);
        const results = await new AxeBuilder({ page })
            .withTags(['wcag2a', 'wcag2aa'])
            .analyze();
        const blocking = results.violations.filter(
            (violation) => violation.impact === 'critical' || violation.impact === 'serious',
        );
        expect(blocking, `${surface}: ${JSON.stringify(blocking, null, 2)}`).toEqual([]);
    }
});
