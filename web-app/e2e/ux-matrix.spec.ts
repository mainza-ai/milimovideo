import { expect, test } from '@playwright/test';
import { stubStudioApi } from './stubs';

const SURFACES = ['Story', 'Generate', 'Edit', 'Review', 'Library', 'Deliver', 'Shorts', 'Setup'];

async function openStudio(page: import('@playwright/test').Page) {
    await stubStudioApi(page);
    await page.goto('/studio');
    await expect(page.getByText('Story spine')).toBeVisible();
}

for (const viewport of [
    { width: 1440, height: 900, name: 'desktop' },
    { width: 390, height: 844, name: 'mobile' },
]) {
    test.describe(`${viewport.name} surface matrix`, () => {
        test.use({ viewport: { width: viewport.width, height: viewport.height } });

        test(`all surfaces fit ${viewport.name} without overflow or tiny controls`, async ({ page }) => {
            await openStudio(page);
            for (const surface of SURFACES) {
                const button = page.getByRole('button', { name: surface, exact: true }).first();
                if (await button.count()) await button.click();
                await page.waitForTimeout(200);

                const metrics = await page.evaluate(() => {
                    const buttons = [...document.querySelectorAll('button')]
                        .map((element) => element.getBoundingClientRect())
                        .filter((rect) => rect.width > 0 && rect.height > 0);
                    const text = [...document.querySelectorAll('body *')].filter(
                        (element) => element.childNodes.length === 1
                            && element.firstChild?.nodeType === 3
                            && element.textContent?.trim(),
                    );
                    return {
                        overflow: document.documentElement.scrollWidth > window.innerWidth,
                        smallText: text.filter((element) => parseFloat(getComputedStyle(element).fontSize) < 10).length,
                        smallButtons: buttons.filter((rect) => rect.height < 28).length,
                    };
                });
                expect(metrics.overflow, `${surface} overflows at ${viewport.name}`).toBe(false);
                expect(metrics.smallText, `${surface} has text under 10px`).toBe(0);
                expect(metrics.smallButtons, `${surface} has controls under 28px`).toBe(0);
            }
        });
    });
}

test('save pill and status strip agree after load', async ({ page }) => {
    await openStudio(page);
    const pill = await page.locator('.st-status-pill').filter({ hasText: /Saved|Unsaved|Not saved|Saving|conflict|failed/ }).first().textContent();
    const footer = await page.locator('.st-status-save').textContent();
    const pillText = (pill || '').replace(/\s+/g, ' ').trim();
    const footerText = (footer || '').replace(/\s+/g, ' ').trim();
    expect(footerText).toContain(pillText.replace(/^.*?(Saved|Unsaved|Not saved|Saving…|Save conflict|Save failed).*$/s, '$1'));
});

test('mobile story has a usable board and a shot-list drawer', async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await openStudio(page);
    const board = await page.locator('.st-board-wrap').boundingBox();
    expect(board?.width ?? 0).toBeGreaterThanOrEqual(320);

    await page.getByRole('button', { name: 'Show shot list' }).click();
    await expect(page.getByText('Story spine')).toBeVisible();
    await page.getByRole('button', { name: 'Close story spine' }).click();
});

test('generate defaults to one right rail and hides params when the inspector opens', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 });
    await openStudio(page);
    await page.getByRole('button', { name: 'Generate', exact: true }).first().click();
    await page.waitForTimeout(300);

    const before = await page.evaluate(() => ({
        stage: document.querySelector('.st-shot-stage')?.getBoundingClientRect().width ?? 0,
        params: document.querySelector('.st-shot-params') ? getComputedStyle(document.querySelector('.st-shot-params')!).display !== 'none' : false,
        inspector: !!document.querySelector('.st-inspector'),
    }));
    expect(before.inspector).toBe(false);

    const toggle = page.getByRole('button', { name: /inspector/i }).first();
    if (await toggle.count()) {
        await toggle.click();
        await page.waitForTimeout(200);
        const after = await page.evaluate(() => ({
            stage: document.querySelector('.st-shot-stage')?.getBoundingClientRect().width ?? 0,
            params: document.querySelector('.st-shot-params') ? getComputedStyle(document.querySelector('.st-shot-params')!).display !== 'none' : false,
            inspector: !!document.querySelector('.st-inspector'),
        }));
        expect(after.inspector).toBe(true);
        expect(after.params).toBe(false);
        expect(after.stage).toBeGreaterThanOrEqual(600);
    }
});
