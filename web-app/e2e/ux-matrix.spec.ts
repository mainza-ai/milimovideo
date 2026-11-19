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
    // Read both from the same DOM snapshot; project bootstrap may settle one
    // tick after the shell paints.
    const { pill, footer } = await page.evaluate(() => {
        const pillElement = [...document.querySelectorAll('.st-status-pill')].find((element) =>
            /Saved|Unsaved|Not saved|Saving|conflict|failed/.test(element.textContent || ''),
        );
        return {
            pill: (pillElement?.textContent || '').replace(/\s+/g, ' ').trim(),
            footer: (document.querySelector('.st-status-save')?.textContent || '').replace(/\s+/g, ' ').trim(),
        };
    });
    const footerStatus = footer.replace(/^Save\s*/i, '');
    expect(footerStatus, `pill="${pill}" footer="${footer}"`).toBe(pill);
});

test('mobile story has a usable board and a shot-list drawer', async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await openStudio(page);
    const board = await page.locator('.st-board-wrap').boundingBox();
    expect(board?.width ?? 0).toBeGreaterThanOrEqual(320);

    const toggle = page.getByRole('button', { name: 'Show shot list' });
    await toggle.click();
    await expect(page.getByText('Story spine')).toBeVisible();
    const spine = page.locator('.st-spine');
    await expect(spine).toHaveAttribute('role', 'dialog');
    await expect(spine).toHaveAttribute('aria-modal', 'true');

    await page.keyboard.press('Escape');
    await expect(page.locator('.st-mobile-backdrop')).toHaveCount(0);
    await expect(toggle).toBeFocused();
});

test('mobile edit can expand the timeline and reach every track', async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await openStudio(page);
    await page.getByRole('button', { name: 'Edit', exact: true }).first().click();
    await page.waitForTimeout(300);

    await page.getByRole('button', { name: 'Expand timeline' }).click();
    const expanded = await page.evaluate(() => ({
        stage: document.querySelector('.st-stage')
            ? getComputedStyle(document.querySelector('.st-stage')!).display : null,
        timeline: document.querySelector('.st-timeline')?.getBoundingClientRect().height ?? 0,
        tracks: document.querySelectorAll('.st-timeline-track').length,
    }));
    expect(expanded.stage).toBe('none');
    expect(expanded.timeline).toBeGreaterThan(400);
    expect(expanded.tracks).toBe(5);
    await expect(page.getByRole('button', { name: 'Collapse timeline' })).toBeVisible();
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
