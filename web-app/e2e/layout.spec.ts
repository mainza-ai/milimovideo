import { expect, test } from '@playwright/test';
import { stubStudioApi } from './stubs';

test.beforeEach(async ({ page }) => {
    await stubStudioApi(page);
});

test('studio chrome stays inside its containers at every supported width', async ({ page }) => {
    for (const width of [2048, 1440, 1280, 1024, 900, 768]) {
        await page.setViewportSize({ width, height: 900 });
        await page.goto('/studio');
        await expect(page.getByText('Story spine')).toBeVisible({ timeout: 15000 });

        const result = await page.evaluate(() => {
            const visible = (el: Element | null): el is HTMLElement => {
                if (!el) return false;
                const rect = el.getBoundingClientRect();
                const style = getComputedStyle(el);
                return style.display !== 'none' && rect.width > 0 && rect.height > 0;
            };
            const clips = (el: HTMLElement) => el.scrollWidth > el.clientWidth + 1;
            const railClipped = [...document.querySelectorAll<HTMLElement>('.st-rail-button > span')]
                .filter(visible)
                .filter(clips)
                .map((el) => el.textContent);
            const inspector = document.querySelector<HTMLElement>('.st-inspector-tabs');
            const toolbar = document.querySelector<HTMLElement>('.st-board-toolbar');
            const toolbarOutside = toolbar
                ? [...toolbar.querySelectorAll<HTMLElement>('button, input, select, .st-menu-wrap')]
                    .filter(visible)
                    .map((el) => ({ el, rect: el.getBoundingClientRect(), box: toolbar.getBoundingClientRect() }))
                    .filter(({ rect, box }) => rect.right > box.right + 1 || rect.bottom > box.bottom + 1)
                    .map(({ el }) => el.textContent?.trim() || el.getAttribute('aria-label'))
                : [];
            const tabs = [...document.querySelectorAll<HTMLElement>('.st-surface-tab')]
                .filter(visible)
                .map((el) => {
                    const rect = el.getBoundingClientRect();
                    const hit = document.elementFromPoint(rect.x + rect.width / 2, rect.y + rect.height / 2);
                    return { text: el.textContent, hit: hit?.className?.toString?.() || hit?.tagName };
                });
            return {
                bodyOverflow: document.documentElement.scrollWidth > document.documentElement.clientWidth + 1,
                railClipped,
                inspectorOverflow: !!inspector && inspector.scrollWidth > inspector.clientWidth + 1,
                toolbarOutside,
                overlappingTabs: tabs.filter((tab) => tab.hit !== 'st-surface-tab'),
                compactVisible: visible(document.querySelector('.st-surface-compact')),
                tabsVisible: tabs.length > 0,
            };
        });

        expect(result.bodyOverflow, `body overflow at ${width}`).toBe(false);
        expect(result.railClipped, `clipped rail labels at ${width}`).toEqual([]);
        expect(result.inspectorOverflow, `inspector tab overflow at ${width}`).toBe(false);
        expect(result.toolbarOutside, `toolbar items outside at ${width}`).toEqual([]);
        expect(result.overlappingTabs, `overlapping top-bar tabs at ${width}`).toEqual([]);
        if (width < 1200) {
            expect(result.compactVisible, `compact surface switcher at ${width}`).toBe(true);
            expect(result.tabsVisible, `wide tabs hidden at ${width}`).toBe(false);
        }
        if (width >= 1200) {
            expect(result.tabsVisible, `wide tabs visible at ${width}`).toBe(true);
        }
    }
});

test('compact surface switcher reaches Generate at 1024px', async ({ page }) => {
    await page.setViewportSize({ width: 1024, height: 900 });
    await page.goto('/studio');
    await page.getByRole('button', { name: 'Switch studio surface' }).click();
    await page.getByRole('menuitem', { name: 'Generate', exact: true }).click();
    await expect(page.getByRole('button', { name: 'Switch studio surface' })).toHaveText(/Generate/);
    await expect(page.getByText(/Shot queue|No shot selected/).first()).toBeVisible();
});

test('command palette changes density and opens the shortcut sheet', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.goto('/studio');
    await expect(page.getByText('Story spine')).toBeVisible({ timeout: 15000 });
    await page.keyboard.press('Meta+k');
    const palette = page.getByRole('dialog', { name: 'Command palette' });
    await expect(palette).toBeVisible();
    await palette.getByLabel('Search commands').fill('compact density');
    await page.keyboard.press('Enter');
    await expect(page.locator('.studio-root')).toHaveAttribute('data-density', 'compact');

    await expect(palette).toBeHidden();
    await page.keyboard.press('?');
    await expect(page.getByRole('dialog', { name: 'Keyboard shortcuts' })).toBeVisible();
});
