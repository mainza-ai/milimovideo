import AxeBuilder from '@axe-core/playwright';
import { expect, test } from '@playwright/test';
import { llmSettings, stubStudioApi } from './stubs';

test.beforeEach(async ({ page }) => {
    await stubStudioApi(page);
});

async function openSetup(page: import('@playwright/test').Page) {
    await page.goto('/studio');
    await expect(page.getByText('Story spine')).toBeVisible();
    await page.getByRole('button', { name: 'Setup', exact: true }).click();
    await expect(page.getByRole('heading', { name: /AI & LLM/ })).toBeVisible();
}

test('studio exposes persisted LLM providers and role routing', async ({ page }) => {
    await openSetup(page);

    await expect(page.getByText('revision 0', { exact: true })).toBeVisible();
    await expect(page.getByTestId('llm-air-gap')).toHaveText(/air-gapped/);
    await expect(page.getByLabel('Ollama base URL')).toHaveValue('http://127.0.0.1:11434');
    await expect(page.getByLabel('Video prompt provider')).toHaveValue('gemma');
    await expect(page.getByLabel('Script parser provider')).toHaveValue('ollama');
    await expect(page.getByText('Gemma only serves video prompts')).toBeVisible();

    const results = await new AxeBuilder({ page })
        .withTags(['wcag2a', 'wcag2aa'])
        .include('[data-testid="llm-panel"]')
        .analyze();
    const blocking = results.violations.filter(
        (violation) => violation.impact === 'critical' || violation.impact === 'serious',
    );
    expect(blocking, JSON.stringify(blocking, null, 2)).toEqual([]);
});

test('saving sends the revision the panel read', async ({ page }) => {
    const puts: { url: string; body: Record<string, unknown> }[] = [];
    await page.route('**/settings/llm**', async (route) => {
        const request = route.request();
        const url = request.url();
        if (request.method() === 'PUT') {
            const body = request.postDataJSON();
            puts.push({ url, body });
            return route.fulfill({
                status: 200,
                contentType: 'application/json',
                body: JSON.stringify({
                    ...llmSettings(1),
                    providers: body.providers,
                    roles: body.roles,
                }),
            });
        }
        if (url.includes('/models')) {
            return route.fulfill({
                status: 200,
                contentType: 'application/json',
                body: JSON.stringify({ provider: 'ollama', models: [{ id: 'qwen3:8b', name: 'qwen3:8b' }] }),
            });
        }
        return route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify(llmSettings(0)),
        });
    });

    await openSetup(page);
    await page.getByLabel('Ollama default model').fill('gemma3:12b');
    const panel = page.getByTestId('llm-panel');
    await expect(panel.getByText('unsaved changes')).toBeVisible();
    await panel.getByRole('button', { name: 'Save' }).click();

    await expect(page.getByText('revision 1', { exact: true })).toBeVisible();
    await expect(panel.getByText('saved', { exact: true })).toBeVisible();
    expect(puts).toHaveLength(1);
    expect(puts[0].url).toContain('expected_revision=0');
    const providers = puts[0].body.providers as Record<string, { default_model: string }>;
    expect(providers.ollama.default_model).toBe('gemma3:12b');
});

test('discovery attaches a local server and verification reports it', async ({ page }) => {
    await openSetup(page);
    await page.getByRole('button', { name: /Scan local servers/ }).click();
    await expect(page.getByText('http://localhost:11434')).toBeVisible();
    await page.getByRole('button', { name: 'Attach' }).click();
    await expect(page.getByLabel('Ollama base URL')).toHaveValue('http://localhost:11434');
    await expect(page.getByText('revision 1', { exact: true })).toBeVisible();

    await page.getByRole('button', { name: /Verify/ }).nth(1).click();
    await expect(page.getByText(/ok · 9 ms/)).toBeVisible();
});

test('the saved plan can be streamed, evaluated, and staged by profile', async ({ page }) => {
    await openSetup(page);
    const panel = page.getByTestId('llm-panel');

    await panel.getByRole('button', { name: 'Run test' }).click();
    await expect(panel.getByText('A slow dolly-in on a rain-soaked street.')).toBeVisible();
    await expect(panel.getByText(/ollama\/qwen3:8b/)).toBeVisible();

    await panel.getByRole('button', { name: 'Run eval' }).click();
    await expect(panel.getByText(/last 100%/)).toBeVisible();
    await expect(panel.getByText('script_parse')).toBeVisible();

    await panel.getByRole('button', { name: 'Sovereign local' }).click();
    await expect(panel.getByTestId('llm-air-gap')).toHaveText(/air-gapped/);
});
