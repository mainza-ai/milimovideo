import { expect, test } from '@playwright/test';

const API = 'http://127.0.0.1:8010';

// 64×64 quadrant PNG: TL red, TR green, BL blue, BR white.
const QUADRANT_PNG = Buffer.from(
    'iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAkklEQVR4nNXOUQ2AMADE0K4JDqYD/1/YmIPpQARZSJ+A3o09Jyfdz9E8EidxEidxEidxEidxEidxEidxEidxEidxEidxEidxEidxEidxEidxEidxEidxEjfm3EcH1rqO9iVO4iRO4iRO4iRO4iRO4iRO4iRO4iRO4iRO4iRO4iRO4iRO4iRO4iRO4iRO4vz7wFcv/L4FOnEQC8AAAAAASUVORK5CYII=',
    'base64',
);

const SWAP_CUBE = [
    'LUT_3D_SIZE 2',
    ...Array.from({ length: 8 }, (_, index) => {
        const red = index % 2;
        const green = Math.floor(index / 2) % 2;
        const blue = Math.floor(index / 4) % 2;
        return `${blue} ${green} ${red}`;
    }),
].join('\n');

async function upload(
    request: import('@playwright/test').APIRequestContext,
    projectId: string,
    name: string,
    mimeType: string,
    buffer: Buffer,
) {
    const response = await request.post(`${API}/upload`, {
        multipart: {
            project_id: projectId,
            file: { name, mimeType, buffer },
        },
    });
    expect(response.ok(), await response.text()).toBe(true);
    return await response.json();
}

test('edit preview applies a cube LUT to a still image', async ({ page, request }) => {
    const project = await (await request.post(`${API}/projects`, {
        data: { name: 'LUT E2E', resolution_w: 1024, resolution_h: 576, fps: 25 },
    })).json();
    const image = await upload(request, project.id, 'quad.png', 'image/png', QUADRANT_PNG);
    const lut = await upload(request, project.id, 'swap.cube', 'text/plain', Buffer.from(SWAP_CUBE));

    const state = {
        ...project,
        shots: [{
            id: `lut-shot-${project.id}`,
            project_id: project.id,
            index: 0,
            sequence_index: 0,
            prompt: 'quadrants',
            seed: 1,
            width: 64,
            height: 64,
            num_frames: 1,
            fps: 25,
            track_index: 0,
            start_frame: 0,
            trim_in: 0,
            trim_out: 0,
            timeline: [],
            video_url: image.url,
            thumbnail_url: image.url,
            status: 'completed',
        }],
        render_lut: lut.url,
    };
    const save = await request.put(`${API}/projects/${project.id}`, { data: state });
    expect(save.ok(), await save.text()).toBe(true);

    await page.addInitScript((id) => localStorage.setItem('milimo_last_project_id', id), project.id);
    await page.goto('/studio');
    await expect(page.getByText('Story spine')).toBeVisible();
    await page.getByRole('button', { name: 'Edit', exact: true }).first().click();

    const canvas = page.locator('.st-lut-canvas');
    await expect(canvas).toBeVisible({ timeout: 10_000 });
    await expect(page.locator('.st-status-pill', { hasText: 'LUT on export' })).toHaveCount(0);

    const pixels = await page.evaluate(() => {
        const element = document.querySelector<HTMLCanvasElement>('.st-lut-canvas');
        if (!element) throw new Error('no LUT canvas');
        const context = element.getContext('2d');
        if (!context) throw new Error('no 2D context');
        const at = (fx: number, fy: number) => Array.from(
            context.getImageData(
                Math.floor(element.width * fx),
                Math.floor(element.height * fy),
                1,
                1,
            ).data.slice(0, 3),
        );
        return {
            width: element.width,
            height: element.height,
            TL: at(0.25, 0.25),
            TR: at(0.75, 0.25),
            BL: at(0.25, 0.75),
            BR: at(0.75, 0.75),
        };
    });

    expect(pixels.width).toBeGreaterThan(1);
    const near = (actual: number[], expected: number[]) => expected.every(
        (value, index) => Math.abs(actual[index] - value) <= 60,
    );
    // Swap R/B: TL blue, TR green, BL red, BR white.
    expect(near(pixels.TL, [0, 0, 255]), JSON.stringify(pixels)).toBe(true);
    expect(near(pixels.TR, [0, 255, 0]), JSON.stringify(pixels)).toBe(true);
    expect(near(pixels.BL, [255, 0, 0]), JSON.stringify(pixels)).toBe(true);
    expect(near(pixels.BR, [255, 255, 255]), JSON.stringify(pixels)).toBe(true);
});
