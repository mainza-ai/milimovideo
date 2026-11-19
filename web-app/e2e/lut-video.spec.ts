import { expect, test } from '@playwright/test';

const API = 'http://127.0.0.1:8010';
const QUADRANT_WEBM = Buffer.from('GkXfo59ChoEBQveBAULygQRC84EIQoKEd2VibUKHgQJChYECGFOAZwEAAAAAAATeEU2bdLpNu4tTq4QVSalmU6yBoU27i1OrhBZUrmtTrIHWTbuMU6uEElTDZ1OsggEyTbuMU6uEHFO7a1OsggTI7AEAAAAAAABZAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAVSalmsCrXsYMPQkBNgIxMYXZmNjMuMS4xMDJXQYxMYXZmNjMuMS4xMDJEiYhAj0AAAAAAABZUrmvXrgEAAAAAAABO14EBc8WI1RNZXlfXqROcgQAitZyDdW5kiIEAhoVWX1ZQOIOBASPjg4QCYloA4JCwgUC6gUCagQJVsIRVuYEBVe6BAOwBAAAAAAAAAgAAElTDZ/pzc59jwIBnyJlFo4dFTkNPREVSRIeMTGF2ZjYzLjEuMTAyc3PVY8CLY8WI1RNZXlfXqRNnyKBFo4dFTkNPREVSRIeTTGF2YzYzLjEuMTAyIGxpYnZweGfIoUWjiERVUkFUSU9ORIeTMDA6MDA6MDEuMDAwMDAwMDAwAB9DtnVDEeeBAKNAvYEAAIAwCACdASpAAEAAAEcIhYWIhYSIAgICddcD+A/gDylVfuA/gD+wH9A5xnQXsD+wG4RMA0APiA+v/d0HsA//8S4almUyCK0t1s7FDED+/Lfi/oCOzxRA8h044eRnE5pEBWdL2UA+fkAAX4leRvsAB9C8N/aYAAEswBLMAABgGAAAAIl/ARL+AAApbgFLcH/9YfjPjSZjoroLkLPe0/oIt4iYZWcX/wxBHpQqfT6dOtCREINK3lMBLbPxAKOkgQAoABECAAEQEAAYABnBf/QAXBvzxNQAAAR9AEswAAGgYAAAo5aBAFAA0QEAARAQABgAGFgv9AAIjoAAo5aBAHgA0QEAARAQABgAGFgv9AAIjoAAo5aBAKAA0QEAARAQABgAGFgv9AAIjoAAo5aBAMgA0QEAARAQABgAGFgv9AAIjoAAo5aBAPAA0QEAARAQABgAGFgv9AAIjoAAo5aBARgA0QEAARAQFGAAYWC/0AAiOgAAo5aBAUAA0QEAARAQABgAGFgv9AAIjoAAo5aBAWgA0QEAARAQABgAGFgv9AAIjoAAo5aBAZAA0QEAARAQABgAGFgv9AAIjoAAo5aBAbgA0QEAARAQABgAGFgv9AAIjoAAo5aBAeAA0QEAARAQABgAGFgv9AAIjoAAo5aBAggA0QEAARAQABgAGFgv9AAIjoAAo5aBAjAA0QEAARAQABgAGFgv9AAIjoAAo5aBAlgA0QEAARAQABgAGFgv9AAIjoAAo5aBAoAA0QEAARAQABgAGFgv9AAIjoAAo5aBAqgA0QEAARAQABgAGFgv9AAIjoAAo5aBAtAA0QEAARAQFGAAYWC/0AAiOgAAo5aBAvgA0QEAARAQABgAGFgv9AAIjoAAo5aBAyAA0QEAARAQABgAGFgv9AAIjoAAo5aBA0gA0QEAARAQABgAGFgv9AAIjoAAo5aBA3AA0QEAARAQABgAGFgv9AAIjoAAo5aBA5gA0QEAARAQABgAGFgv9AAIjoAAo5aBA8AA0QEAARAQABgAGFgv9AAIjoAAHFO7a5G7j7OBALeK94EB8YIBsfCBAw==', 'base64');
const SWAP_CUBE = ['LUT_3D_SIZE 2', ...Array.from({ length: 8 }, (_, index) => {
    const red = index % 2;
    const green = Math.floor(index / 2) % 2;
    const blue = Math.floor(index / 4) % 2;
    return `${blue} ${green} ${red}`;
})].join('\n');

async function upload(request: import('@playwright/test').APIRequestContext, projectId: string, name: string, mimeType: string, buffer: Buffer) {
    const response = await request.post(`${API}/upload`, {
        multipart: { project_id: projectId, file: { name, mimeType, buffer } },
    });
    expect(response.ok(), await response.text()).toBe(true);
    return await response.json();
}

test('video playback applies the cube LUT and redraws frames', async ({ page, request }) => {
    const project = await (await request.post(`${API}/projects`, {
        data: { name: 'LUT Video', resolution_w: 1024, resolution_h: 576, fps: 25 },
    })).json();
    const video = await upload(request, project.id, 'quad.webm', 'video/webm', QUADRANT_WEBM);
    const lut = await upload(request, project.id, 'swap.cube', 'text/plain', Buffer.from(SWAP_CUBE));
    const state = {
        ...project,
        shots: [{
            id: `video-${project.id}`,
            project_id: project.id,
            index: 0,
            sequence_index: 0,
            prompt: 'video quadrants',
            seed: 1,
            width: 64,
            height: 64,
            num_frames: 25,
            fps: 25,
            track_index: 0,
            start_frame: 0,
            trim_in: 0,
            trim_out: 0,
            timeline: [],
            video_url: video.url,
            status: 'completed',
        }],
        render_lut: lut.url,
    };
    expect((await request.put(`${API}/projects/${project.id}`, { data: state })).ok()).toBe(true);

    await page.addInitScript((id) => localStorage.setItem('milimo_last_project_id', id), project.id);
    await page.goto('/studio');
    await expect(page.getByText('Story spine')).toBeVisible();
    await page.getByRole('button', { name: 'Edit', exact: true }).first().click();

    const canvas = page.locator('.st-lut-canvas');
    await expect(canvas).toBeVisible({ timeout: 10_000 });
    await page.evaluate(async () => {
        const element = document.querySelector<HTMLVideoElement>('.st-stage-primary video');
        if (!element) throw new Error('no video');
        element.muted = true;
        await element.play().catch(() => undefined);
    });
    await page.waitForTimeout(900);

    const result = await page.evaluate(() => {
        const element = document.querySelector<HTMLCanvasElement>('.st-lut-canvas');
        if (!element) throw new Error('no LUT canvas');
        const context = element.getContext('2d');
        if (!context) throw new Error('no 2D context');
        const at = (fx: number, fy: number) => Array.from(
            context.getImageData(Math.floor(element.width * fx), Math.floor(element.height * fy), 1, 1).data.slice(0, 3),
        );
        return {
            frames: Number(element.dataset.frames || 0),
            TL: at(0.25, 0.25),
            BR: at(0.75, 0.75),
        };
    });
    expect(result.frames).toBeGreaterThan(1);
    expect(Math.abs(result.TL[2] - 255) <= 60, JSON.stringify(result)).toBe(true);
    expect(result.BR[0]).toBeGreaterThan(180);
    expect(result.BR[1]).toBeGreaterThan(180);
    expect(result.BR[2]).toBeGreaterThan(180);
});
