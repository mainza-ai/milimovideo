import { expect, test } from '@playwright/test';
import { stubStudioApi } from './stubs';

function json(data: unknown) {
    return { status: 200, contentType: 'application/json', body: JSON.stringify(data) };
}

function projectState() {
    return {
        id: 'project-test',
        name: 'Stale Job Test',
        fps: 25,
        seed: 42,
        resolution_w: 1024,
        resolution_h: 576,
        revision: 1,
        scenes: [],
        captions: [],
        titles: [],
        shots: [{
            id: 'stale-shot',
            project_id: 'project-test',
            index: 0,
            sequence_index: 0,
            prompt: 'running shot',
            seed: 1,
            width: 1024,
            height: 576,
            num_frames: 25,
            fps: 25,
            track_index: 0,
            start_frame: 0,
            trim_in: 0,
            trim_out: 0,
            timeline: [],
            status: 'generating',
            last_job_id: 'stale-job',
        }],
    };
}

test('a vanished active job leaves Running and the shell owns one poller', async ({ page }) => {
    await stubStudioApi(page);
    let activeCalls = 0;
    await page.route('**/projects/project-test', (route) => route.fulfill(json(projectState())));
    await page.route('**/projects/project-test/active_jobs', (route) => {
        activeCalls += 1;
        // The first snapshot says running; after that the job has vanished.
        return route.fulfill(json(activeCalls <= 2 ? [{
            job_id: 'stale-job',
            type: 'video',
            kind: 'video',
            shot_id: 'stale-shot',
            status: 'running',
            stage: 'denoising',
            progress: 42,
            status_message: 'Working…',
        }] : []));
    });
    await page.route('**/status/stale-job', (route) => route.fulfill(json({
        status: 'completed',
        video_url: '/projects/project-test/generated/stale.mp4',
        thumbnail_url: '/projects/project-test/thumbnails/stale.jpg',
    })));

    let polls = 0;
    page.on('request', (request) => {
        if (request.url().includes('/active_jobs')) polls += 1;
    });

    await page.goto('/studio');
    const card = page.locator('.st-shot-card-status').first();
    await expect(card).toContainText('Running', { timeout: 8_000 });
    await expect(card).toContainText('Done', { timeout: 8_000 });
    expect(polls).toBeLessThanOrEqual(5);
});
