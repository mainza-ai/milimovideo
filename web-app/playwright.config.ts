import { existsSync } from 'node:fs';
import { defineConfig, devices } from '@playwright/test';

// Local installs have a venv; CI installs into the active interpreter.
const python = process.env.MILIMO_E2E_PYTHON
    || (existsSync('./milimov/bin/python') ? './milimov/bin/python' : 'python');
const runId = Date.now();
const e2eDatabase = `/tmp/milimo-e2e-${runId}.db`;
const e2eProjects = `/tmp/milimo-e2e-projects-${runId}`;

export default defineConfig({
    testDir: './e2e',
    timeout: 45_000,
    fullyParallel: false,
    workers: 1,
    globalSetup: './e2e/global-setup.ts',
    reporter: [['list']],
    use: {
        baseURL: 'http://127.0.0.1:5173',
        trace: 'retain-on-failure',
        launchOptions: {
            args: ['--disable-gpu', '--no-sandbox'],
        },
    },
    projects: [
        { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
    ],
    webServer: [
        {
            command: [
                'cd .. &&',
                'MILIMO_API_PORT=8010',
                `MILIMO_DATABASE_URL=sqlite:///${e2eDatabase}`,
                `MILIMO_PROJECTS_DIR=${e2eProjects}`,
                `${python} backend/server.py`,
            ].join(' '),
            url: 'http://127.0.0.1:8010/health',
            reuseExistingServer: false,
            timeout: 120_000,
            stdout: 'ignore',
            stderr: 'pipe',
        },
        {
            command: 'npm run dev -- --host 127.0.0.1',
            url: 'http://127.0.0.1:5173/studio',
            reuseExistingServer: true,
            timeout: 90_000,
            env: {
                ...process.env,
                VITE_API_URL: 'http://127.0.0.1:8010',
            },
        },
    ],
});
