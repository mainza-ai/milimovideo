import { readdirSync, rmSync, statSync } from 'node:fs';

// Playwright may start webServer before globalSetup, so never delete the
// current run's DB. Each run uses a unique path; clean up old ones by age.
export default async function globalSetup() {
    const maxAgeMs = 60 * 60 * 1000;
    const cutoff = Date.now() - maxAgeMs;
    for (const name of readdirSync('/tmp')) {
        if (!/^milimo-e2e-(projects-)?\d+/.test(name)) continue;
        const path = `/tmp/${name}`;
        try {
            const stats = statSync(path);
            if (stats.mtimeMs < cutoff) {
                rmSync(path, { recursive: true, force: true });
            }
        } catch {
            // Another run may have removed it already.
        }
    }
}
