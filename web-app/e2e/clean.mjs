import { rmSync } from 'node:fs';

// Runs before Playwright starts its webServers, so removing the DB cannot race a
// live SQLite connection (which would recreate an empty file without tables).
rmSync('/tmp/milimo-e2e.db', { force: true });
rmSync('/tmp/milimo-e2e-projects', { recursive: true, force: true });
