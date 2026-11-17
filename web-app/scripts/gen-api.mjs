import { execFileSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { resolve } from 'node:path';

const root = resolve(import.meta.dirname, '../..');
const python = process.env.MILIMO_PYTHON
    || (existsSync(resolve(root, 'milimov/bin/python')) ? resolve(root, 'milimov/bin/python') : 'python');

const schema = execFileSync(python, ['backend/export_openapi.py'], {
    cwd: root,
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'ignore'],
});
const jsonPath = resolve(root, 'web-app/openapi.json');
const { writeFileSync } = await import('node:fs');
writeFileSync(jsonPath, schema);

execFileSync('npx', ['openapi-typescript', 'openapi.json', '-o', 'src/api/schema.d.ts'], {
    cwd: resolve(root, 'web-app'),
    stdio: 'inherit',
});
