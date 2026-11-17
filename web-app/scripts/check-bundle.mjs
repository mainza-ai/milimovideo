import { readdirSync, statSync } from 'node:fs';
import { resolve } from 'node:path';

const assetsDir = resolve(import.meta.dirname, '../dist/assets');
const totalBudget = 2_000_000;
const chunkBudget = 800_000;

let files;
try {
    files = readdirSync(assetsDir).filter((name) => name.endsWith('.js'));
} catch {
    console.error('dist/assets is missing. Run `npm run build` first.');
    process.exit(1);
}

const sizes = files
    .map((name) => ({ name, bytes: statSync(resolve(assetsDir, name)).size }))
    .sort((a, b) => b.bytes - a.bytes);

if (sizes.length === 0) {
    console.error('No JavaScript bundles found.');
    process.exit(1);
}

const total = sizes.reduce((sum, file) => sum + file.bytes, 0);
const largest = sizes[0];
console.log(`JS bundles: ${sizes.length} · total ${(total / 1024).toFixed(0)} KiB · largest ${(largest.bytes / 1024).toFixed(0)} KiB (${largest.name})`);

let failed = false;
if (total > totalBudget) {
    console.error(`Bundle budget exceeded: ${total} > ${totalBudget} bytes`);
    failed = true;
}
if (largest.bytes > chunkBudget) {
    console.error(`Chunk budget exceeded: ${largest.name} is ${largest.bytes} > ${chunkBudget} bytes`);
    failed = true;
}
process.exit(failed ? 1 : 0);
