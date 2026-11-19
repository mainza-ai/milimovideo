import { readFileSync, readdirSync, statSync } from 'node:fs';
import { resolve } from 'node:path';
import { gzipSync } from 'node:zlib';

const root = resolve(import.meta.dirname, '..');
const assetsDir = resolve(root, 'dist/assets');
const manifestPath = resolve(root, 'dist/.vite/manifest.json');

const totalBudget = 2_000_000;
const chunkBudget = 800_000;
// Initial JS for either route: shared entry + that route's shell. The studio
// computes this as entry + StudioApp; the legacy route is entry + LegacyEditor.
const initialGzipBudget = 250 * 1024;

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

function gzipBytes(file) {
    return gzipSync(readFileSync(resolve(root, 'dist', file))).length;
}

function closure(manifest, key, seen = new Set()) {
    const entry = manifest[key];
    if (!entry || seen.has(key)) return seen;
    seen.add(key);
    for (const imported of entry.imports || []) closure(manifest, imported, seen);
    return seen;
}

function shellGzip(manifest, keys) {
    const chunks = new Set();
    for (const key of keys) for (const file of closure(manifest, key)) chunks.add(file);
    let bytes = 0;
    for (const key of chunks) {
        const file = manifest[key].file;
        bytes += gzipBytes(file);
    }
    return bytes;
}

try {
    const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
    const entryKey = Object.keys(manifest).find((key) => manifest[key].isEntry);
    // StudioApp is a shared chunk (App.tsx imports it dynamically), so the
    // manifest key is generated rather than the source path.
    const studioKey = Object.keys(manifest).find(
        (key) => key.includes('StudioApp') && manifest[key].file.endsWith('.js'),
    );
    const legacyKey = Object.keys(manifest).find((key) => key.endsWith('src/LegacyEditor.tsx'));
    const storyKey = Object.keys(manifest).find((key) => key.endsWith('src/studio/surfaces/StorySurface.tsx'));
    if (entryKey) {
        const shared = shellGzip(manifest, [entryKey]);
        const studio = studioKey ? shellGzip(manifest, [entryKey, studioKey]) : null;
        const story = storyKey ? shellGzip(manifest, [entryKey, studioKey, storyKey]) : null;
        const legacy = legacyKey ? shellGzip(manifest, [entryKey, legacyKey]) : null;
        console.log(`Entry gzip: ${(shared / 1024).toFixed(1)} KiB`);
        if (studio !== null) console.log(`Studio initial gzip: ${(studio / 1024).toFixed(1)} KiB`);
        if (story !== null) console.log(`Studio + Story gzip: ${(story / 1024).toFixed(1)} KiB`);
        if (legacy !== null) console.log(`Legacy initial gzip: ${(legacy / 1024).toFixed(1)} KiB`);
        for (const [label, value] of [['studio', studio], ['legacy', legacy]]) {
            if (value !== null && value > initialGzipBudget) {
                console.error(`Initial gzip budget exceeded (${label}): ${(value / 1024).toFixed(1)} KiB > 250 KiB`);
                failed = true;
            }
        }
    }
} catch (error) {
    console.error(`Could not read ${manifestPath}: ${error.message}`);
    failed = true;
}

process.exit(failed ? 1 : 0);
