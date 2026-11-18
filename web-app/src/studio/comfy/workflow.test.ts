import { describe, expect, it } from 'vitest';
import {
    applyBindings,
    detectBindings,
    formatWorkflow,
    outputKind,
    parseWorkflowText,
    scalarInputs,
    workflowStats,
} from './workflow';

const WORKFLOW = {
    '3': { class_type: 'KSampler', inputs: { seed: 1, steps: 20, cfg: 4 } },
    '5': { class_type: 'EmptyLatentImage', inputs: { width: 768, height: 512 } },
    '6': { class_type: 'CLIPTextEncode', inputs: { text: 'old prompt' } },
    '7': { class_type: 'CLIPTextEncode', inputs: { text: 'old negative' } },
    '8': { class_type: 'CheckpointLoaderSimple', inputs: { ckpt_name: 'model.safetensors' } },
    '9': { class_type: 'SaveImage', inputs: { images: ['5', 0] } },
};

const SHOT = {
    id: 's1', prompt: 'new prompt', negativePrompt: 'bad', seed: 42,
    width: 512, height: 320, numFrames: 25, numInferenceSteps: 8,
    cfgScale: 3, pipelineOverride: 'auto', trackIndex: 0,
    trimIn: 0, trimOut: 0, timeline: [], status: 'pending',
} as any;

describe('comfy workflow helpers', () => {
    it('parses API JSON and rejects UI JSON', () => {
        expect(parseWorkflowText(JSON.stringify(WORKFLOW))).toEqual(WORKFLOW);
        expect(() => parseWorkflowText('{"nodes":[],"links":[]}')).toThrow('Save (API Format)');
        expect(() => parseWorkflowText('{}')).not.toThrow();
    });

    it('detects and applies shot bindings', () => {
        const bindings = detectBindings(WORKFLOW);
        expect(bindings.prompt).toEqual({ node: '6', input: 'text' });
        expect(bindings.negative).toEqual({ node: '7', input: 'text' });
        expect(bindings.seed).toEqual({ node: '3', input: 'seed' });
        expect(bindings.model?.value).toBe('model.safetensors');

        const resolved = applyBindings(WORKFLOW, bindings, SHOT);
        expect(resolved['6'].inputs.text).toBe('new prompt');
        expect(resolved['3'].inputs.seed).toBe(42);
        expect(resolved['3'].inputs.steps).toBe(8);
        expect(resolved['5'].inputs.width).toBe(512);
        expect(WORKFLOW['6'].inputs.text).toBe('old prompt');
    });

    it('lists scalar inputs and workflow stats', () => {
        const paths = scalarInputs(WORKFLOW).map((row) => `${row.node}.${row.input}`);
        expect(paths).toContain('3.seed');
        expect(paths).not.toContain('9.images');
        expect(workflowStats(WORKFLOW)).toEqual({ nodes: 6, outputs: 1 });
        expect(formatWorkflow(WORKFLOW)).toContain('"class_type"');
    });

    it('classifies outputs', () => {
        expect(outputKind({ kind: 'video' })).toBe('video');
        expect(outputKind({ source_filename: 'x.png' })).toBe('image');
        expect(outputKind({ source_filename: 'x.wav' })).toBe('audio');
        expect(outputKind({ source_filename: 'x.json' })).toBe('text');
    });
});
