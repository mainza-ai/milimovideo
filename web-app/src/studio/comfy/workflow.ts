import type { Shot } from '../state/documentStore';

export type ComfyWorkflow = Record<string, any>;

export interface ComfyBinding {
    node: string;
    input: string;
    value?: string | number | boolean;
}

export type ComfyBindingKey =
    | 'prompt'
    | 'negative'
    | 'seed'
    | 'width'
    | 'height'
    | 'steps'
    | 'cfg'
    | 'model';

export type ComfyBindings = Partial<Record<ComfyBindingKey, ComfyBinding>>;

export const BINDING_FIELDS: Array<{ key: ComfyBindingKey; label: string }> = [
    { key: 'prompt', label: 'Prompt' },
    { key: 'negative', label: 'Negative' },
    { key: 'seed', label: 'Seed' },
    { key: 'width', label: 'Width' },
    { key: 'height', label: 'Height' },
    { key: 'steps', label: 'Steps' },
    { key: 'cfg', label: 'CFG' },
    { key: 'model', label: 'Model' },
];

export interface ScalarInput {
    node: string;
    input: string;
    classType: string;
    value: string | number | boolean;
}

export function parseWorkflowText(text: string): ComfyWorkflow {
    let data: unknown;
    try {
        data = JSON.parse(text);
    } catch (error) {
        throw new Error(error instanceof Error ? `Invalid JSON: ${error.message}` : 'Invalid JSON');
    }
    if (!data || typeof data !== 'object' || Array.isArray(data)) {
        throw new Error('Workflow must be a JSON object in ComfyUI API format');
    }
    const workflow = data as ComfyWorkflow;
    if ('nodes' in workflow && 'links' in workflow) {
        throw new Error('Use Workflow → Save (API Format), not the UI-format JSON');
    }
    for (const [nodeId, node] of Object.entries(workflow)) {
        if (!node || typeof node !== 'object' || !('class_type' in node)) {
            throw new Error(`Node ${nodeId} is missing class_type`);
        }
    }
    return workflow;
}

export function formatWorkflow(workflow: ComfyWorkflow): string {
    return JSON.stringify(workflow, null, 2);
}

export function scalarInputs(workflow: ComfyWorkflow): ScalarInput[] {
    const rows: ScalarInput[] = [];
    for (const [nodeId, node] of Object.entries(workflow)) {
        const inputs = (node as any)?.inputs;
        if (!inputs || typeof inputs !== 'object') continue;
        for (const [input, value] of Object.entries(inputs)) {
            if (Array.isArray(value)) continue; // linked input
            if (['string', 'number', 'boolean'].includes(typeof value)) {
                rows.push({
                    node: nodeId,
                    input,
                    classType: String((node as any).class_type || ''),
                    value: value as string | number | boolean,
                });
            }
        }
    }
    return rows;
}

function firstInput(workflow: ComfyWorkflow, names: string[], classTypes?: string[]) {
    for (const [nodeId, node] of Object.entries(workflow)) {
        const classType = String((node as any)?.class_type || '');
        if (classTypes && !classTypes.includes(classType)) continue;
        const inputs = (node as any)?.inputs || {};
        for (const name of names) {
            if (typeof inputs[name] !== 'undefined' && !Array.isArray(inputs[name])) {
                return { node: nodeId, input: name, value: inputs[name] };
            }
        }
    }
    return null;
}

export function detectBindings(workflow: ComfyWorkflow): ComfyBindings {
    const textNodes = Object.entries(workflow)
        .filter(([, node]) => String((node as any)?.class_type || '') === 'CLIPTextEncode')
        .map(([nodeId, node]) => ({ node: nodeId, value: (node as any)?.inputs?.text }));
    const bindings: ComfyBindings = {};
    if (textNodes[0]) bindings.prompt = { node: textNodes[0].node, input: 'text' };
    if (textNodes[1]) bindings.negative = { node: textNodes[1].node, input: 'text' };

    const seed = firstInput(workflow, ['seed', 'noise_seed']);
    if (seed) bindings.seed = { node: seed.node, input: seed.input };
    const width = firstInput(workflow, ['width']);
    if (width) bindings.width = { node: width.node, input: width.input };
    const height = firstInput(workflow, ['height']);
    if (height) bindings.height = { node: height.node, input: height.input };
    const steps = firstInput(workflow, ['steps']);
    if (steps) bindings.steps = { node: steps.node, input: steps.input };
    const cfg = firstInput(workflow, ['cfg', 'guidance', 'guidance_scale']);
    if (cfg) bindings.cfg = { node: cfg.node, input: cfg.input };
    const model = firstInput(workflow, ['ckpt_name', 'unet_name', 'model_name']);
    if (model) {
        bindings.model = { node: model.node, input: model.input, value: model.value as string };
    }
    return bindings;
}

export function applyBindings(
    workflow: ComfyWorkflow,
    bindings: ComfyBindings,
    shot: Shot,
): ComfyWorkflow {
    const next = JSON.parse(JSON.stringify(workflow)) as ComfyWorkflow;
    const values: Record<ComfyBindingKey, string | number | boolean | undefined> = {
        prompt: shot.prompt,
        negative: shot.negativePrompt,
        seed: shot.seed,
        width: shot.width,
        height: shot.height,
        steps: shot.numInferenceSteps ?? 40,
        cfg: shot.cfgScale,
        model: bindings.model?.value,
    };
    for (const [key, binding] of Object.entries(bindings) as Array<[ComfyBindingKey, ComfyBinding]>) {
        if (!binding?.node || !binding.input) continue;
        const node = next[binding.node];
        if (!node) continue;
        const value = binding.value !== undefined ? binding.value : values[key];
        if (value === undefined || value === null) continue;
        node.inputs = node.inputs || {};
        node.inputs[binding.input] = value;
    }
    return next;
}

export function workflowStats(workflow: ComfyWorkflow) {
    const nodes = Object.entries(workflow);
    const outputs = nodes.filter(([, node]) => /^(Save|Preview)/.test(String((node as any)?.class_type || '')));
    return { nodes: nodes.length, outputs: outputs.length };
}

export function outputKind(record: { kind?: string; url?: string; source_filename?: string }): 'image' | 'video' | 'audio' | 'text' {
    if (record.kind === 'image' || record.kind === 'video' || record.kind === 'audio') return record.kind;
    const name = `${record.url || ''} ${record.source_filename || ''}`.toLowerCase();
    if (/\.(png|jpe?g|webp|gif|avif)$/.test(name)) return 'image';
    if (/\.(mp4|mov|m4v|webm|mkv|avi)$/.test(name)) return 'video';
    if (/\.(mp3|wav|m4a|aac|ogg|flac)$/.test(name)) return 'audio';
    return 'text';
}
