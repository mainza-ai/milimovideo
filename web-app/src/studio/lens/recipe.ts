export interface PipelineRecipe {
    prompt: string;
    negativePrompt: string;
    model: 'auto' | 'ti2vid' | 'ic_lora' | 'keyframe';
    seed: number;
    frames: number;
    width: number;
    height: number;
    enhance: boolean;
    upscale: boolean;
}

export const PIPELINE_MODELS: PipelineRecipe['model'][] = ['auto', 'ti2vid', 'ic_lora', 'keyframe'];

function readNumber(value: unknown, field: string): number {
    const number = typeof value === 'number' ? value : Number(value);
    if (!Number.isFinite(number)) throw new Error(`${field} must be a number`);
    return Math.round(number);
}

function readResolution(value: unknown): { width: number; height: number } {
    if (typeof value === 'string') {
        const match = value.match(/^(\d+)\s*[x×]\s*(\d+)$/i);
        if (match) return { width: Number(match[1]), height: Number(match[2]) };
    }
    if (value && typeof value === 'object') {
        const record = value as Record<string, unknown>;
        return {
            width: readNumber(record.width, 'resolution.width'),
            height: readNumber(record.height, 'resolution.height'),
        };
    }
    throw new Error('resolution must be "WxH" or {width,height}');
}

export function parsePipelineRecipe(raw: string): PipelineRecipe {
    let data: unknown;
    try {
        data = JSON.parse(raw);
    } catch {
        throw new Error('Recipe is not valid JSON');
    }
    if (!data || typeof data !== 'object') throw new Error('Recipe must be an object');
    const record = data as Record<string, unknown>;

    const prompt = typeof record.prompt === 'string' ? record.prompt : '';
    if (!prompt.trim()) throw new Error('Recipe needs a prompt');

    const model = typeof record.model === 'string' ? record.model : 'auto';
    if (!PIPELINE_MODELS.includes(model as PipelineRecipe['model'])) {
        throw new Error(`Unknown model "${model}"`);
    }

    const frames = readNumber(record.frames ?? record.num_frames, 'frames');
    if (frames < 1) throw new Error('frames must be positive');

    const { width, height } = readResolution(record.resolution ?? {
        width: record.width,
        height: record.height,
    });

    return {
        prompt,
        negativePrompt: typeof record.negative_prompt === 'string'
            ? record.negative_prompt
            : typeof record.negativePrompt === 'string'
                ? record.negativePrompt
                : '',
        model: model as PipelineRecipe['model'],
        seed: readNumber(record.seed ?? 0, 'seed'),
        frames,
        width,
        height,
        enhance: Boolean(record.enhance),
        upscale: Boolean(record.upscale),
    };
}
