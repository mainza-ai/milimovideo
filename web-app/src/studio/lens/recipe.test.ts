import { describe, expect, it } from 'vitest';
import { parsePipelineRecipe } from './recipe';

describe('parsePipelineRecipe', () => {
    it('reads a saved recipe and normalizes the resolution string', () => {
        const recipe = parsePipelineRecipe(JSON.stringify({
            prompt: 'A lighthouse in a storm',
            negative_prompt: 'blurry',
            model: 'ic_lora',
            seed: 42,
            frames: 121,
            resolution: '768x512',
            enhance: true,
            upscale: false,
        }));
        expect(recipe).toEqual({
            prompt: 'A lighthouse in a storm',
            negativePrompt: 'blurry',
            model: 'ic_lora',
            seed: 42,
            frames: 121,
            width: 768,
            height: 512,
            enhance: true,
            upscale: false,
        });
    });

    it('accepts width/height fields and the snake_case prompt keys', () => {
        const recipe = parsePipelineRecipe(JSON.stringify({
            prompt: 'x',
            num_frames: 81,
            width: 640,
            height: 360,
        }));
        expect(recipe.frames).toBe(81);
        expect(recipe.width).toBe(640);
        expect(recipe.height).toBe(360);
        expect(recipe.model).toBe('auto');
    });

    it('rejects malformed recipes with a readable message', () => {
        expect(() => parsePipelineRecipe('not json')).toThrow('not valid JSON');
        expect(() => parsePipelineRecipe('{}')).toThrow('needs a prompt');
        expect(() => parsePipelineRecipe(JSON.stringify({
            prompt: 'x', frames: 121, resolution: '768x512', model: 'nope',
        }))).toThrow('Unknown model');
        expect(() => parsePipelineRecipe(JSON.stringify({
            prompt: 'x', frames: 121, resolution: 'wide',
        }))).toThrow('resolution');
    });
});
