import { describe, expect, it } from 'vitest';
import { parsePipelineRecipe, parseRecipePack } from './recipe';

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

describe('parseRecipePack', () => {
    const base = { prompt: 'shot', frames: 49, resolution: '512x288' };

    it('reads a named pack and preserves order', () => {
        const pack = parseRecipePack(JSON.stringify({
            name: 'Noir',
            recipes: [base, { ...base, prompt: 'second' }],
        }));
        expect(pack.name).toBe('Noir');
        expect(pack.recipes.map((recipe) => recipe.prompt)).toEqual(['shot', 'second']);
    });

    it('accepts a bare array as an unnamed pack', () => {
        const pack = parseRecipePack(JSON.stringify([base]));
        expect(pack.name).toBe('Recipe pack');
        expect(pack.recipes).toHaveLength(1);
    });

    it('rejects empty packs and names the bad entry', () => {
        expect(() => parseRecipePack('[]')).toThrow('no recipes');
        expect(() => parseRecipePack(JSON.stringify({ recipes: [{ prompt: 'x' }] })))
            .toThrow('Recipe 1:');
    });
});
