import { describe, expect, it } from 'vitest';
import { API_BASE_URL, getAssetUrl } from './config';

describe('asset URL resolution', () => {
    it('rewrites legacy localhost media onto the current API origin', () => {
        expect(getAssetUrl('http://localhost:8000/projects/p/generated/a.jpg'))
            .toBe(`${API_BASE_URL}/projects/p/generated/a.jpg`);
    });

    it('leaves external media URLs untouched', () => {
        expect(getAssetUrl('https://cdn.example.com/a.jpg')).toBe('https://cdn.example.com/a.jpg');
    });

    it('resolves project-relative media', () => {
        expect(getAssetUrl('/projects/p/thumb.jpg')).toBe(`${API_BASE_URL}/projects/p/thumb.jpg`);
    });
});
