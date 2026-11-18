import { apiFetch } from '../../config';

export interface UploadedLut {
    url: string;
    type: string;
    filename: string;
    asset_id: string;
}

/** Upload a 3D LUT into the project's asset folder. */
export async function uploadProjectLut(projectId: string, file: File): Promise<UploadedLut> {
    const form = new FormData();
    form.append('file', file);
    form.append('project_id', projectId);
    form.append('type', 'lut');
    const response = await apiFetch('/upload', { method: 'POST', body: form });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
    if (!data.url) throw new Error('Upload did not return a LUT URL');
    return data as UploadedLut;
}
