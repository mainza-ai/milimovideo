import type { components } from './schema';

/** Generated from `backend/export_openapi.py`; run `npm run gen:api`. */
export type ApiSchemas = components['schemas'];
export type ApiProjectState = ApiSchemas['ProjectState'];
export type ApiCaptionCue = ApiSchemas['CaptionCue'];
export type ApiPlanResponse = ApiSchemas['PlanResponse'];
