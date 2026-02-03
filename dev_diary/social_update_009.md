# Dev Log: Legacy Path Removal & Prompt Fix

**Date**: 2026-02-02
**Topic**: Removing Legacy Paths & Fixing Image Conditioning

## Summary

Cleanup session tackling two interconnected bugs: broken image conditioning (images not passed to Gemma for prompt enhancement) and legacy path references causing missing files.

## Root Causes

**1. Truncated Enhanced Prompts**
Gemma outputs were fragments like "and then smiles at the camera." Causes:
- `clean_response()` stripped "and " prefix
- I2V system prompt too brief

**2. Image Path Resolution**
Browser state stored legacy absolute paths (`/backend/generated/images/...`), but files now live at `/projects/{id}/generated/images/...`.

**3. Legacy Fallbacks**
Both `worker.py` and `server.py` had `else` branches producing `/generated/...` URLs without project prefix.

## Fixes

**Prompt Enhancement:**
- Removed "and " stripping from `helpers.py`
- Rewrote I2V system prompt with image analysis guidelines
- Result: Complete cinematic descriptions with audio layers

**Path Resolution:**
- Added fallback to search project directories for legacy paths
- `get_project_output_paths()` now requires `project_id`
- `generate_image_task()` requires `project_id`
- Removed `GENERATED_DIR` from server imports

## Verification

```
✓ Fallback path found and added
Enhanced prompt: The woman with freckled skin... she slowly 
begins to speak, clearly articulating "Hi, welcome to Milimo Quantum"...
```

## Files Modified

| File | Change |
|------|--------|
| `helpers.py` | Removed prefix stripping |
| `worker.py` | I2V prompt, path fallback |
| `server.py` | Removed legacy endpoints |

---

#MilimoVideo #PathRefactor #ImageConditioning #PromptEnhancement #AppleSilicon
