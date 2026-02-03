# MPS Dtype Fix Notes

## Current Status
**Stage 1**: ✅ Working (40/40 denoising steps complete)  
**Stage 2**: ✅ **FIXED** - transformer now correctly uses float16 on MPS

## Problem (Resolved)
`RuntimeError: Destination NDArray and Accumulator NDArray cannot have different datatype in MPSNDArrayMatrixMultiplication`

This occurred when float16 tensors met bfloat16 model weights on MPS.

---

## Resolution (2026-02-02)

### Root Cause
The `model_ledger.py` transformer() method used `.to(self.device)` which didn't force dtype conversion. Weights were loaded as bfloat16 and remained bfloat16 after moving to MPS.

### Fix Applied
Updated all model methods in `model_ledger.py` to use:
```python
.to(device=self.device, dtype=self.dtype)
```

This ensures all model weights are converted to float16 on MPS.

---

## All Fixes Applied

### 1. Pipeline Dtype = float16 for MPS
**File:** `ti2vid_two_stages.py` (line 65-69)
```python
if device == "mps":
    self.dtype = torch.float16
else:
    self.dtype = torch.bfloat16
```

### 2. VAE Decoder = float32 for MPS
**File:** `model_ledger.py` (video_decoder method)
```python
vae_dtype = torch.float32 if torch.backends.mps.is_available() else self.dtype
return self.vae_decoder_builder.build(..., dtype=vae_dtype).to(..., dtype=vae_dtype).eval()
```

### 3. LoRA Weight Casting
**File:** `fuse_loras.py`
- LoRA weights cast to target dtype before matmul

### 4. Latent to float32 before VAE
**File:** `ti2vid_two_stages.py` (before VAE decode)
```python
if self.device == "mps":
    video_latent_for_decode = video_state.latent.to(torch.float32)
```

### 5. Model dtype enforcement (NEW FIX)
**File:** `model_ledger.py` - All model methods
```python
# Changed from:
.to(self.device)
# To:
.to(device=self.device, dtype=self.dtype)
```

---

## Additional Features Restored

Also restored missing features from original LTX-2 during this fix session:

1. **MultiModalGuider** - Advanced guidance with STG, modality isolation
2. **VideoConditionByReferenceLatent** - IC-LoRA reference video support
3. **multi_modal_guider_denoising_func** - Perturbation-based denoising
4. **reference_downscale_factor** - LoRA metadata reading for IC-LoRA

---

## Architecture

```
TI2VidTwoStagesPipeline
├── stage_1_model_ledger (dtype=float16 on MPS)
│   └── transformer() → Stage 1 model (40 steps) ✅
└── stage_2_model_ledger (dtype=float16 on MPS)
    └── transformer() → Stage 2 model (3 steps) ✅ FIXED
```

Both ledgers now correctly convert model weights to float16 on MPS.

