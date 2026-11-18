# LTX-2.5

**Status:** pipeline adapter implemented; gated weights not bundled.

LTX-2.5 is Lightricks' 22B successor to LTX-2, with native synchronized audio,
prompt/keyframe camera control, and an official diffusers path. It is the primary
video upgrade for 32 GB+ machines. The weights are gated under the LTX-2.x
Community license, so Milimo implements the pipeline but does not ship the model.

## Integration

| Piece | Behavior |
|---|---|
| `models/ltx25_wrapper.py` | `LTX2Pipeline.from_pretrained` from `MILIMO_LTX25_MODEL`; CUDA model offload, VAE tiling, native audio capture |
| `tasks/ltx25.py` | Queued video job with admission, progress, MP4 encode, audio mux, poster, asset/shot ingestion |
| `pipeline_override=ltx25` | Explicit per-shot selection in Generate → Advanced → Pipeline |
| Manifest | `ltx25-diffusers` (gated, recommended) |
| Memory | 64 GB Apple / 40 GB CUDA base, plus frames/resolution workspace |

The adapter runs the prompt-enhancement pass before the model loads and uses the
normal `job.*` event plane. Frame counts are `8k+1` and both dimensions are
multiples of 32. The native audio track is muxed into the MP4 when the pipeline
returns one.

## Install

1. Accept the community license on
   [Lightricks/LTX-2.5-Diffusers](https://huggingface.co/Lightricks/LTX-2.5-Diffusers).
2. Authenticate the local Hugging Face client.
3. Download `LTX-2.5-Diffusers` in **Setup → Models**, or set:

```bash
export MILIMO_LTX25_MODEL=/path/to/ltx-2.5-diffusers
```

4. Select **LTX-2.5** in Generate → Advanced → Pipeline.

CUDA can use `MILIMO_LTX25_OFFLOAD=0` when the full model fits; offload is on by
default.

## Verification

- `tests/test_ltx25_engine.py` covers normalization, the diffusers call contract,
  manifest gating/recommendation, the memory estimate, and dispatch through
  `generate_video_task`.
- Real generation is not claimed here: the gated weights are not present in the
  repository, so the pipeline is verified with a fake pipe and must be exercised on
  a licensed machine with the downloaded directory.

## Limits

- Gated weights and license acceptance are required.
- Native audio sample rate/layout depends on the released checkpoint; the adapter
  defaults to 24 kHz and a single channel layout, and the audio mux is skipped if
  the pipeline returns no audio.
- No chained multi-chunk path for LTX-2.5 yet; it runs as a single generation.
- CUDA 32 GB validation and Apple Silicon bf16/fp16 behavior still need a licensed
  model on the target machine.
