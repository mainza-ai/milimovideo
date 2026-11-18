# LTX-2.5

**Status:** pipeline adapter implemented; gated weights downloaded locally and a
tiny MPS generation verified (video + native audio). Weights are not committed.

LTX-2.5 is Lightricks' 22B successor to LTX-2, with native synchronized audio,
prompt/keyframe camera control, and an official diffusers path. It is the primary
video upgrade for 32 GB+ machines. The weights are gated under the LTX-2.x
Community license, so Milimo implements the pipeline but does not ship the model.

## Integration

| Piece | Behavior |
|---|---|
| `models/ltx25_wrapper.py` | `LTX2Pipeline.from_pretrained` from `MILIMO_LTX25_MODEL`; CUDA model offload, VAE tiling, MPS float64 fallback, native audio capture |
| `tasks/ltx25.py` | Queued video job with admission, progress, MP4 encode, audio mux, poster, asset/shot ingestion |
| `pipeline_override=ltx25` | Explicit per-shot selection in Generate → Advanced → Pipeline |
| Manifest | `ltx25-diffusers` (gated, recommended) |
| Memory | 64 GB Apple / 40 GB CUDA base, plus frames/resolution workspace; local run peaked at ~52 GB |
| Dependencies | `diffusers==0.40.0`, `transformers==5.23.0`, `huggingface-hub==1.33.0` |

The adapter runs the prompt-enhancement pass before the model loads and uses the
normal `job.*` event plane. Frame counts are `8k+1` and both dimensions are
multiples of 32. The native audio track is written at 48 kHz and muxed into the
MP4 when the pipeline returns one. A cold load takes about 57 seconds on the
128 GB reference machine.

## Install

1. Accept the community license on
   [Lightricks/LTX-2.5-Diffusers](https://huggingface.co/Lightricks/LTX-2.5-Diffusers).
2. Authenticate the local Hugging Face client.
3. Download `LTX-2.5-Diffusers` in **Setup → Models**, or set:

```bash
export MILIMO_LTX25_MODEL=/path/to/ltx-2.5-diffusers
```

4. Select **LTX-2.5** in Generate → Advanced → Pipeline. See the
   [User Guide](../../docs/user_guide.md) for the Studio path.

CUDA can use `MILIMO_LTX25_OFFLOAD=0` when the full model fits; offload is on by
default.

## Verification

- `tests/test_ltx25_engine.py` covers normalization, the diffusers call contract,
  manifest gating/recommendation, the memory estimate, and dispatch through
  `generate_video_task`.
- Real generation was verified locally on MPS: 9 frames at 256², 1 step, video
  plus 48 kHz native audio, with shot/poster ingestion. The gated weights are not
  committed; CUDA 32 GB validation still needs the target machine.

## Limits

- Gated weights and license acceptance are required.
- Native audio sample rate/layout depends on the released checkpoint; the adapter
  defaults to 48 kHz and a single channel layout, and the audio mux is skipped if
  the pipeline returns no audio.
- No chained multi-chunk path for LTX-2.5 yet; it runs as a single generation.
- CUDA 32 GB validation still needs the target machine. The Apple Silicon run used
  float16 with the LTX-2 RoPE double-precision fallback disabled for MPS.
