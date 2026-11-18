"""Quantized FLUX.2 Klein 9B via a GGUF transformer.

The vendored `flux2` package only loads a full-precision state dict. This path
uses the official diffusers `Flux2KleinPipeline` with a GGUF transformer, which is
the low-VRAM option requested in issue #2. It covers text-to-image; IP-Adapter and
RePaint remain on the full checkpoint.
"""

from __future__ import annotations

import gc
import logging
import os
from typing import Callable, Optional

import torch

logger = logging.getLogger("flux.gguf")


class FluxGGUF:
    def __init__(
        self,
        model_dir: str,
        gguf_path: str,
        device: str,
        *,
        offload: bool = False,
    ):
        from diffusers import (
            AutoencoderKLFlux2,
            FlowMatchEulerDiscreteScheduler,
            Flux2KleinPipeline,
            Flux2Transformer2DModel,
            GGUFQuantizationConfig,
        )
        from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

        self.device = device
        self.model_dir = model_dir
        self.gguf_path = gguf_path
        self.offload = bool(offload and device == "cuda")
        # MPS needs float32 for stable Flux/VAE math; CUDA uses half precision.
        self.dtype = torch.float32 if device == "mps" else torch.float16

        compute_dtype = torch.float32 if device == "mps" else torch.float16
        logger.info("Loading quantized Flux 2 GGUF: %s (%s)", gguf_path, compute_dtype)
        transformer = Flux2Transformer2DModel.from_single_file(
            gguf_path,
            quantization_config=GGUFQuantizationConfig(compute_dtype=compute_dtype),
            torch_dtype=compute_dtype,
            config=os.path.join(model_dir, "transformer"),
        )
        text_encoder = Qwen3ForCausalLM.from_pretrained(
            os.path.join(model_dir, "text_encoder"),
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
        )
        tokenizer = Qwen2TokenizerFast.from_pretrained(os.path.join(model_dir, "tokenizer"))
        vae = AutoencoderKLFlux2.from_pretrained(
            os.path.join(model_dir, "vae"),
            torch_dtype=torch.float32,
        )
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            os.path.join(model_dir, "scheduler"),
        )

        pipe = Flux2KleinPipeline(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            is_distilled=True,
        )
        if self.offload:
            # CUDA low-VRAM path: keep only the active component on the GPU.
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)

        # The pipeline latents follow the transformer/text dtype, while the FLUX
        # VAE is float32 for stability. Cast at the boundary.
        if self.dtype != torch.float32:
            original_decode = pipe.vae.decode

            def decode_fp32(latents, *args, **kwargs):
                return original_decode(latents.to(torch.float32), *args, **kwargs)

            pipe.vae.decode = decode_fp32

        self.pipe = pipe
        logger.info("Quantized Flux 2 ready (%s%s)", device, ", CPU offload" if self.offload else "")

    def generate(
        self,
        prompt: str,
        width: int,
        height: int,
        guidance: float,
        num_inference_steps: int,
        seed: Optional[int] = None,
        callback: Optional[Callable[[int, int], None]] = None,
    ):
        generator_device = "cpu" if self.offload else self.device
        generator = torch.Generator(generator_device).manual_seed(
            int(seed if seed is not None else torch.seed())
        )

        def on_step(_pipe, step, _timestep, kwargs):
            if callback:
                callback(step + 1, num_inference_steps)
            return kwargs

        output = self.pipe(
            prompt=prompt,
            width=int(width),
            height=int(height),
            num_inference_steps=max(1, int(num_inference_steps)),
            guidance_scale=float(guidance),
            generator=generator,
            callback_on_step_end=on_step,
        )
        return output.images[0]

    def unload(self) -> None:
        self.pipe = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
