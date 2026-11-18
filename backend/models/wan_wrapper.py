"""Wan 2.2 TI2V-5B video engine (Apache 2.0).

The LTX path is deeply tied to LTX latents, Gemma, chunking, and audio. Wan is a
separate engine: a 5B dense DiT with a high-compression VAE and UMT5 encoder that
fits consumer CUDA cards. This adapter is intentionally narrow — text-to-video
with a prompt and an optional first-frame image is the first supported surface.
"""

from __future__ import annotations

import gc
import logging
import os
from typing import Callable, Optional

import torch

logger = logging.getLogger("wan.engine")


class WanEngine:
    def __init__(self, model_dir: str, device: str):
        self.model_dir = model_dir
        self.device = device
        # Wan's VAE is stable in half precision on MPS in practice; CPU is fp32.
        self.dtype = torch.float32 if device == "cpu" else torch.float16
        self.pipe = None

    def load(self) -> None:
        if self.pipe is not None:
            return
        from diffusers import WanPipeline

        if not os.path.isdir(self.model_dir):
            raise RuntimeError(
                f"Wan 2.2 model directory not found: {self.model_dir}. "
                "Download Wan-AI/Wan2.2-TI2V-5B-Diffusers in Setup first."
            )
        logger.info("Loading Wan 2.2 TI2V-5B from %s on %s (%s)", self.model_dir, self.device, self.dtype)
        pipe = WanPipeline.from_pretrained(self.model_dir, torch_dtype=self.dtype)
        pipe.to(self.device)
        self.pipe = pipe
        logger.info("Wan 2.2 ready")

    @staticmethod
    def normalize(num_frames: int, width: int, height: int) -> tuple[int, int, int]:
        frames = max(9, int(num_frames or 49))
        # Wan's temporal VAE expects 4n+1; pick the nearest valid count.
        frames = max(9, 4 * round((frames - 1) / 4) + 1)
        width = max(256, int(width or 480) // 16 * 16)
        height = max(256, int(height or 480) // 16 * 16)
        return frames, width, height

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        *,
        width: int = 480,
        height: int = 480,
        num_frames: int = 49,
        num_inference_steps: int = 20,
        guidance: float = 5.0,
        seed: Optional[int] = None,
        callback: Optional[Callable[[int, int], None]] = None,
    ):
        if self.pipe is None:
            self.load()
        frames, width, height = self.normalize(num_frames, width, height)
        generator = torch.Generator(self.device).manual_seed(
            int(seed if seed is not None else torch.seed())
        )

        def on_step(_pipe, step, _timestep, kwargs):
            if callback:
                callback(step + 1, num_inference_steps)
            return kwargs

        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or "",
            height=height,
            width=width,
            num_frames=frames,
            num_inference_steps=max(1, int(num_inference_steps)),
            guidance_scale=float(guidance),
            generator=generator,
            output_type="np",
            callback_on_step_end=on_step,
        )
        return output.frames[0], frames

    def export_frames(self, frames, output_path: str, fps: int) -> None:
        """Encode a (T,H,W,C) float/uint8 array with the repo's existing av stack."""
        import av
        import numpy as np

        array = np.asarray(frames)
        if array.dtype != np.uint8:
            array = (np.clip(array, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        height, width = array.shape[1], array.shape[2]
        container = av.open(output_path, mode="w", options={"movflags": "+faststart"})
        stream = container.add_stream("libx264", rate=int(fps))
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "medium"}
        for frame in array:
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()

    def unload(self) -> None:
        self.pipe = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
