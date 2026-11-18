"""LTX-2.5 diffusers engine.

The LTX-2.5 weights are gated, so this adapter is deliberately isolated from the
vendored LTX-2 pipeline. It loads the official diffusers directory, enables CPU
offload/VAE tiling when asked, generates video plus the native audio track, and
lets the normal task encode/ingest the result.
"""

from __future__ import annotations

import gc
import logging
import os
from typing import Callable, Optional

import numpy as np
import torch

import config

logger = logging.getLogger("ltx25.engine")


class LTX25Engine:
    def __init__(self, model_dir: str, device: str):
        self.model_dir = model_dir
        self.device = device
        self.dtype = torch.float32 if device == "cpu" else torch.bfloat16 if device == "cuda" else torch.float16
        self.pipe = None

    def load(self) -> None:
        if self.pipe is not None:
            return
        from diffusers import LTX2Pipeline

        if not os.path.isdir(self.model_dir):
            raise RuntimeError(
                f"LTX-2.5 diffusers directory not found: {self.model_dir}. "
                "Download Lightricks/LTX-2.5-Diffusers in Setup after accepting the "
                "gated community license."
            )
        logger.info("Loading LTX-2.5 from %s on %s (%s)", self.model_dir, self.device, self.dtype)
        pipe = LTX2Pipeline.from_pretrained(self.model_dir, torch_dtype=self.dtype)
        if self.device == "cuda" and config.LTX25_OFFLOAD:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(self.device)
        if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()
        self.pipe = pipe
        logger.info("LTX-2.5 ready")

    @staticmethod
    def normalize(num_frames: int, width: int, height: int) -> tuple[int, int, int]:
        frames = max(9, int(num_frames or 121))
        frames = max(9, 8 * round((frames - 1) / 8) + 1)
        width = max(256, int(width or 768) // 32 * 32)
        height = max(256, int(height or 512) // 32 * 32)
        return frames, width, height

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        *,
        width: int = 768,
        height: int = 512,
        num_frames: int = 121,
        fps: float = 25.0,
        num_inference_steps: int = 40,
        guidance: float = 4.0,
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
            frame_rate=float(fps or 25.0),
            num_inference_steps=max(1, int(num_inference_steps)),
            guidance_scale=float(guidance),
            generator=generator,
            output_type="np",
            callback_on_step_end=on_step,
        )
        video = output.frames
        while isinstance(video, (list, tuple)) and len(video) == 1:
            video = video[0]
        if isinstance(video, torch.Tensor):
            video = video.detach().cpu().numpy()
        audio = getattr(output, "audio", None)
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().float().numpy()
        return np.asarray(video), audio, frames

    def export_frames(self, frames, output_path: str, fps: int) -> None:
        import av

        array = np.asarray(frames)
        if array.ndim == 5:
            array = array[0]
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

    @staticmethod
    def export_audio(audio, output_path: str, sample_rate: int = 24000) -> bool:
        if audio is None:
            return False
        try:
            import torchaudio

            array = np.asarray(audio)
            if array.ndim == 3:
                array = array[0]
            if array.ndim == 1:
                array = array[None, :]
            tensor = torch.from_numpy(array)
            torchaudio.save(output_path, tensor, sample_rate)
            return True
        except Exception as exc:
            logger.warning("LTX-2.5 audio export failed: %s", exc)
            return False

    def unload(self) -> None:
        self.pipe = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


def device_name() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
