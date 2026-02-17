import asyncio
import os
import torch
import gc
import logging
import config

# Setup paths to ensure LTX modules are importable
config.setup_paths()

from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
from ltx_pipelines.ic_lora import ICLoraPipeline
from ltx_pipelines.keyframe_interpolation import KeyframeInterpolationPipeline
from ltx_core.loader import LoraPathStrengthAndSDOps

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self):
        self._pipeline = None
        self._pipeline_type = None
        self._lock = asyncio.Lock()
        
    def get_base_dir(self):
        return config.BACKEND_DIR

    def get_model_paths(self):
        # Use centralized config
        # LTX_DIR is now in config, but we used config.LTX_DIR in worker.py?
        # worker.py used: ltx2_root = config.LTX_DIR
        ltx2_root = config.LTX_DIR
        models_dir = os.path.join(ltx2_root, "models")
        
        # Auto-select best available checkpoint
        # Priority 1: Full Precision / BF16 (Better for MPS/CPU stability if available)
        ckpt_full = os.path.join(models_dir, "checkpoints", "ltx-2-19b-distilled.safetensors")
        # Priority 2: FP8 (Smaller, but requires cast on MPS)
        ckpt_fp8 = os.path.join(models_dir, "checkpoints", "ltx-2-19b-distilled-fp8.safetensors")
        
        selected_ckpt = ckpt_full
        if os.path.exists(ckpt_full):
            logger.info(f"Selected Main Checkpoint: {ckpt_full}")
        elif os.path.exists(ckpt_fp8):
            selected_ckpt = ckpt_fp8
            logger.info(f"Selected FP8 Checkpoint: {ckpt_fp8}")
        else:
            logger.warning("No checkpoint found! Defaulting to full path logic.")

        return {
            "checkpoint_path": selected_ckpt,
            "distilled_lora_path": os.path.join(models_dir, "checkpoints", "ltx-2-19b-distilled-lora-384.safetensors"), 
            "spatial_upsampler_path": os.path.join(models_dir, "upscalers", "ltx-2-spatial-upscaler-x2-1.0.safetensors"),
            "temporal_upsampler_path": os.path.join(models_dir, "upscalers", "ltx-2-temporal-upscaler-x2-1.0.safetensors"),
            "gemma_root": os.path.join(models_dir, "text_encoders", "gemma3"),
            # Placeholder for IP-Adapters
            "flux_ip_adapter": config.FLUX_IP_ADAPTER_PATH,
        }

    async def load_pipeline(self, pipeline_type: str, loras: list = None):
        if loras is None:
            loras = []

        async with self._lock:
            # Check if we already have this pipeline loaded
            # Note: For strict correctness, we should also check if loras changed, 
            # but for now we assume loras are mostly static or we reload if needed.
            # Simpler: just check type for now.
            if self._pipeline is not None and self._pipeline_type == pipeline_type:
                logger.info(f"Pipeline {pipeline_type} already loaded.")
                return self._pipeline

            # Unload existing
            if self._pipeline is not None:
                logger.info("Unloading previous pipeline...")
                del self._pipeline
                self._pipeline = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            
            logger.info(f"Loading LTX-2 Pipeline: {pipeline_type}...")
            
            # Unload conflicting models (e.g., Flux) before loading LTX
            from memory_manager import memory_manager
            memory_manager.prepare_for("video")
            
            paths = self.get_model_paths()
            
            # Check device
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            logger.info(f"Using device: {device}")
            
            # Common args
            distilled_lora_objs = []
            if os.path.exists(paths["distilled_lora_path"]):
                distilled_lora_objs.append(
                    LoraPathStrengthAndSDOps(paths["distilled_lora_path"], 1.0, None)
                )

            is_mps = (device == "mps")
            fp8 = False if is_mps else True 
            
            if pipeline_type == "ti2vid":
                self._pipeline = TI2VidTwoStagesPipeline(
                    checkpoint_path=paths["checkpoint_path"],
                    distilled_lora=distilled_lora_objs,
                    spatial_upsampler_path=paths["spatial_upsampler_path"],
                    temporal_upsampler_path=paths["temporal_upsampler_path"],
                    gemma_root=paths["gemma_root"],
                    loras=loras,
                    device=device,
                    fp8transformer=fp8
                )
            elif pipeline_type == "ic_lora":
                # ICLoraPipeline takes 'loras' as the specific IC-LoRA
                self._pipeline = ICLoraPipeline(
                    checkpoint_path=paths["checkpoint_path"],
                    spatial_upsampler_path=paths["spatial_upsampler_path"],
                    gemma_root=paths["gemma_root"],
                    loras=loras, # These will be the IC-LoRAs
                    device=device,
                    fp8transformer=fp8
                )
            elif pipeline_type == "keyframe":
                self._pipeline = KeyframeInterpolationPipeline(
                    checkpoint_path=paths["checkpoint_path"],
                    distilled_lora=distilled_lora_objs,
                    spatial_upsampler_path=paths["spatial_upsampler_path"],
                    gemma_root=paths["gemma_root"],
                    loras=loras,
                    device=device,
                    fp8transformer=fp8
                )
            else:
                raise ValueError(f"Unknown pipeline type: {pipeline_type}")

            self._pipeline_type = pipeline_type
            
            # MPS VAE Fix: Force Float32 to prevent black output
            if device == "mps" and hasattr(self._pipeline, "vae"):
                logger.info("MPS detected: Converting VAE to float32 to prevent black output...")
                self._pipeline.vae = self._pipeline.vae.to(dtype=torch.float32)

            logger.info(f"LTX-2 Pipeline {pipeline_type} loaded successfully.")
            return self._pipeline

# Singleton instance
manager = ModelManager()
