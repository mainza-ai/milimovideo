
import logging
import os
import sys
import torch
import numpy as np
from PIL import Image
from einops import rearrange
import config

# Add Flux2 source to path
FLUX_SRC_DIR = os.path.join(config.PROJECT_ROOT, "flux2", "src")
if FLUX_SRC_DIR not in sys.path:
    sys.path.append(FLUX_SRC_DIR)

logger = logging.getLogger("flux_wrapper")

class FluxAEWrapper:
    def __init__(self, ae_path, device, dtype=torch.bfloat16):
        from diffusers import AutoencoderKL
        self.device = device
        self.dtype = dtype
        
        # Try loading with low_cpu_mem_usage=False to avoid accelerate issues if any
        try:
            self.ae = AutoencoderKL.from_pretrained(
                ae_path if os.path.isdir(ae_path) else os.path.dirname(ae_path),
                torch_dtype=dtype
            ).to(device)
        except Exception:
            # Try single file if likely
            if ae_path.endswith(".safetensors"):
                self.ae = AutoencoderKL.from_single_file(
                    ae_path,
                    torch_dtype=dtype
                ).to(device)
            else:
                raise
                
        self.ae.eval()

    def encode(self, x):
        # x: (B, C, H, W)
        with torch.no_grad():
             dist = self.ae.encode(x).latent_dist
             z = dist.mode() 
             # Rearrange: (B, 32, H, W) -> (B, 128, H/2, W/2)
             z = rearrange(z, "b c (h pi) (w pj) -> b (c pi pj) h w", pi=2, pj=2)
             return z

    def decode(self, z):
        # z: (B, 128, H/2, W/2)
        z = rearrange(z, "b (c pi pj) h w -> b c (h pi) (w pj)", pi=2, pj=2, c=32)
        
        
        # MPS Hack: VAE Decode is unstable in float16/bfloat16
        # Force to float32 for the decode step only.
        # AND MOST IMPORTANTLY: Run decode on CPU if on MPS to avoid the "Black Image" bug 100%.
        
        use_cpu_offload = self.device == "mps" or self.device == torch.device("mps") or str(self.device) == "mps"
        print(f"[FluxWrapper] Decoding. Device: {self.device}, Type: {type(self.device)}, Offload CPU: {use_cpu_offload}")
        
        with torch.no_grad():
            if use_cpu_offload:
                 # Move strictly to CPU and float32
                 self.ae = self.ae.to(device="cpu", dtype=torch.float32)
                 z = z.to(device="cpu", dtype=torch.float32)
            
            dec = self.ae.decode(z).sample
            
            if use_cpu_offload:
                # Move back to original device/dtype
                # dec is on CPU now
                dec = dec.to(device=self.device, dtype=torch.float32) 
                
                # Restore model to original device
                self.ae = self.ae.to(device=self.device, dtype=self.dtype)
                
        return dec
                
        return dec

    def to(self, device):
        self.ae.to(device)
        self.device = device
        return self

class FluxInpainter:
    def __init__(self, device=None):
        if device:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        
        # MPS doesn't support bfloat16 well.
        # float16 often causes NaNs in deep transformers (Flux) on MPS.
        # We FORCE float32 for MPS to guarantee stability, even if it uses more RAM.
        if self.device == "mps":
             self.dtype = torch.float32 
        else:
             self.dtype = torch.bfloat16
        
        logger.info(f"FluxInpainter initialized on {self.device} with {self.dtype}")

        self.model = None
        self.ae = None
        self.text_encoder = None
        self.model_loaded = False
        self.params = None
        self.defaults = {}

    def load_model(self):
        if self.model_loaded:
            return

        logger.info(f"Loading Flux 2 (Klein) Model on {self.device}...")

        try:
            from flux2.util import load_ae, load_flow_model, load_text_encoder, FLUX2_MODEL_INFO
            from flux2.sampling import get_schedule, batched_prc_img, batched_prc_txt

            # Setup Paths
            base_path = config.FLUX_WEIGHTS_PATH
            
            # Set environment variables for util.py
            os.environ["KLEIN_9B_MODEL_PATH"] = os.path.join(base_path, "flux-2-klein-9b.safetensors")
            
            # Check for AE in vae/
            ae_path_dir = os.path.join(base_path, "vae")
            ae_path_file = os.path.join(base_path, "ae.safetensors") # fallback
            
            # Decide AE path
            final_ae_path = None
            if os.path.exists(ae_path_dir) and os.path.exists(os.path.join(ae_path_dir, "config.json")):
                 final_ae_path = ae_path_dir
            elif os.path.exists(ae_path_file):
                 final_ae_path = ae_path_file
            
            # Set Qwen paths
            qwen_path = os.path.join(base_path, "text_encoder")
            qwen_tokenizer_path = os.path.join(base_path, "tokenizer")
            if os.path.exists(qwen_path):
                os.environ["QWEN3_8B_PATH"] = qwen_path
            if os.path.exists(qwen_tokenizer_path):
                os.environ["QWEN3_8B_TOKENIZER_PATH"] = qwen_tokenizer_path
            
            model_name = "flux.2-klein-9b"
            
            # Load components
            logger.info("Loading Text Encoder...")
            # We must pass device to load_text_encoder
            self.text_encoder = load_text_encoder(model_name, device=self.device)
            
            logger.info("Loading AutoEncoder (Diffusers Wrapper)...")
            if final_ae_path:
                self.ae = FluxAEWrapper(final_ae_path, self.device, dtype=self.dtype)
            else:
                logger.warning("Local VAE not found, trying native load...")
                self.ae = load_ae(model_name, device=self.device)

            logger.info("Loading Flow Model...")
            self.model = load_flow_model(model_name, device=self.device)
            
            # Cast model to correct dtype (float32 for MPS to avoid NaNs, bf16 for CUDA)
            self.model.to(dtype=self.dtype)
            
            self.model_loaded = True
            logger.info("Flux 2 Model Loaded Successfully")
            
        except ImportError as e:
            logger.error(f"Failed to import flux2 modules: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load Flux 2: {e}")
            raise

    def denoise_inpaint(self, model, x, x_ids, ctx, ctx_ids, timesteps, guidance, 
                        orig_image=None, mask=None):
        """
        Custom denoise loop with RePaint-style inpainting.
        """
        guidance_vec = torch.full((x.shape[0],), guidance, device=x.device, dtype=x.dtype)
        
        for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            if i % 5 == 0:
                logger.info(f"Denoising Step {i+1}/{len(timesteps)-1}")

            t_vec = torch.full((x.shape[0],), t_curr, dtype=x.dtype, device=x.device)
            
            # Model prediction
            pred = model(
                x=x,
                x_ids=x_ids,
                timesteps=t_vec,
                ctx=ctx,
                ctx_ids=ctx_ids,
                guidance=guidance_vec,
            )
            
            # Step update (Euler / flow matching)
            x_pred = x + (t_prev - t_curr) * pred
            
            # Inpainting / Re-noising (RePaint)
            if orig_image is not None and mask is not None:
                # Simpler approximation: Blend x_pred and x_known.
                noise = torch.randn_like(orig_image)
                x_known = t_prev * noise + (1 - t_prev) * orig_image
                
                x_pred = mask * x_pred + (1 - mask) * x_known
            
            x = x_pred

        return x

    def _trace(self, name, tensor):
        if tensor is None:
            logger.info(f"[Trace] {name}: None")
            return
        if not isinstance(tensor, torch.Tensor):
            logger.info(f"[Trace] {name}: Type={type(tensor)}")
            return
            
        min_v = tensor.min().item()
        max_v = tensor.max().item()
        mean_v = tensor.mean().item()
        has_nan = torch.isnan(tensor).any().item()
        logger.info(f"[Trace] {name}: Shape={tensor.shape}, Range=[{min_v:.4f}, {max_v:.4f}], Mean={mean_v:.4f}, HasNaN={has_nan}")

    def generate_image(self, prompt: str, width: int = 1024, height: int = 1024, guidance: float = 3.5) -> Image.Image:
        if not self.model_loaded:
            self.load_model()

        logger.info(f"Generating Image with Flux 2. Prompt: {prompt}")
        
        from flux2.sampling import get_schedule, batched_prc_img, batched_prc_txt, scatter_ids

        try:
            with torch.no_grad():
                # 1. Prepare Text
                ctx = self.text_encoder([prompt]).to(self.dtype)
                ctx, ctx_ids = batched_prc_txt(ctx)
                
                # 2. Prepare Latents (Noise)
                W = (width // 16) * 16
                H = (height // 16) * 16
                
                dummy_img = Image.new("RGB", (W, H), (0, 0, 0))
                img_tensor = torch.from_numpy(np.array(dummy_img)).float() / 127.5 - 1.0
                img_tensor = rearrange(img_tensor, "h w c -> 1 c h w").to(self.device).to(self.dtype)
                z_shape_ref = self.ae.encode(img_tensor)
                
                x = torch.randn_like(z_shape_ref)
                self._trace("Initial Latents (x)", x)
                if torch.isnan(x).any():
                    logger.warning("[FluxInpainter] WARNING: Initial latents contain NaNs!")

                x, x_ids = batched_prc_img(x)
                self._trace("Initial Latents (x) after batched_prc_img", x)
                
                # 3. Denoise Loop
                timesteps = get_schedule(25, x.shape[1]) 
                
                x_out = self.denoise_inpaint(
                    self.model,
                    x, x_ids,
                    ctx, ctx_ids,
                    timesteps,
                    guidance,
                    orig_image=None, 
                    mask=None
                )
                self._trace("Latents After Denoise (x_out)", x_out)
                if torch.isnan(x_out).any():
                    logger.critical("[FluxInpainter] CRITICAL: Latents after denoise contain NaNs!")
                
                # 4. Decode
                x_out = torch.cat(scatter_ids(x_out, x_ids)).squeeze(2)
                decoded = self.ae.decode(x_out).float()
                self._trace("Decoded Image Tensor", decoded)
                if torch.isnan(decoded).any():
                    logger.warning("[FluxInpainter] WARNING: Decoded image tensor contains NaNs!")
                
                decoded = decoded.clamp(-1, 1)
                decoded = rearrange(decoded[0], "c h w -> h w c")
                out_img = Image.fromarray((127.5 * (decoded + 1.0)).cpu().byte().numpy())
                
                return out_img

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise e

    def unload(self):
        self.model = None
        self.ae = None
        self.text_encoder = None
        self.model_loaded = False
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Flux Model Unloaded")

flux_inpainter = FluxInpainter()
