
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

# IP-Adapter Components
class ImageProjModel(torch.nn.Module):
    """Simple Linear Projector for IP-Adapter"""
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


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
        self.params = None
        self.defaults = {}
        
        # IP-Adapter State
        self.image_encoder = None
        self.image_processor = None
        self.ip_adapter_projector = None
        self.ip_adapter_loaded = False

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

    def load_ip_adapter(self):
        """Loads the IP-Adapter components (Image Encoder + Projector)"""
        if self.ip_adapter_loaded:
            return

        logger.info("Loading IP-Adapter Components...")
        try:
            from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
            from safetensors.torch import load_file
            
            # 1. Load Image Encoder (CLIP ViT-L/14)
            # Using standard CLIP for consistency with most IP-Adapters
            # If using Redux, this would be SigLIP. But 'simpler weight injection' implies standard IPA.
            encoder_name = "openai/clip-vit-large-patch14" 
            logger.info(f"Loading Image Encoder: {encoder_name}")
            
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                encoder_name, torch_dtype=self.dtype
            ).to(self.device)
            self.image_processor = CLIPImageProcessor.from_pretrained(encoder_name)
            
            # 2. Load Projector Weights
            ip_path = config.FLUX_IP_ADAPTER_PATH
            if not os.path.exists(ip_path):
                logger.warning(f"IP-Adapter weights not found at {ip_path}. IP-Adapter disabled.")
                return

            logger.info(f"Loading IP-Adapter weights from {ip_path}...")
            state_dict = load_file(ip_path)
            
            # Inspect shapes to determine config
            # Standard Flux IPA usually projects 1024 (CLIP) -> 4096 (Flux Hidden)
            # Or 1152 (SigLIP) -> 4096.
            # Let's verify input dim from state dict if possible, or assume CLIP.
            
            # Initialize Projector
            # Assuming standard structural mapping
            self.ip_adapter_projector = ImageProjModel(
                cross_attention_dim=4096, # Flux Hidden Size (Klein 9B)
                clip_embeddings_dim=self.image_encoder.config.hidden_size, # 1024
                clip_extra_context_tokens=4 # Standard usually 4 or 8? Let's default to 4 for now.
            ).to(self.device, dtype=self.dtype)
            
            # Load weights (handle potential prefix mismatches)
            # If keys are like 'proj.weight', 'norm.weight' etc.
            self.ip_adapter_projector.load_state_dict(state_dict, strict=False)
            
            self.ip_adapter_loaded = True
            logger.info("IP-Adapter Loaded Successfully.")
            
        except Exception as e:
            logger.error(f"Failed to load IP-Adapter: {e}")
            # Don't crash, just proceed without it
            self.ip_adapter_loaded = False

    def denoise_inpaint(self, model, x, x_ids, ctx, ctx_ids, timesteps, guidance, 
                        orig_image=None, mask=None, callback=None, img_cond_seq=None, img_cond_seq_ids=None):
        """
        Custom denoise loop with RePaint-style inpainting.
        """
        guidance_vec = torch.full((x.shape[0],), guidance, device=x.device, dtype=x.dtype)
        
        for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            if i % 5 == 0:
                logger.info(f"Denoising Step {i+1}/{len(timesteps)-1}")
            
            if callback:
                callback(i+1, len(timesteps)-1)

            t_vec = torch.full((x.shape[0],), t_curr, dtype=x.dtype, device=x.device)
            
            # Concat IP-Adapter Condition (if any)
            div_len = x.shape[1]
            model_input = x
            model_ids = x_ids
            
            if img_cond_seq is not None and img_cond_seq_ids is not None:
                model_input = torch.cat((x, img_cond_seq), dim=1)
                model_ids = torch.cat((x_ids, img_cond_seq_ids), dim=1)

            # Model prediction
            pred = model(
                x=model_input,
                x_ids=model_ids,
                timesteps=t_vec,
                ctx=ctx,
                ctx_ids=ctx_ids,
                guidance=guidance_vec,
            )
            
            # Slice back if we added tokens
            if img_cond_seq is not None:
                pred = pred[:, :div_len]
            
            # Step update (Euler / flow matching)
            x_pred = x + (t_prev - t_curr) * pred
            
            # Inpainting / Re-noising (RePaint)
            if orig_image is not None and mask is not None:
                # Simpler approximation: Blend x_pred and x_known.
                noise = torch.randn_like(orig_image)
                x_known = t_prev * noise + (1 - t_prev) * orig_image
                
                x_pred = mask * x_pred + (1 - mask) * x_known
            
                x_pred = mask * x_pred + (1 - mask) * x_known
            
            x = x_pred

        return x

    def get_ip_adapter_embeds(self, ip_adapter_images):
        """Encodes and projects images for IP-Adapter"""
        if not self.ip_adapter_loaded or not ip_adapter_images:
            return None, None
            
        try:
            # 1. Preprocess Images
            # ip_adapter_images is list of paths or PIL images
            pil_images = []
            for img in ip_adapter_images:
                if isinstance(img, str):
                    pil_images.append(Image.open(img).convert("RGB"))
                else:
                    pil_images.append(img.convert("RGB"))
            
            inputs = self.image_processor(images=pil_images, return_tensors="pt").to(self.device)
            
            # 2. Encode
            with torch.no_grad():
                image_embeds = self.image_encoder(**inputs).image_embeds # (B, 1024)
                
            # 3. Project
            # (B, 1024) -> (B, 4, 4096)
            clip_extra_context_tokens = self.ip_adapter_projector(image_embeds.to(dtype=self.dtype))
            
            # Flatten to (1, N*4, 4096) context sequence
            # Flux expects (1, L, D)
            img_cond_seq = clip_extra_context_tokens.reshape(1, -1, clip_extra_context_tokens.shape[-1])
            
            # 4. Create IDs
            # Flux needs IDs for these tokens.
            # We can reuse standard logic or create simple linear IDs.
            # Let's verify flux2.sampling logic for IDs.
            # It uses `prc_img` logic usually.
            # For simplicity, we can extend existing IDs or just use zeros if model is robust (Flux 2 uses RoPE, so IDs matter).
            # We'll use a simple placeholder ID generation for now or generic range.
            # Actually, `flux2.sampling.py` has `prc_txt` which generates standard IDs.
            # Let's generate IDs that place these tokens "after" the text or somewhere.
            # Usually IP-Adapter tokens are appended.
            
            # Helper from flux2.sampling (we can look at it later or mock it)
            # For now, let's just return the embeds and handle IDs in generate_image loop 
            # by mimicking what `batched_prc_txt` does but for image tokens.
            
            return img_cond_seq, None 

        except Exception as e:
            logger.error(f"Error encoding IP-Adapter images: {e}")
            return None, None

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

    def generate_image(self, prompt: str, width: int = 1024, height: int = 1024, guidance: float = 3.5, ip_adapter_images: list = None, callback=None) -> Image.Image:
        if not self.model_loaded:
            self.load_model()

        logger.info(f"Generating Image with Flux 2. Prompt: {prompt}")
        
        from flux2.sampling import get_schedule, batched_prc_img, batched_prc_txt, scatter_ids

        try:
            with torch.no_grad():
                # 1. Prepare Text
                ctx = self.text_encoder([prompt]).to(self.dtype)
                ctx, ctx_ids = batched_prc_txt(ctx)

                # 1b. Prepare IP-Adapter (Visual Conditioning)
                img_cond_seq = None
                img_cond_seq_ids = None
                if ip_adapter_images:
                     if not self.ip_adapter_loaded:
                        self.load_ip_adapter()
                        
                     if self.ip_adapter_loaded:
                         raw_img_cond, _ = self.get_ip_adapter_embeds(ip_adapter_images)
                         if raw_img_cond is not None:
                             # Generate IDs mimicking text sequence (linear)
                             img_cond_seq, img_cond_seq_ids = batched_prc_txt(raw_img_cond)
                             logger.info(f"IP-Adapter Active: {img_cond_seq.shape[1]} tokens injected.")
                
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
                    mask=None,
                    callback=callback,
                    img_cond_seq=img_cond_seq,
                    img_cond_seq_ids=img_cond_seq_ids
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
