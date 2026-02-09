import logging
from typing import Callable
from collections.abc import Iterator
from dataclasses import replace

import torch

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.guiders import CFGGuider
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.protocols import DiffusionStepProtocol
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
from ltx_core.model.upsampler import upsample_video
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.model.video_vae import decode_video as vae_decode_video
from ltx_core.text_encoders.gemma import encode_text
from ltx_core.types import LatentState, VideoPixelShape
from ltx_pipelines.utils import ModelLedger
from ltx_pipelines.utils.args import default_2_stage_arg_parser
from ltx_pipelines.utils.constants import (
    AUDIO_SAMPLE_RATE,
    STAGE_2_DISTILLED_SIGMA_VALUES,
)
from ltx_pipelines.utils.helpers import (
    assert_resolution,
    cleanup_memory,
    denoise_audio_video,
    euler_denoising_loop,
    generate_enhanced_prompt,
    get_device,
    guider_denoising_func,
    image_conditionings_by_replacing_latent,
    simple_denoising_func,
    smart_inference_mode,
    synchronize_device,
)
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.types import PipelineComponents

device = get_device()


class TI2VidTwoStagesPipeline:
    """
    Two-stage text/image-to-video generation pipeline.
    Stage 1 generates video at the target resolution with CFG guidance, then
    Stage 2 upsamples by 2x and refines using a distilled LoRA for higher
    quality output. Supports optional image conditioning via the images parameter.
    """

    def __init__(
        self,
        checkpoint_path: str,
        distilled_lora: list[LoraPathStrengthAndSDOps],
        spatial_upsampler_path: str,
        gemma_root: str,
        loras: list[LoraPathStrengthAndSDOps],
        temporal_upsampler_path: str | None = None, # Added
        device: str = device,
        fp8transformer: bool = False,
    ):
        self.device = device
        self.dtype = torch.bfloat16
        self.stage_1_model_ledger = ModelLedger(
            dtype=self.dtype,
            device=device,
            checkpoint_path=checkpoint_path,
            gemma_root_path=gemma_root,
            spatial_upsampler_path=spatial_upsampler_path,
            temporal_upsampler_path=temporal_upsampler_path, # Added
            loras=loras,
            fp8transformer=fp8transformer,
        )

        self.stage_2_model_ledger = self.stage_1_model_ledger.with_loras(
            loras=distilled_lora,
        )

        self.pipeline_components = PipelineComponents(
            dtype=self.dtype,
            device=device,
        )

    @smart_inference_mode()
    def __call__(
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        num_inference_steps: int,
        cfg_guidance_scale: float,
        images: list[tuple[str, int, float]],
        previous_latent_tensor: torch.Tensor | None = None, # New: Latent Handoff
        tiling_config: TilingConfig | None = None,
        enhance_prompt: bool = False,
        upscale: bool = True,
        callback_on_step_end: Callable | None = None,
    ) -> tuple[Iterator[torch.Tensor], torch.Tensor, LatentState]:
        logging.info("DEBUG: TI2VidTwoStagesPipeline.__call__ START")
        assert_resolution(height=height, width=width, is_two_stage=True)
        # ... (setup code skipped, see diff) ...
        # (imports inside function to avoid circular dep if needed, or assume global)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()
        cfg_guider = CFGGuider(cfg_guidance_scale)
        dtype = self.dtype

        text_encoder = self.stage_1_model_ledger.text_encoder()
        if enhance_prompt:
            prompt = generate_enhanced_prompt(
                text_encoder, prompt, images[0][0] if len(images) > 0 else None, seed=seed
            )
        context_p, context_n = encode_text(text_encoder, prompts=[prompt, negative_prompt])
        v_context_p, a_context_p = context_p
        v_context_n, a_context_n = context_n

        synchronize_device()
        del text_encoder
        cleanup_memory()

        # Stage 1: Initial low resolution video generation.
        video_encoder = self.stage_1_model_ledger.video_encoder()
        transformer = self.stage_1_model_ledger.transformer()
        sigmas = LTX2Scheduler().execute(steps=num_inference_steps).to(dtype=torch.float32, device=self.device)

        def first_stage_denoising_loop(
            sigmas: torch.Tensor,
            video_state: LatentState,
            audio_state: LatentState,
            stepper: DiffusionStepProtocol,
            callback_on_step_end: Callable | None = None,
        ) -> tuple[LatentState, LatentState]:
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=guider_denoising_func(
                    cfg_guider,
                    v_context_p,
                    v_context_n,
                    a_context_p,
                    a_context_n,
                    transformer=transformer,
                ),
                callback_on_step_end=callback_on_step_end,
            )

        stage_1_output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width // 2,
            height=height // 2,
            fps=frame_rate,
        )
        
        # CONDITIONING PREP
        stage_1_conditionings = image_conditionings_by_replacing_latent(
            images=images,
            height=stage_1_output_shape.height,
            width=stage_1_output_shape.width,
            video_encoder=video_encoder,
            dtype=dtype,
            device=self.device,
        )
        
        # LATENT HANDOFF (Stage 1 Downsample)
        if previous_latent_tensor is not None:
             # ... existing handoff logic ...
             from ltx_pipelines.utils.helpers import latent_conditionings_from_tensor
             # Downsample H, W
             s1_lat = torch.nn.functional.interpolate(
                previous_latent_tensor, 
                scale_factor=(1.0, 0.5, 0.5), 
                mode='trilinear',
                align_corners=False
            ).to(dtype=dtype)
             s1_items = latent_conditionings_from_tensor(s1_lat, start_frame_idx=0, strength=1.0)
             stage_1_conditionings.extend(s1_items)

        video_state, audio_state = denoise_audio_video(
            output_shape=stage_1_output_shape,
            conditionings=stage_1_conditionings,
            noiser=noiser,
            sigmas=sigmas,
            stepper=stepper,
            denoising_loop_fn=first_stage_denoising_loop,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            callback_on_step_end=callback_on_step_end,
        )

        synchronize_device()
        del transformer
        cleanup_memory()

        # Stage 2: Upsample
        upscaled_video_latent = upsample_video(
            latent=video_state.latent[:1],
            video_encoder=video_encoder,
            upsampler=self.stage_2_model_ledger.spatial_upsampler(),
        )

        synchronize_device()
        cleanup_memory()

        transformer = self.stage_2_model_ledger.transformer()
        distilled_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(self.device)

        def second_stage_denoising_loop(
            sigmas: torch.Tensor,
            video_state: LatentState,
            audio_state: LatentState,
            stepper: DiffusionStepProtocol,
            callback_on_step_end: Callable | None = None,
        ) -> tuple[LatentState, LatentState]:
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=simple_denoising_func(
                    video_context=v_context_p,
                    audio_context=a_context_p,
                    transformer=transformer,  # noqa: F821
                ),
                callback_on_step_end=callback_on_step_end,
            )

        stage_2_output_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=frame_rate)
        
        stage_2_conditionings = image_conditionings_by_replacing_latent(
            images=images,
            height=stage_2_output_shape.height,
            width=stage_2_output_shape.width,
            video_encoder=video_encoder,
            dtype=dtype,
            device=self.device,
        )
        
        # LATENT HANDOFF (Stage 2 Direct)
        if previous_latent_tensor is not None:
            # Use directly
            from ltx_pipelines.utils.helpers import latent_conditionings_from_tensor
            s2_items = latent_conditionings_from_tensor(previous_latent_tensor, start_frame_idx=0, strength=1.0)
            stage_2_conditionings.extend(s2_items)

        video_state, audio_state = denoise_audio_video(
            output_shape=stage_2_output_shape,
            conditionings=stage_2_conditionings,
            noiser=noiser,
            sigmas=distilled_sigmas,
            stepper=stepper,
            denoising_loop_fn=second_stage_denoising_loop,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            noise_scale=distilled_sigmas[0],
            initial_video_latent=upscaled_video_latent,
            initial_audio_latent=audio_state.latent,
            callback_on_step_end=callback_on_step_end,
        )
        
        synchronize_device()
        del transformer
        cleanup_memory()

        # Stage 3: Temporal Upsample (Optional)
        # If the model ledger has a temporal upsampler AND upscale is requested, apply it.
        if upscale:
            try:
                temporal_upsampler = self.stage_2_model_ledger.temporal_upsampler()
                logging.info("Applying Temporal Upsample (Stage 3)...")
                
                # Upsample
                time_upscaled_latent = upsample_video(
                    latent=video_state.latent,
                    video_encoder=video_encoder,
                    upsampler=temporal_upsampler,
                )
                
                video_state = replace(video_state, latent=time_upscaled_latent)
                
                # Audio handling (if present)
                if audio_state.latent is not None:
                    # For LTX-2, audio might not be upsampled the same way, 
                    # but we keep the logic consistent if audio upsampler exists (rare)
                    pass
                
                synchronize_device()
                del temporal_upsampler
                cleanup_memory()

            except ValueError:
                # No temporal upsampler configured, continue.
                logging.info("Stage 3 Skipped: No temporal upsampler found in ledger.")
                pass
            except Exception as e:
                logging.warning(f"Temporal Upsample Failed: {e}. proceed with Stage 2 output.")
                pass
        else:
            logging.info("Stage 3 Skipped: Upscale disabled by user.")
        
        # Final cleanup for video encoder before decode
        del video_encoder 
        cleanup_memory()
        
        decoded_video = vae_decode_video(
            video_state.latent, self.stage_2_model_ledger.video_decoder(), tiling_config, generator
        )
        decoded_audio = vae_decode_audio(
            audio_state.latent, self.stage_2_model_ledger.audio_decoder(), self.stage_2_model_ledger.vocoder()
        )

        # Return LatentState for chaining
        logging.info(f"DEBUG: TI2VidTwoStagesPipeline.__call__ END. Returning: Video type={type(decoded_video)}")
        return decoded_video, decoded_audio, video_state.latent


@smart_inference_mode()
def main() -> None:
    logging.getLogger().setLevel(logging.INFO)
    parser = default_2_stage_arg_parser()
    args = parser.parse_args()
    pipeline = TI2VidTwoStagesPipeline(
        checkpoint_path=args.checkpoint_path,
        distilled_lora=args.distilled_lora,
        spatial_upsampler_path=args.spatial_upsampler_path,
        gemma_root=args.gemma_root,
        loras=args.lora,
        fp8transformer=args.enable_fp8,
    )
    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(args.num_frames, tiling_config)
    video, audio = pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        num_inference_steps=args.num_inference_steps,
        cfg_guidance_scale=args.cfg_guidance_scale,
        images=args.images,
        tiling_config=tiling_config,
    )

    encode_video(
        video=video,
        fps=args.frame_rate,
        audio=audio,
        audio_sample_rate=AUDIO_SAMPLE_RATE,
        output_path=args.output_path,
        video_chunks_number=video_chunks_number,
    )


if __name__ == "__main__":
    main()
