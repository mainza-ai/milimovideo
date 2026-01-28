# ruff: noqa: PLC0415

"""
8-bit Gemma text encoder loading utilities.
This module provides functionality for loading the Gemma text encoder in 8-bit precision
using bitsandbytes, which significantly reduces GPU memory usage.
Example usage:
    from ltx_trainer.gemma_8bit import load_8bit_gemma
    text_encoder = load_8bit_gemma(
        checkpoint_path="/path/to/ltx2.safetensors",
        gemma_model_path="/path/to/gemma",
    )
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader
from ltx_core.text_encoders.gemma.embeddings_connector import Embeddings1DConnectorConfigurator
from ltx_core.text_encoders.gemma.encoders.av_encoder import (
    AV_GEMMA_TEXT_ENCODER_KEY_OPS,
    AVGemmaTextEncoderModel,
)
from ltx_core.text_encoders.gemma.feature_extractor import GemmaFeaturesExtractorProjLinear
from ltx_core.text_encoders.gemma.tokenizer import LTXVGemmaTokenizer

if TYPE_CHECKING:
    from ltx_core.text_encoders.gemma import AVGemmaTextEncoderModel


def load_8bit_gemma(
    checkpoint_path: str | Path,
    gemma_model_path: str | Path,
    dtype: torch.dtype = torch.bfloat16,
) -> "AVGemmaTextEncoderModel":
    """Load the Gemma text encoder in 8-bit precision using bitsandbytes.
    This function bypasses ltx-core's standard loading path to enable 8-bit quantization
    via the bitsandbytes library. The Gemma model is loaded with load_in_8bit=True and
    torch_dtype=bfloat16, while the feature extractor and connector weights are loaded
    from the LTX-2 checkpoint.
    Args:
        checkpoint_path: Path to the LTX-2 safetensors checkpoint file
        gemma_model_path: Path to Gemma model directory
        dtype: Data type for non-quantized model weights (feature extractor, connectors)
    Returns:
        Loaded AVGemmaTextEncoderModel with 8-bit quantized Gemma backbone
    Raises:
        ImportError: If bitsandbytes is not installed
        FileNotFoundError: If required model files are not found
    """
    try:
        from transformers import BitsAndBytesConfig, Gemma3ForConditionalGeneration
    except ImportError as e:
        raise ImportError(
            "8-bit text encoder loading requires bitsandbytes. Install it with: uv pip install bitsandbytes"
        ) from e

    # Find paths within gemma_model_path
    gemma_path = _find_gemma_subpath(gemma_model_path, "model*.safetensors")
    tokenizer_path = _find_gemma_subpath(gemma_model_path, "tokenizer.model")

    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    with _suppress_accelerate_memory_warnings():
        gemma_model = Gemma3ForConditionalGeneration.from_pretrained(
            gemma_path,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
        )

    # Load tokenizer
    tokenizer = LTXVGemmaTokenizer(tokenizer_path, 1024)

    # Load config and weights from the LTX-2 checkpoint
    loader = SafetensorsModelStateDictLoader()
    config = loader.metadata(str(checkpoint_path))
    sd = loader.load(str(checkpoint_path), sd_ops=AV_GEMMA_TEXT_ENCODER_KEY_OPS)

    # Helper to extract state dict for a given prefix
    def extract_state_dict(prefix: str) -> dict[str, torch.Tensor]:
        return {k.replace(prefix, ""): v for k, v in sd.sd.items() if k.startswith(prefix)}

    # Create and load feature extractor
    feature_extractor = GemmaFeaturesExtractorProjLinear()
    feature_extractor.load_state_dict(extract_state_dict("feature_extractor_linear."))
    feature_extractor = feature_extractor.to(device=gemma_model.device, dtype=dtype)

    # Create and load video embeddings connector
    embeddings_connector = Embeddings1DConnectorConfigurator.from_config(config)
    embeddings_connector.load_state_dict(extract_state_dict("embeddings_connector."))
    embeddings_connector = embeddings_connector.to(device=gemma_model.device, dtype=dtype)

    # Create and load audio embeddings connector
    audio_embeddings_connector = Embeddings1DConnectorConfigurator.from_config(config)
    audio_embeddings_connector.load_state_dict(extract_state_dict("audio_embeddings_connector."))
    audio_embeddings_connector = audio_embeddings_connector.to(device=gemma_model.device, dtype=dtype)

    # Construct the text encoder
    text_encoder = AVGemmaTextEncoderModel(
        feature_extractor_linear=feature_extractor,
        embeddings_connector=embeddings_connector,
        audio_embeddings_connector=audio_embeddings_connector,
        tokenizer=tokenizer,
        model=gemma_model,
    )

    return text_encoder


def _find_gemma_subpath(root_path: str | Path, pattern: str) -> str:
    """Find a file matching a glob pattern and return its parent directory."""
    matches = list(Path(root_path).rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matching pattern '{pattern}' found under {root_path}")
    return str(matches[0].parent)


@contextmanager
def _suppress_accelerate_memory_warnings() -> Generator[None, None, None]:
    """Temporarily suppress INFO warnings from accelerate about memory allocation."""
    accelerate_logger = logging.getLogger("accelerate.utils.modeling")
    old_level = accelerate_logger.level
    accelerate_logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        accelerate_logger.setLevel(old_level)
