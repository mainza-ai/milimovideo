"""
LLM prompt enhancement — configurable provider (Gemma or Ollama).

Usage:
    from llm import enhance_prompt
    enhanced = enhance_prompt(prompt, system_prompt="...", is_video=True)
"""
import logging
import requests
import config

logger = logging.getLogger(__name__)

# ── Video-Aware System Prompt (shared across providers) ──────────
VIDEO_SYSTEM_PROMPT = (
    "You are a Creative Assistant writing concise, action-focused image-to-video prompts. "
    "Given a user's Raw Input Prompt, generate a richly detailed prompt to guide video generation.\n\n"
    "#### Guidelines:\n"
    "- Expand the user's prompt with vivid visual details, motion, camera movement, and atmosphere.\n"
    "- Active language: Use present-progressive verbs ('is walking,' 'speaking').\n"
    "- Chronological flow: Use temporal connectors ('as,' 'then,' 'while').\n"
    "- Speech (only when requested): Provide exact words in quotes with voice characteristics.\n"
    "- Style: Include visual style at beginning: 'Style: <style>, <rest of prompt>.' If unclear, omit.\n"
    "- Visual and audio: Describe complete soundscape alongside actions.\n"
    "- Restrained language: Avoid dramatic terms. Use mild, natural, understated phrasing.\n"
    "- Formatting: Single concise paragraph (3-5 sentences). No titles or markdown.\n"
    "- Output ONLY the complete prompt text."
)

IMAGE_SYSTEM_PROMPT = (
    "You are a Creative Assistant writing detailed image generation prompts. "
    "Transform the user's brief description into a rich, detailed image prompt.\n\n"
    "#### Guidelines:\n"
    "- Expand with visual details: lighting, composition, colors, textures, atmosphere.\n"
    "- Include style cues: photography style, art medium, or rendering technique.\n"
    "- Be specific about subject positioning, expression, and environment.\n"
    "- Formatting: Single concise paragraph (2-4 sentences). No titles or markdown.\n"
    "- Output ONLY the complete prompt text."
)


def enhance_prompt_ollama(
    prompt: str,
    system_prompt: str | None = None,
    is_video: bool = True,
) -> str:
    """Enhance a prompt using Ollama's local LLM API."""
    if not system_prompt:
        system_prompt = VIDEO_SYSTEM_PROMPT if is_video else IMAGE_SYSTEM_PROMPT

    url = f"{config.OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": config.OLLAMA_MODEL,
        "prompt": f"Raw Input Prompt: {prompt}",
        "system": system_prompt,
        "stream": False,
        "keep_alive": config.OLLAMA_KEEP_ALIVE,  # "0" = unload immediately after
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "num_predict": 300,
        }
    }

    try:
        logger.info(f"Enhancing prompt via Ollama ({config.OLLAMA_MODEL})...")
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        result = resp.json().get("response", "").strip()

        if not result:
            logger.warning("Ollama returned empty response, using original prompt")
            return prompt

        # Clean up: remove quotes, markdown artifacts
        result = result.strip('"\'')
        if result.startswith("```"):
            result = result.split("```")[1] if "```" in result[3:] else result[3:]
            result = result.strip()

        logger.info(f"Ollama enhanced prompt: {result[:100]}...")
        
        # Explicitly request Ollama to unload the model to free VRAM
        # This is critical: models like gemma3-27b use ~49GB that we need for generation
        if config.OLLAMA_KEEP_ALIVE == "0":
            try:
                requests.post(url, json={
                    "model": config.OLLAMA_MODEL,
                    "prompt": "",
                    "keep_alive": 0,
                    "stream": False,
                }, timeout=5)
                logger.info(f"Ollama model {config.OLLAMA_MODEL} unloaded to free VRAM")
            except Exception:
                pass  # Best-effort unload
        
        return result

    except requests.ConnectionError:
        logger.error(f"Cannot reach Ollama at {config.OLLAMA_BASE_URL} — is it running?")
        return prompt
    except requests.Timeout:
        logger.warning("Ollama timed out, using original prompt")
        return prompt
    except Exception as e:
        logger.error(f"Ollama enhancement failed: {e}")
        return prompt


def enhance_prompt(
    prompt: str,
    system_prompt: str | None = None,
    is_video: bool = True,
    text_encoder=None,
    image_path: str | None = None,
    seed: int = 42,
) -> str:
    """
    Unified prompt enhancement dispatcher.
    Routes to Ollama or Gemma based on config.LLM_PROVIDER.
    """
    provider = config.LLM_PROVIDER.lower()

    if provider == "ollama":
        return enhance_prompt_ollama(prompt, system_prompt, is_video)

    elif provider == "gemma":
        if text_encoder is None:
            logger.warning("Gemma provider selected but no text_encoder provided, using original prompt")
            return prompt
        # Use the LTX built-in Gemma enhancement
        from ltx_pipelines.utils.helpers import generate_enhanced_prompt
        return generate_enhanced_prompt(
            text_encoder,
            prompt,
            image_path=image_path,
            seed=seed,
            is_image=not is_video,
            system_prompt=system_prompt,
        )

    else:
        logger.warning(f"Unknown LLM provider '{provider}', using original prompt")
        return prompt
