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

def get_video_system_prompt(duration_seconds: float, has_input_image: bool = False) -> str:
    """Generate a dynamic system prompt scaled to the target video duration with JSON output."""
    if duration_seconds <= 6.0:
        length_guideline = "- Formatting: Single concise paragraph of 2-3 sentences max. Focus on a SINGLE continuous physical action to prevent scene complexity overload."
    elif duration_seconds <= 12.0:
        length_guideline = "- Formatting: Single flowing paragraph of 3-5 sentences. Maintain a steady physical pace without rushing the action."
    else:
        length_guideline = "- Formatting: Single flowing paragraph of 4-8 sentences. Describe detailed physical scene progression suitable for a longer video."
        
    i2v_guideline = ""
    if has_input_image:
        i2v_guideline = (
            "CRITICAL I2V INSTRUCTIONS:\n"
            "- You are generating a video FROM AN EXISTING IMAGE.\n"
            "- DO NOT hallucinate new clothing, backgrounds, or subjects not present in the image.\n"
            "- Analyze the image: Is it a Character Sheet, Turnaround, Collage, or Diagram? If yes, set `is_reference_only` to true, and thoroughly extract the character's physical details into the prompt (hair, clothing, face). Do not attempt to animate the character sheet itself.\n"
            "- If it is a natural scene or photograph, set `is_reference_only` to false, and focus on physical action happening to the subjects in the scene.\n"
        )

    return (
        "You are an expert Vision-Language Master tasked with writing highly structured, action-focused video prompts. "
        "Given a user's Raw Input Prompt (and potentially an image), generate a rich, accurate prompt for video diffusion.\n\n"
        "#### Guidelines:\n"
        "- Subject Motion ALWAYS Trumps Camera Motion: Explicitly describe what the character/subject is physically doing (e.g., walking, talking). Avoid prompts where the subjects are frozen and only the camera 'pans' or 'zooms'.\n"
        "- Active language: Use present-progressive verbs ('is walking,' 'speaking').\n"
        "- Chronological flow: Use temporal connectors ('as,' 'then,' 'while').\n"
        "- Audio/Speech: Describe any sounds, speech, or music occurring naturally. Provide exact words in quotes if requested.\n"
        "- CRITICAL: PRESERVE all specific details about Characters, Locations, and Settings provided in the input. Do not summarize them away.\n"
        "- Style: Include visual style at beginning: 'Style: <style>.' If unclear, omit.\n"
        f"{length_guideline}\n\n"
        f"{i2v_guideline}"
        "#### Output Format:\n"
        "You MUST output raw JSON without markdown formatting. Return this exact structure:\n"
        "{\n"
        '  "enhanced_prompt": "Your beautifully detailed, action-focused prompt string here",\n'
        '  "is_reference_only": false\n'
        "}\n"
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
    duration_seconds: float = 4.8,
    has_input_image: bool = False,
) -> tuple[str, bool]:
    """Enhance a prompt using Ollama's local LLM API."""
    if not system_prompt:
        system_prompt = get_video_system_prompt(duration_seconds, has_input_image) if is_video else IMAGE_SYSTEM_PROMPT

    url = f"{config.OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": config.OLLAMA_MODEL,
        "prompt": f"Raw Input Prompt: {prompt}",
        "system": system_prompt,
        "format": "json",
        "stream": False,
        "keep_alive": config.OLLAMA_KEEP_ALIVE,  # "0" = unload immediately after
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "num_predict": 400,
        }
    }

    try:
        logger.info(f"Enhancing prompt via Ollama ({config.OLLAMA_MODEL})...")
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        result_str = resp.json().get("response", "").strip()

        if not result_str:
            logger.warning("Ollama returned empty response, using original prompt")
            return prompt, False
            
        import json, re
        match = re.search(r'\{.*\}', result_str, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                enhanced_prompt = data.get("enhanced_prompt", prompt)
                is_reference_only = data.get("is_reference_only", False)
                logger.info(f"Ollama structured JSON response parsed. Reference Only: {is_reference_only}")
                return enhanced_prompt, is_reference_only
            except Exception as e:
                logger.warning(f"Failed to parse Ollama JSON: {e}")

        logger.info(f"Ollama fallback parsing string: {result_str[:100]}...")
        # Explicitly request Ollama to unload the model to free VRAM
        if config.OLLAMA_KEEP_ALIVE == "0":
            try:
                requests.post(url, json={
                    "model": config.OLLAMA_MODEL,
                    "prompt": "",
                    "keep_alive": 0,
                    "stream": False,
                }, timeout=5)
            except Exception:
                pass  # Best-effort unload
        
        return result_str, False

    except requests.ConnectionError:
        logger.error(f"Cannot reach Ollama at {config.OLLAMA_BASE_URL} — is it running?")
        return prompt, False
    except requests.Timeout:
        logger.warning("Ollama timed out, using original prompt")
        return prompt, False
    except Exception as e:
        logger.error(f"Ollama enhancement failed: {e}")
        return prompt, False


def enhance_prompt(
    prompt: str,
    system_prompt: str | None = None,
    is_video: bool = True,
    text_encoder=None,
    image_path: str | None = None,
    seed: int = 42,
    duration_seconds: float = 4.8,
    has_input_image: bool = False,
) -> tuple[str, bool]:
    """
    Unified prompt enhancement dispatcher.
    Routes to Ollama or Gemma based on config.LLM_PROVIDER.
    Returns (enhanced_prompt_string, is_reference_only_boolean)
    """
    provider = config.LLM_PROVIDER.lower()

    if not system_prompt:
        system_prompt = get_video_system_prompt(duration_seconds, has_input_image) if is_video else IMAGE_SYSTEM_PROMPT

    if provider == "ollama":
        return enhance_prompt_ollama(prompt, system_prompt, is_video, duration_seconds, has_input_image)

    elif provider == "gemma":
        if text_encoder is None:
            logger.warning("Gemma provider selected but no text_encoder provided, using original prompt")
            return prompt, False
            
        import torch
        from ltx_pipelines.utils.media_io import decode_image, resize_aspect_ratio_preserving
        import json
        import re

        try:
            if image_path and has_input_image:
                logger.info(f"Loading image for Gemma VLM: {image_path}")
                encoded_image = decode_image(image_path=image_path)
                encoded_image = torch.tensor(encoded_image)
                encoded_image = resize_aspect_ratio_preserving(encoded_image, 896).to(torch.uint8)
                result_str = text_encoder.enhance_i2v(prompt, encoded_image, seed=seed, system_prompt=system_prompt)
            else:
                result_str = text_encoder.enhance_t2v(prompt, seed=seed, system_prompt=system_prompt)

            # Parse JSON
            result_str = result_str.strip()
            match = re.search(r'\{.*\}', result_str, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                    return data.get("enhanced_prompt", prompt), data.get("is_reference_only", False)
                except Exception as e:
                    logger.warning(f"Failed to parse Gemma JSON: {e}")
            
            # Fallback
            result_str = result_str.replace('```json', '').replace('```', '')
            return result_str, False
            
        except Exception as e:
            logger.error(f"Gemma VLM extraction failed: {e}")
            return prompt, False

    else:
        logger.warning(f"Unknown LLM provider '{provider}', using original prompt")
        return prompt, False
