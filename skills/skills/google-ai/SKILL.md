---
name: google-genai-media
description: >
  Full integration guide for Google's latest AI media generation services: Imagen 4 (image
  generation, editing, upscaling), Veo 3/3.1 (video generation, image-to-video, video extension),
  Lyria 2 (music generation), and Gemini multimodal image generation — all via the unified
  Google Gen AI SDK (google-genai). Use this skill whenever the user wants to generate, edit,
  or manipulate images or video using Google AI, work with Vertex AI media APIs, build creative
  content pipelines, integrate Imagen or Veo into an app, or asks about Gemini image/video
  capabilities. Always trigger for tasks involving: text-to-image, text-to-video, image-to-video,
  video extension, inpainting, outpainting, background editing, style transfer, image upscaling,
  AI music generation, or SynthID watermarking via Google services.
---

# Google GenAI Media Services — Full Integration Guide

> **Last verified:** February 2026  
> **SDK:** `google-genai` (unified, replaces legacy `google-generativeai` which EOL'd Nov 2025)  
> **Docs:** https://googleapis.github.io/python-genai/ | https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models

---

## 1. SDK Setup (Always Start Here)

### Installation

```bash
pip install google-genai            # Core SDK
pip install google-genai[aiohttp]   # For async support
pip install pillow                  # For image display/saving
```

> ⚠️ **Never use** `google-generativeai` (legacy, EOL November 30, 2025). Always use `google-genai`.

### Client Initialization

**Gemini Developer API (AI Studio — easiest for prototyping):**
```python
from google import genai

# Option A: pass key directly
client = genai.Client(api_key="GEMINI_API_KEY")

# Option B: use env var (recommended)
# export GEMINI_API_KEY=your_key
client = genai.Client()
```

**Vertex AI (enterprise, required for image editing & upscaling):**
```python
from google import genai
from google.genai.types import HttpOptions

# Option A: code-level
client = genai.Client(
    vertexai=True,
    project="your-gcp-project-id",
    location="us-central1"
)

# Option B: env vars (recommended for CI/CD)
# export GOOGLE_GENAI_USE_VERTEXAI=true
# export GOOGLE_CLOUD_PROJECT=your-project-id
# export GOOGLE_CLOUD_LOCATION=us-central1
client = genai.Client(http_options=HttpOptions(api_version="v1"))
```

**Authentication for Vertex AI:**
```bash
gcloud auth application-default login
# OR for service accounts:
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

---

## 2. Recommended Models (as of Feb 2026)

### Image Generation
| Model ID | Use Case |
|---|---|
| `imagen-4.0-generate-001` | ✅ GA — best quality text-to-image |
| `imagen-4.0-ultra-generate-001` | ✅ GA — highest fidelity, slower |
| `imagen-4.0-fast-generate-001` | ✅ GA — fast generation, simpler prompts |
| `imagen-3.0-generate-002` | Stable fallback if Imagen 4 unavailable |

### Image Editing (Vertex AI only)
| Model ID | Use Case |
|---|---|
| `imagen-3.0-capability-001` | Inpainting, outpainting, background edit |
| `imagen-product-recontext-preview-06-30` | Product image background replacement |

### Video Generation
| Model ID | Use Case |
|---|---|
| `veo-3.1-generate-preview` | ✅ Best quality, advanced controls |
| `veo-3.1-fast-generate-preview` | Fast generation, lower latency |
| `veo-3.0-generate-001` | GA stable high-fidelity |
| `veo-3.0-fast-generate-001` | GA fast video |
| `veo-2.0-generate-001` | Stable fallback; also supports style images |

### Gemini Multimodal Image Generation
| Model ID | Use Case |
|---|---|
| `gemini-2.5-flash-image` | Fast image gen+edit via chat |
| `gemini-3-pro-image-preview` | Highest quality via chat |

### Music Generation (Vertex AI)
| Model ID | Use Case |
|---|---|
| `lyria-002` | GA — text-to-music, studio-grade |

---

## 3. Image Generation with Imagen 4

### Text-to-Image (Basic)
```python
from google import genai
from google.genai import types

client = genai.Client()  # or Vertex AI client

response = client.models.generate_images(
    model="imagen-4.0-generate-001",
    prompt="A golden retriever puppy playing in autumn leaves, soft morning light",
    config=types.GenerateImagesConfig(
        number_of_images=4,           # 1–4
        aspect_ratio="16:9",          # "1:1", "9:16", "4:3", "3:4", "16:9"
        output_mime_type="image/jpeg",
        include_rai_reason=True,      # include safety filter reason if blocked
        person_generation="allow_adult",  # "allow_all", "allow_adult", "dont_allow"
        safety_filter_level="block_medium_and_above",
        enhance_prompt=True,          # auto-enhance prompt quality
    ),
)

for i, img in enumerate(response.generated_images):
    img.image.save(f"output_{i}.jpg")
    # or img.image.show() to display inline
```

### Save Image to File (PIL)
```python
from PIL import Image
from io import BytesIO
import base64

# If response returns bytes:
pil_image = Image.open(BytesIO(response.generated_images[0].image._image_bytes))
pil_image.save("output.png")
```

### Image Upscaling (Vertex AI only)
```python
# Must use Vertex AI client
response_upscaled = client.models.upscale_image(
    model="imagen-4.0-upscale-preview",
    image=response.generated_images[0].image,
    upscale_factor="x2",   # "x2" or "x4"
    config=types.UpscaleImageConfig(
        include_rai_reason=True,
        output_mime_type="image/jpeg",
    ),
)
response_upscaled.generated_images[0].image.save("upscaled.jpg")
```

---

## 4. Image Editing with Imagen (Vertex AI Only)

All editing operations require a Vertex AI client and `imagen-3.0-capability-001`.

### Inpainting — Insert into Masked Region
```python
from google.genai.types import RawReferenceImage, MaskReferenceImage

# Load source image
with open("source.jpg", "rb") as f:
    image_bytes = f.read()

raw_ref = RawReferenceImage(
    reference_id=1,
    reference_image=types.Image(image_bytes=image_bytes, mime_type="image/jpeg"),
)

mask_ref = MaskReferenceImage(
    reference_id=2,
    config=types.MaskReferenceConfig(
        mask_mode="MASK_MODE_BACKGROUND",   # auto-mask background
        mask_dilation=0,
    ),
)

response = client.models.edit_image(
    model="imagen-3.0-capability-001",
    prompt="Replace background with a tropical beach at sunset",
    reference_images=[raw_ref, mask_ref],
    config=types.EditImageConfig(
        edit_mode="EDIT_MODE_INPAINT_INSERTION",
        number_of_images=2,
        include_rai_reason=True,
        output_mime_type="image/jpeg",
    ),
)
response.generated_images[0].image.save("edited.jpg")
```

### Mask Modes
| `mask_mode` | Behavior |
|---|---|
| `MASK_MODE_BACKGROUND` | Auto-mask the background |
| `MASK_MODE_FOREGROUND` | Auto-mask the foreground subject |
| `MASK_MODE_SEMANTIC` | Mask by semantic segment (e.g., "sky") |
| `MASK_MODE_USER_PROVIDED` | Use your own binary mask image |

### Edit Modes
| `edit_mode` | Behavior |
|---|---|
| `EDIT_MODE_INPAINT_INSERTION` | Generate content inside the mask |
| `EDIT_MODE_INPAINT_REMOVAL` | Remove masked content, fill naturally |
| `EDIT_MODE_OUTPAINT` | Extend image beyond borders |
| `EDIT_MODE_BGSWAP` | Full background replacement |
| `EDIT_MODE_PRODUCT_IMAGE` | Product recontext (background scenes) |

---

## 5. Video Generation with Veo

> ⚠️ **Cost warning:** Veo is significantly more expensive than image generation. Always check current pricing at https://cloud.google.com/vertex-ai/generative-ai/pricing before running in production.

Video generation is **asynchronous** — submit a job, then poll until complete.

### Text-to-Video (Basic)
```python
import time
from google import genai
from google.genai import types

client = genai.Client(vertexai=True, project="your-project", location="us-central1")

# Submit generation job
operation = client.models.generate_videos(
    model="veo-3.0-generate-001",
    prompt="A time-lapse of a city skyline from dawn to dusk, cinematic 4K quality",
    config=types.GenerateVideosConfig(
        number_of_videos=1,
        duration_seconds=8,        # 5–8 seconds typically
        aspect_ratio="16:9",       # "16:9" or "9:16"
        resolution="1080p",        # "720p" or "1080p" (Veo 3 preview)
        enhance_prompt=True,
        output_gcs_uri="gs://your-bucket/video-outputs/",  # optional GCS output
    ),
)

# Poll until done
while not operation.done:
    time.sleep(20)
    operation = client.operations.get(operation)

# Access result
video = operation.response.generated_videos[0].video
video.save("output.mp4")
```

### Image-to-Video
```python
import base64

with open("start_frame.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

operation = client.models.generate_videos(
    model="veo-3.1-generate-preview",
    prompt="The cat slowly turns its head and yawns",
    image=types.Image(
        image_bytes=base64.b64decode(image_b64),
        mime_type="image/jpeg",
    ),
    config=types.GenerateVideosConfig(
        number_of_videos=1,
        duration_seconds=6,
        aspect_ratio="16:9",
    ),
)

while not operation.done:
    time.sleep(20)
    operation = client.operations.get(operation)

operation.response.generated_videos[0].video.save("animated.mp4")
```

### Video Extension (Veo 2 — Vertex AI)
```python
# Extend an existing video by providing it as input
video_input = types.Video(uri="gs://your-bucket/existing-video.mp4")

operation = client.models.generate_videos(
    model="veo-2.0-generate-001",
    prompt="Continue the scene as the character walks into the building",
    video=video_input,
    config=types.GenerateVideosConfig(
        number_of_videos=1,
        duration_seconds=5,
    ),
)
```

### Advanced Video Controls (Veo 3.1)
```python
# First frame + last frame control
operation = client.models.generate_videos(
    model="veo-3.1-generate-preview",
    prompt="Camera pushes through the forest canopy",
    config=types.GenerateVideosConfig(
        number_of_videos=1,
        duration_seconds=8,
        # Reference images for first/last frame control
        reference_images=[
            types.ReferenceImage(
                reference_id=1,
                reference_image=first_frame_image,
                config=types.ReferenceImageConfig(
                    reference_type="REFERENCE_TYPE_FIRST_FRAME"
                ),
            ),
            types.ReferenceImage(
                reference_id=2,
                reference_image=last_frame_image,
                config=types.ReferenceImageConfig(
                    reference_type="REFERENCE_TYPE_LAST_FRAME"
                ),
            ),
        ],
    ),
)
```

> **Note:** Style images with `referenceImages.style` require `veo-2.0-generate-exp` — Veo 3.1 does NOT support style reference images.

---

## 6. Gemini Multimodal Image Generation (Chat-based)

Gemini image models allow iterative edit-in-conversation workflows.

```python
from google import genai
from PIL import Image
from io import BytesIO

client = genai.Client()

# Create a chat session for iterative editing
chat = client.chats.create(model="gemini-2.5-flash-image")

# Generate initial image
response = chat.send_message(
    "Create a photorealistic image of a cozy coffee shop interior at night, warm lighting"
)

# Parse response parts (text + image)
for part in response.candidates[0].content.parts:
    if part.text:
        print(part.text)
    elif part.inline_data:
        img = part.as_image()
        img.save("coffee_shop.png")

# Iteratively edit
response2 = chat.send_message("Add a cat sleeping on one of the chairs")
for part in response2.candidates[0].content.parts:
    if part.inline_data:
        part.as_image().save("coffee_shop_v2.png")
```

---

## 7. Async Usage

```python
import asyncio
from google import genai
from google.genai import types

async def generate_async():
    client = genai.Client()  # uses aiohttp if installed
    
    response = await client.aio.models.generate_images(
        model="imagen-4.0-generate-001",
        prompt="A futuristic city at dusk",
        config=types.GenerateImagesConfig(number_of_images=2),
    )
    return response.generated_images

asyncio.run(generate_async())
```

---

## 8. GCS Output Pattern (Production)

For production workflows, always write outputs to GCS rather than returning bytes:

```python
operation = client.models.generate_videos(
    model="veo-3.0-generate-001",
    prompt="...",
    config=types.GenerateVideosConfig(
        output_gcs_uri="gs://your-bucket/outputs/",
        number_of_videos=2,
    ),
)

# Response contains GCS URIs
for video in operation.response.generated_videos:
    print(video.video.uri)  # e.g., gs://your-bucket/outputs/sample_0.mp4
```

---

## 9. Safety & Content Filters

All Google media models include automatic SynthID digital watermarking in every output.

### Safety Filter Levels (Imagen)
```python
config=types.GenerateImagesConfig(
    safety_filter_level="block_medium_and_above",  # default
    # Options: "block_low_and_above" (strictest), "block_medium_and_above", "block_only_high"
    person_generation="allow_adult",
    # Options: "allow_all", "allow_adult", "dont_allow"
)
```

### Checking RAI Refusals
```python
for img in response.generated_images:
    if img.rai_filtered_reason:
        print(f"Blocked: {img.rai_filtered_reason}")
    else:
        img.image.save("output.jpg")
```

---

## 10. Error Handling

```python
from google.genai.errors import APIError, ClientError

try:
    response = client.models.generate_images(
        model="imagen-4.0-generate-001",
        prompt=user_prompt,
        config=types.GenerateImagesConfig(number_of_images=1),
    )
except ClientError as e:
    print(f"Client error (bad request): {e.status_code} — {e.message}")
except APIError as e:
    print(f"API error: {e.status_code} — {e.message}")
```

---

## 11. Quick-Reference Cheatsheet

| Task | Model | API |
|---|---|---|
| Text → Image (quality) | `imagen-4.0-generate-001` | `client.models.generate_images()` |
| Text → Image (fast) | `imagen-4.0-fast-generate-001` | `client.models.generate_images()` |
| Upscale image 2x/4x | `imagen-4.0-upscale-preview` | `client.models.upscale_image()` |
| Inpaint / background swap | `imagen-3.0-capability-001` | `client.models.edit_image()` |
| Text → Video (quality) | `veo-3.0-generate-001` | `client.models.generate_videos()` |
| Text → Video (fast) | `veo-3.0-fast-generate-001` | `client.models.generate_videos()` |
| Image → Video | `veo-3.1-generate-preview` | `client.models.generate_videos()` |
| Video extension | `veo-2.0-generate-001` | `client.models.generate_videos()` |
| Iterative image chat | `gemini-2.5-flash-image` | `client.chats.create()` |
| Text → Music | `lyria-002` (Vertex only) | Vertex AI Studio / REST API |

---

## 12. Key Gotchas

- **Image editing and upscaling require Vertex AI** — they are not available on the Gemini Developer API (AI Studio).
- **Video generation is async** — always poll with `client.operations.get()` in a loop.
- **Veo 3.1 does not support style reference images** — use `veo-2.0-generate-exp` for that feature.
- **`imagen-4.0-fast-generate-001` may produce poor results on complex prompts** — set `enhance_prompt=False` or switch to the standard model.
- **Do not use `google-generativeai`** — it reached end-of-life November 30, 2025.
- **Veo is expensive** — always confirm pricing before running bulk jobs: https://cloud.google.com/vertex-ai/generative-ai/pricing
- **GCS output is recommended for video** — returning video bytes inline can cause timeouts for long generations.
- **Supported video durations:** typically 5–8 seconds; check per model documentation.
- **Location matters:** Most Imagen/Veo models are only available in `us-central1`.

---

## 13. References

- SDK Docs: https://googleapis.github.io/python-genai/
- Vertex AI Model List: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models
- Imagen API Reference: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/model-reference/imagen-api
- Veo API Reference: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/model-reference/veo-video-generation
- Release Notes: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/release-notes
- Pricing: https://cloud.google.com/vertex-ai/generative-ai/pricing
