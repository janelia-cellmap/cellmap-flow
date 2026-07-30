"""Vertex-AI Gemini image recoloring call, ported from ask-to-mask's
GeminiImageBackend (agents/gen_backend.py) and trimmed to a single function:
no API-key path, no Imagen path, no multi-backend registry - just the
Vertex/ADC call this feature needs.
"""

from __future__ import annotations

import io
import os
import time

from PIL import Image


def _upsample_for_edit(image: Image.Image, floor: int = 1024) -> Image.Image:
    """Lanczos-upsample small crops before sending, so the model's own
    uncontrolled internal resize doesn't coarsen detail first. The result is
    resized back to the caller's original size regardless.
    """
    w, h = image.size
    scale = max(1.0, floor / min(w, h))
    if scale == 1.0:
        return image
    return image.resize((round(w * scale), round(h * scale)), Image.LANCZOS)


def _make_vertex_client(project: str | None, location: str):
    from google import genai

    project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise ValueError(
            "Vertex AI requires a GCP project. Set GOOGLE_CLOUD_PROJECT env var "
            "or pass vertex_project explicitly."
        )
    return genai.Client(vertexai=True, project=project, location=location)


def _generate_gemini(client, image: Image.Image, prompt: str, model: str) -> Image.Image:
    from google import genai

    response = client.models.generate_content(
        model=model,
        contents=[image, prompt],
        config=genai.types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
        ),
    )

    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            return Image.open(io.BytesIO(part.inline_data.data)).convert("RGB")

    raise RuntimeError(
        "Gemini returned no image. Response text: "
        f"{response.text[:500] if response.text else '(empty)'}"
    )


def generate_recolored_image(
    image: Image.Image,
    prompt: str,
    model: str = "gemini-3-pro-image",
    vertex_project: str | None = None,
    vertex_location: str = "global",
) -> Image.Image:
    """Send an EM crop + recolor prompt to Gemini via Vertex AI, return the result.

    location must be "global" (not e.g. "us-central1") for native image-gen
    Gemini models - confirmed in ask-to-mask's HANDOFF.md / cli.py override.
    """
    from google.genai.errors import ClientError

    client = _make_vertex_client(vertex_project, vertex_location)
    send_image = _upsample_for_edit(image)

    generated_image = None
    for attempt in range(3):
        try:
            generated_image = _generate_gemini(client, send_image, prompt, model)
            break
        except ClientError as e:
            if "429" in str(e) and attempt < 2:
                wait = 30 * (attempt + 1)
                time.sleep(wait)
            else:
                raise

    if generated_image.size != image.size:
        generated_image = generated_image.resize(image.size, Image.LANCZOS)

    return generated_image
