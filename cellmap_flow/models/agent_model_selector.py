"""
Agent-based model selection using Gemini vision models.

Extracts sample slices from EM datasets and uses a vision-capable LLM
to recommend which model(s) from the catalog should be used for inference.
"""

import io
import os
import json
import logging
from typing import Dict, List, Optional, Any

import numpy as np

logger = logging.getLogger(__name__)

# Default Gemini models
GEMINI_FLASH = "gemini-2.5-flash-preview-05-20"
GEMINI_PRO = "gemini-2.5-pro-preview-05-06"

SYSTEM_PROMPT = """\
You are an expert in electron microscopy (EM) image analysis, specifically \
focused on cellular ultrastructure segmentation using FIB-SEM and connectomics data.

You will be shown one or more 2D slices from a 3D EM volume dataset. Your task \
is to analyze the image content and recommend which segmentation model(s) from \
the available catalog should be applied.

When analyzing the image, consider:
1. What organelles or structures are clearly visible (mitochondria, ER, nucleus, \
vesicles, plasma membrane, lipid droplets, peroxisomes, etc.)
2. The approximate resolution/quality of the data
3. Whether the tissue appears to be from a specific organism (fly, mouse, etc.)
4. Which model(s) would produce the best segmentation results

IMPORTANT: You MUST respond with valid JSON only. No markdown, no explanation outside JSON.
Respond with this exact JSON structure:
{
    "recommended_models": ["model_key_1", "model_key_2"],
    "reasoning": "Brief explanation of why these models were selected",
    "detected_structures": ["structure1", "structure2"],
    "confidence": 0.85
}

Where "recommended_models" contains the exact model keys from the catalog, \
"confidence" is between 0.0 and 1.0.
"""


def _extract_sample_slices(
    dataset_path: str,
    num_slices: int = 3,
    target_size: int = 512,
) -> List[bytes]:
    """
    Extract 2D sample slices from the dataset and encode as PNG bytes.

    Args:
        dataset_path: Path to the zarr/n5 dataset.
        num_slices: Number of slices to extract (spread across z-axis).
        target_size: Target size for downsampling large images.

    Returns:
        List of PNG-encoded image bytes.
    """
    from cellmap_flow.image_data_interface import ImageDataInterface
    from funlib.geometry import Roi, Coordinate
    from PIL import Image

    idi = ImageDataInterface(dataset_path, normalize=False)
    roi = idi.roi
    voxel_size = idi.voxel_size

    # Calculate slice positions spread across z-axis
    z_start = roi.offset[0]
    z_end = roi.offset[0] + roi.shape[0]
    z_positions = np.linspace(
        z_start + roi.shape[0] * 0.25,
        z_start + roi.shape[0] * 0.75,
        num_slices,
    ).astype(int)

    # Calculate a reasonable xy crop (center crop)
    y_center = roi.offset[1] + roi.shape[1] // 2
    x_center = roi.offset[2] + roi.shape[2] // 2
    crop_size_world = target_size * voxel_size[1]
    half_crop = crop_size_world // 2

    slices: List[bytes] = []
    for z_pos in z_positions:
        try:
            # Align to voxel grid
            z_aligned = (z_pos // voxel_size[0]) * voxel_size[0]
            y_start = ((y_center - half_crop) // voxel_size[1]) * voxel_size[1]
            x_start = ((x_center - half_crop) // voxel_size[2]) * voxel_size[2]

            slice_roi = Roi(
                offset=Coordinate(z_aligned, y_start, x_start),
                shape=Coordinate(voxel_size[0], crop_size_world, crop_size_world),
            )
            # Intersect with actual ROI to avoid out-of-bounds
            slice_roi = slice_roi.intersect(roi)
            if slice_roi.empty:
                continue

            data = idi.to_ndarray_ts(slice_roi)
            # Squeeze z dimension
            if data.ndim == 3:
                data = data[0]
            elif data.ndim > 3:
                data = data[0, ..., 0] if data.shape[-1] < data.shape[0] else data[0]

            # Normalize to uint8 for visualization
            if data.dtype != np.uint8:
                d_min, d_max = float(data.min()), float(data.max())
                if d_max > d_min:
                    data = ((data - d_min) / (d_max - d_min) * 255).astype(np.uint8)
                else:
                    data = np.zeros_like(data, dtype=np.uint8)

            # Resize if too large
            img = Image.fromarray(data)
            if max(img.size) > target_size:
                img.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)

            buf = io.BytesIO()
            img.save(buf, format="PNG")
            slices.append(buf.getvalue())
            logger.info(
                f"Extracted slice at z={z_pos}, shape={data.shape}, "
                f"roi={slice_roi}"
            )
        except Exception as exc:
            logger.warning(f"Failed to extract slice at z={z_pos}: {exc}")

    if not slices:
        raise RuntimeError(
            f"Could not extract any slices from dataset: {dataset_path}"
        )
    return slices


def _build_model_catalog_description(model_catalog: Dict[str, Any]) -> str:
    """
    Build a human-readable description of all available models.

    Args:
        model_catalog: The model catalog dict from globals (group -> model -> path).

    Returns:
        Formatted string describing each model.
    """
    from cellmap_flow.models.cellmap_models import CellmapModel

    descriptions: List[str] = []
    for group_name, models in model_catalog.items():
        if not isinstance(models, dict):
            continue
        for model_name, model_path in models.items():
            model_key = f"{group_name}/{model_name}"
            desc = f"- **{model_key}**"

            # Try to load metadata for richer descriptions
            metadata_path = os.path.join(model_path, "metadata.json")
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, "r") as f:
                        meta = json.load(f)
                    channels = meta.get("channels_names", [])
                    description = meta.get("description", "")
                    voxel = meta.get("input_voxel_size", [])
                    out_channels = meta.get("out_channels", "?")
                    desc += (
                        f"\n  Channels: {channels}"
                        f"\n  Description: {description}"
                        f"\n  Input voxel size: {voxel}"
                        f"\n  Output channels: {out_channels}"
                    )
                except Exception:
                    pass
            else:
                desc += f"\n  Path: {model_path}"

            descriptions.append(desc)

    if not descriptions:
        return "No models available in catalog."
    return "Available models:\n" + "\n".join(descriptions)


def select_models_with_agent(
    dataset_path: str,
    model_catalog: Optional[Dict[str, Any]] = None,
    gemini_model: str = GEMINI_FLASH,
    api_key: Optional[str] = None,
    num_slices: int = 3,
    custom_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Use a Gemini vision model to analyze EM data and recommend models.

    Args:
        dataset_path: Path to the zarr/n5 EM dataset.
        model_catalog: Dict of available models. If None, loads from globals.
        gemini_model: Gemini model to use (default: gemini-2.5-flash).
        api_key: Google AI API key. Falls back to GOOGLE_API_KEY env var.
        num_slices: Number of sample slices to extract.
        custom_prompt: Optional custom prompt to append.

    Returns:
        Dict with keys: recommended_models, reasoning, detected_structures, confidence.

    Raises:
        ImportError: If google-genai is not installed.
        RuntimeError: If the agent fails to produce a valid response.
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise ImportError(
            "google-genai package is required for agent model selection. "
            "Install it with: pip install google-genai"
        ) from exc

    # Resolve API key
    resolved_key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not resolved_key:
        raise ValueError(
            "Google AI API key required. Set GOOGLE_API_KEY environment variable "
            "or pass api_key parameter."
        )

    # Load model catalog if not provided
    if model_catalog is None:
        from cellmap_flow.globals import Flow

        g = Flow()
        model_catalog = g.model_catalog

    # Build catalog description
    catalog_desc = _build_model_catalog_description(model_catalog)

    # Extract sample slices
    logger.info(f"Extracting {num_slices} sample slices from {dataset_path}...")
    image_bytes_list = _extract_sample_slices(
        dataset_path, num_slices=num_slices
    )
    logger.info(f"Extracted {len(image_bytes_list)} slices successfully.")

    # Build the prompt
    user_prompt = (
        f"Here are {len(image_bytes_list)} sample slices from an EM dataset "
        f"located at: {dataset_path}\n\n"
        f"{catalog_desc}\n\n"
        "Analyze these images and recommend which model(s) to use. "
        "Return ONLY valid JSON."
    )
    if custom_prompt:
        user_prompt += f"\n\nAdditional context: {custom_prompt}"

    # Build content parts: images first, then text
    contents: list = []
    for img_bytes in image_bytes_list:
        contents.append(
            types.Part.from_bytes(data=img_bytes, mime_type="image/png")
        )
    contents.append(user_prompt)

    # Call Gemini
    client = genai.Client(api_key=resolved_key)
    logger.info(f"Sending {len(image_bytes_list)} images to {gemini_model}...")

    response = client.models.generate_content(
        model=gemini_model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            temperature=0.1,
        ),
    )

    # Parse the response
    response_text = response.text.strip()
    logger.info(f"Agent response: {response_text}")

    try:
        result = json.loads(response_text)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown fences
        import re

        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(1))
        else:
            raise RuntimeError(
                f"Agent returned invalid JSON response: {response_text}"
            )

    # Validate required fields
    if "recommended_models" not in result:
        raise RuntimeError(
            f"Agent response missing 'recommended_models': {result}"
        )

    logger.info(
        f"Agent recommended: {result['recommended_models']} "
        f"(confidence: {result.get('confidence', 'N/A')})"
    )
    logger.info(f"Reasoning: {result.get('reasoning', 'N/A')}")

    return result


def build_model_configs_from_agent(
    dataset_path: str,
    agent_result: Dict[str, Any],
    model_catalog: Optional[Dict[str, Any]] = None,
) -> list:
    """
    Convert agent recommendations into ModelConfig instances.

    Args:
        dataset_path: Path to the dataset.
        agent_result: Result dict from select_models_with_agent().
        model_catalog: Model catalog dict. If None, loads from globals.

    Returns:
        List of instantiated ModelConfig objects.
    """
    from cellmap_flow.models.models_config import CellMapModelConfig

    if model_catalog is None:
        from cellmap_flow.globals import Flow

        g = Flow()
        model_catalog = g.model_catalog

    recommended = agent_result.get("recommended_models", [])
    configs = []

    for model_key in recommended:
        # model_key format: "group/model_name"
        parts = model_key.split("/", 1)
        if len(parts) == 2:
            group, model_name = parts
            if group in model_catalog and model_name in model_catalog[group]:
                model_path = model_catalog[group][model_name]
                config = CellMapModelConfig(
                    folder_path=model_path,
                    name=model_name,
                )
                configs.append(config)
                logger.info(f"Created CellMapModelConfig for: {model_key}")
            else:
                logger.warning(
                    f"Agent recommended '{model_key}' but it's not in the catalog. "
                    f"Available: {list(model_catalog.keys())}"
                )
        else:
            # Try to find the model by name across all groups
            found = False
            for group_name, models in model_catalog.items():
                if isinstance(models, dict) and model_key in models:
                    model_path = models[model_key]
                    config = CellMapModelConfig(
                        folder_path=model_path,
                        name=model_key,
                    )
                    configs.append(config)
                    logger.info(
                        f"Created CellMapModelConfig for: {group_name}/{model_key}"
                    )
                    found = True
                    break
            if not found:
                logger.warning(
                    f"Agent recommended '{model_key}' but could not find it "
                    f"in any catalog group."
                )

    if not configs:
        logger.error(
            f"No valid models found from agent recommendations: {recommended}"
        )

    return configs
