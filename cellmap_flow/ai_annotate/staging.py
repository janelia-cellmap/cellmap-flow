"""Where an AI annotation waits for review: a staging directory per annotation.

A model's result is not written into the annotation volume until the user
accepts it. Until then it lives in ``<corrections_dir>/.ai_annotate/<id>/``:
the image sent (``input.png``), the image the model returned
(``model.png``, when it returned one), the mask on the write box's voxels
(``mask_write.npy``) and ``meta.json`` (the plan, the prompt, the model, the
label name). Accepting writes the mask and discards the directory, as
rejecting does. Nothing secret is staged: the prompt and ids are the user's
own, and no provider credential ever reaches these functions.

The id is the server's (``new_annotate_id``) and is checked against
``ANNOTATE_ID_RE`` before it is joined into any path, so an id from a
request can only ever name a directory under ``.ai_annotate``.
"""

import json
import os
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from cellmap_flow.ai_annotate.errors import AIAnnotateError

ANNOTATE_ID_RE = re.compile(r"^[0-9a-f]{32}$")
STAGING_DIRNAME = ".ai_annotate"

INPUT_PNG = "input.png"
MODEL_PNG = "model.png"
MASK_NPY = "mask_write.npy"
META_JSON = "meta.json"

# How strongly the overlay tints the mask with the target colour.
OVERLAY_ALPHA = 0.5


def new_annotate_id() -> str:
    """A fresh annotation id: 32 lowercase hex characters."""
    return uuid.uuid4().hex


def _checked(annotate_id) -> str:
    if not isinstance(annotate_id, str) or not ANNOTATE_ID_RE.fullmatch(annotate_id):
        raise AIAnnotateError("refused", "That is not an AI annotation id.", http_status=400)
    return annotate_id


def staging_dir(corrections_dir, annotate_id) -> Path:
    """``<corrections_dir>/.ai_annotate/<annotate_id>``; the id is validated first."""
    return Path(corrections_dir) / STAGING_DIRNAME / _checked(annotate_id)


def _write_atomic(path: Path, write) -> None:
    """``write(tmp_path)`` then rename over ``path``, so a reader never sees half a file."""
    tmp = path.with_name(f".{path.name}.tmp")
    write(tmp)
    os.replace(tmp, path)


def stage(
    corrections_dir,
    annotate_id,
    *,
    plan,
    request,
    result,
    mask_write,
    provider_id,
    model,
    label_name,
    volume_id=None,
) -> dict:
    """Stage a model's result for review; returns the meta written to meta.json.

    ``plan`` is the ``PlanePlan``, ``request`` and ``result`` the backend's
    ``SegmentRequest`` and ``SegmentResult``, ``mask_write`` the mask on the
    write box (``pipeline.mask_to_write_shape``). Staging the same id again
    (a resend) replaces what was there; meta.json is written last, so a
    staging with one is complete.
    """
    directory = staging_dir(corrections_dir, annotate_id)
    directory.mkdir(parents=True, exist_ok=True)
    mask_write = np.asarray(mask_write, dtype=bool)
    if mask_write.shape != tuple(plan.write_shape):
        raise ValueError(f"The mask is {mask_write.shape}, the plan's write box {tuple(plan.write_shape)}")

    _write_atomic(directory / INPUT_PNG, lambda p: request.image.convert("RGB").save(p, format="PNG"))
    model_png = directory / MODEL_PNG
    if result.model_image is not None:
        _write_atomic(model_png, lambda p: result.model_image.convert("RGB").save(p, format="PNG"))
    elif model_png.exists():
        model_png.unlink()

    def save_mask(p):
        with open(p, "wb") as f:
            np.save(f, mask_write, allow_pickle=False)

    _write_atomic(directory / MASK_NPY, save_mask)
    meta = {
        "annotate_id": annotate_id,
        "volume_id": volume_id,
        "plan": plan.to_dict(),
        "prompt": request.prompt,
        "target_rgb": [int(c) for c in request.target_rgb],
        "provider_id": provider_id,
        "model": model,
        "label_name": label_name,
        "mask_fraction": float(mask_write.mean()) if mask_write.size else 0.0,
        "usage": dict(result.usage or {}),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    _write_atomic(directory / META_JSON, lambda p: p.write_text(json.dumps(meta, indent=2, default=str)))
    return meta


def load(corrections_dir, annotate_id):
    """``(meta, mask_write, input_image)`` of a staging.

    Raises AIAnnotateError ("refused", 404) when there is none, as after an
    accept, a reject or a restart that cleared it.
    """
    directory = staging_dir(corrections_dir, annotate_id)
    meta_path = directory / META_JSON
    if not meta_path.is_file():
        raise AIAnnotateError(
            "refused", "That AI annotation is no longer staged: it was accepted or rejected.", http_status=404
        )
    meta = json.loads(meta_path.read_text())
    mask_write = np.load(directory / MASK_NPY, allow_pickle=False)
    with Image.open(directory / INPUT_PNG) as image:
        input_image = image.convert("RGB")
    return meta, mask_write, input_image


def overlay(input_image: Image.Image, mask_write: np.ndarray, target_rgb) -> Image.Image:
    """``input_image`` with the voxels the mask labels tinted ``target_rgb``.

    The mask is the write box's, scaled to the image by nearest neighbour,
    so the overlay shows what accepting writes, not the model's own image.
    """
    rgb = np.asarray(input_image.convert("RGB"), dtype=np.float32)
    mask = Image.fromarray(np.asarray(mask_write, dtype=np.uint8) * 255).resize(input_image.size, Image.NEAREST)
    on = np.asarray(mask) > 127
    tint = np.asarray(target_rgb, dtype=np.float32)
    rgb[on] = (1 - OVERLAY_ALPHA) * rgb[on] + OVERLAY_ALPHA * tint
    return Image.fromarray(np.round(rgb).astype(np.uint8))


def preview(corrections_dir, annotate_id) -> dict:
    """The review panel's images, base64 PNG: ``input_png``, ``model_png`` (None
    when the model returned no image) and ``overlay_png``.

    Every image is re-encoded here, so nothing the model sent reaches the
    browser as it came.
    """
    from cellmap_flow.ai_annotate.images import png_base64

    meta, mask_write, input_image = load(corrections_dir, annotate_id)
    model_path = staging_dir(corrections_dir, annotate_id) / MODEL_PNG
    model_png = None
    if model_path.is_file():
        with Image.open(model_path) as image:
            model_png = png_base64(image.convert("RGB"))
    return {
        "input_png": png_base64(input_image),
        "model_png": model_png,
        "overlay_png": png_base64(overlay(input_image, mask_write, meta["target_rgb"])),
    }


def discard(corrections_dir, annotate_id) -> None:
    """Delete a staging; nothing happens when there is none."""
    directory = staging_dir(corrections_dir, annotate_id)
    shutil.rmtree(directory, ignore_errors=True)
