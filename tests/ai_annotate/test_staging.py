"""Staging an AI annotation for review: writing, reading, previewing, discarding."""

import base64
import io
import json

import numpy as np
import pytest
from PIL import Image

from cellmap_flow.ai_annotate import staging
from cellmap_flow.ai_annotate.backends.base import SegmentRequest, SegmentResult
from cellmap_flow.ai_annotate.errors import AIAnnotateError
from cellmap_flow.ai_annotate.geometry import plan_plane

RED = (255, 0, 0)


def _staged_parts(model_image=True):
    volume = {"output_voxel_size": [8.0] * 3, "input_voxel_size": [4.0] * 3, "dataset_offset_nm": [4.0] * 3}
    plan = plan_plane(volume, (24, 32, 40), (81, 113, 177), 0, crop_size_px=8)
    image = Image.fromarray(np.full((*plan.image_shape, 3), 100, dtype=np.uint8))
    request = SegmentRequest(image=image, prompt="Paint the mitochondria red.", target_rgb=RED, click_px=plan.click_px)
    mask = np.zeros(plan.image_shape, dtype=bool)
    mask[:8, :8] = True
    result = SegmentResult(
        mask=mask,
        model_image=Image.fromarray(np.zeros((*plan.image_shape, 3), dtype=np.uint8)) if model_image else None,
        model="fake-threshold",
        usage={"prompt_tokens": 560},
    )
    mask_write = np.zeros(plan.write_shape, dtype=bool)
    mask_write[:4, :4] = True
    return plan, request, result, mask_write


def _stage(corrections_dir, annotate_id, model_image=True):
    plan, request, result, mask_write = _staged_parts(model_image)
    return staging.stage(
        corrections_dir, annotate_id, plan=plan, request=request, result=result, mask_write=mask_write,
        provider_id="fake", model="fake-threshold", label_name="mito", volume_id="vol-1",
    ), plan, mask_write


def _decode(b64):
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"))


def test_new_ids_are_valid_and_distinct():
    ids = {staging.new_annotate_id() for _ in range(100)}
    assert len(ids) == 100
    assert all(staging.ANNOTATE_ID_RE.fullmatch(i) for i in ids)


def test_stage_load_preview_discard(tmp_path):
    annotate_id = staging.new_annotate_id()
    meta, plan, mask_write = _stage(tmp_path, annotate_id)

    directory = tmp_path / ".ai_annotate" / annotate_id
    assert sorted(p.name for p in directory.iterdir()) == ["input.png", "mask_write.npy", "meta.json", "model.png"]
    assert json.loads((directory / "meta.json").read_text()) == meta
    assert set(meta) == {
        "annotate_id", "volume_id", "plan", "prompt", "target_rgb", "provider_id", "model",
        "label_name", "mask_fraction", "usage", "created_at",
    }
    assert meta["annotate_id"] == annotate_id and meta["volume_id"] == "vol-1"
    assert meta["mask_fraction"] == pytest.approx(16 / 64)
    assert meta["target_rgb"] == list(RED)

    loaded_meta, loaded_mask, input_image = staging.load(tmp_path, annotate_id)
    assert loaded_meta == meta
    assert np.array_equal(loaded_mask, mask_write) and loaded_mask.dtype == bool
    assert input_image.size == (plan.image_shape[1], plan.image_shape[0])
    from cellmap_flow.ai_annotate.geometry import PlanePlan

    assert PlanePlan.from_dict(loaded_meta["plan"]) == plan

    shown = staging.preview(tmp_path, annotate_id)
    assert set(shown) == {"input_png", "model_png", "overlay_png"}
    assert np.all(_decode(shown["input_png"]) == 100)
    assert np.all(_decode(shown["model_png"]) == 0)
    overlay = _decode(shown["overlay_png"])
    # The mask's voxels (the first 4 x 4 of 8 x 8) cover the first 8 x 8 pixels.
    assert tuple(overlay[0, 0]) == (178, 50, 50)
    assert tuple(overlay[7, 7]) == (178, 50, 50)
    assert tuple(overlay[8, 8]) == (100, 100, 100)

    staging.discard(tmp_path, annotate_id)
    assert not directory.exists()
    staging.discard(tmp_path, annotate_id)  # twice is fine
    with pytest.raises(AIAnnotateError) as caught:
        staging.load(tmp_path, annotate_id)
    assert caught.value.http_status == 404


def test_restaging_replaces_and_drops_a_stale_model_image(tmp_path):
    annotate_id = staging.new_annotate_id()
    _stage(tmp_path, annotate_id)
    _stage(tmp_path, annotate_id, model_image=False)
    assert not (tmp_path / ".ai_annotate" / annotate_id / "model.png").exists()
    assert staging.preview(tmp_path, annotate_id)["model_png"] is None


def test_a_mask_off_the_plan_is_refused(tmp_path):
    plan, request, result, _ = _staged_parts()
    with pytest.raises(ValueError):
        staging.stage(tmp_path, staging.new_annotate_id(), plan=plan, request=request, result=result,
                      mask_write=np.zeros((3, 3), dtype=bool), provider_id="fake", model="m", label_name="x")


@pytest.mark.parametrize(
    "bad_id",
    [
        "../x",
        "../" + "0" * 29,
        "0" * 31,
        "0" * 33,
        "A" * 32,
        "0" * 31 + "/",
        "/etc/passwd",
        "0" * 32 + "\n",
        "",
        None,
        32,
    ],
)
def test_ids_that_are_not_ours_are_refused_before_any_path(tmp_path, bad_id):
    for call in (
        lambda: staging.staging_dir(tmp_path, bad_id),
        lambda: staging.load(tmp_path, bad_id),
        lambda: staging.preview(tmp_path, bad_id),
        lambda: staging.discard(tmp_path, bad_id),
        lambda: _stage(tmp_path, bad_id),
    ):
        with pytest.raises(AIAnnotateError) as caught:
            call()
        assert caught.value.category == "refused" and caught.value.http_status == 400
    assert list(tmp_path.iterdir()) == []


def test_discard_never_leaves_the_staging_directory(tmp_path):
    # A sibling the traversal id would name survives a refused discard.
    (tmp_path / "x").mkdir()
    with pytest.raises(AIAnnotateError):
        staging.discard(tmp_path / ".ai_annotate", "../x")
    assert (tmp_path / "x").is_dir()
