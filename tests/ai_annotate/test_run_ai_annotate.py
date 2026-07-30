"""Tests for run_ai_annotate's click -> context-crop -> destination-chunk
geometry, with ImageDataInterface and Gemini mocked out. This exercises the
input-voxel <-> output-voxel coordinate math directly (a prior draft had a
unit-mixing bug here that these tests would have caught).
"""

import json
import os

import numpy as np
from PIL import Image

import cellmap_flow.dashboard.routes.finetune.ai_annotate as ai_annotate


class _FakeIDI:
    def __init__(self, raw_crop):
        self._raw_crop = raw_crop

    def to_ndarray_ts(self, roi):
        return self._raw_crop


def _patch_common(monkeypatch, raw_crop, recolored_image, volume_meta):
    import cellmap_flow.image_data_interface as idi_module

    # Real generate_recolored_image always resizes its result back to the
    # input image's size (see gemini_backend.py) -- mimic that contract here
    # so extract_mask's output stays aligned with input_image/mask_for_preview.
    def _fake_generate(image, *a, **kw):
        return recolored_image.resize(image.size)

    monkeypatch.setattr(idi_module, "ImageDataInterface", lambda *a, **kw: _FakeIDI(raw_crop))
    monkeypatch.setattr(ai_annotate, "generate_recolored_image", _fake_generate)
    monkeypatch.setattr(ai_annotate, "_get_volume_metadata", lambda volume_id: volume_meta)


def test_run_ai_annotate_downsamples_mask_to_output_chunk_resolution(monkeypatch, tmp_path):
    # Output voxel size is 2x coarser than input voxel size, so the mask
    # (extracted at input resolution) must be downsampled 2x to land in the
    # output-voxel-resolution annotation chunk.
    volume_meta = {
        "output_size": [4, 8, 8],
        "output_voxel_size": [4, 4, 4],
        "input_size": [16, 32, 32],
        "input_voxel_size": [2, 2, 2],
        "dataset_offset_nm": [0, 0, 0],
        "dataset_path": "fake-dataset",
        "ai_annotate_label_name": "test_organelle",
        "ai_annotate_gemini_model": "gemini-3-pro-image",
        "corrections_dir": str(tmp_path),
    }

    raw_crop = np.full((16, 32, 32), 50, dtype=np.uint8)
    # Left half of the destination window (computed by hand in the module
    # docstring/plan: window is raw_crop[8, 15:31, 15:31]) gets a distinct
    # value so we can tell which half survives downsampling.
    raw_crop[8, 15:31, 15:23] = 200

    recolored = np.zeros((16, 16, 3), dtype=np.uint8)
    recolored[:, :8] = (255, 0, 0)  # left half recolored red -> foreground
    recolored_image = Image.fromarray(recolored, mode="RGB")

    _patch_common(monkeypatch, raw_crop, recolored_image, volume_meta)

    point_nm = np.array([18.0, 34.0, 34.0])
    ai_annotate.run_ai_annotate(point_nm, "vol-test", "annotate-1")

    staging_dir = ai_annotate._staging_dir(volume_meta, "annotate-1")
    mask = np.load(os.path.join(staging_dir, "mask.npy"))
    with open(os.path.join(staging_dir, "meta.json")) as f:
        meta = json.load(f)

    assert mask.shape == (8, 8)
    assert (mask[:, :4] == 255).all()
    assert (mask[:, 4:] == 0).all()
    assert meta["chunk_indices"] == [1, 1, 1]
    assert meta["z_row_index"] == 0

    progress = ai_annotate._get_progress("vol-test")
    assert progress["status"] == "ready"
    assert progress["annotate_id"] == "annotate-1"


def test_run_ai_annotate_clips_gracefully_when_context_crop_is_small(monkeypatch, tmp_path, caplog):
    # input_size is too small for the chunk's full XY footprint to fit inside
    # the fetched context crop (but the click's own chunk still overlaps it,
    # since the crop is centered on the click and the click is inside the
    # chunk by construction) -- this should clip and zero-pad rather than
    # crash or silently produce a wrong-shaped mask.
    volume_meta = {
        "output_size": [4, 8, 8],
        "output_voxel_size": [4, 4, 4],
        "input_size": [4, 4, 4],
        "input_voxel_size": [2, 2, 2],
        "dataset_offset_nm": [0, 0, 0],
        "dataset_path": "fake-dataset",
        "ai_annotate_label_name": "test_organelle",
        "ai_annotate_gemini_model": "gemini-3-pro-image",
        "corrections_dir": str(tmp_path),
    }
    raw_crop = np.full((4, 4, 4), 50, dtype=np.uint8)
    recolored_image = Image.new("RGB", (4, 4), (255, 0, 0))
    _patch_common(monkeypatch, raw_crop, recolored_image, volume_meta)

    point_nm = np.array([18.0, 34.0, 34.0])
    ai_annotate.run_ai_annotate(point_nm, "vol-test", "annotate-2")

    staging_dir = ai_annotate._staging_dir(volume_meta, "annotate-2")
    mask = np.load(os.path.join(staging_dir, "mask.npy"))

    assert mask.shape == (8, 8)
    assert "clipped" in caplog.text
