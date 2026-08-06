"""Tests for run_ai_annotate's click -> context-crop geometry, with
ImageDataInterface and Gemini mocked out. This exercises the input-voxel <->
output-voxel coordinate math directly (a prior draft had a unit-mixing bug
here that these tests would have caught), and the depth-axis generalization
(the click can be made in any of neuroglancer's three orthogonal views, not
just the default XY/z-depth one).

The write region is the ENTIRE reviewed context crop (not a single
grid-aligned model chunk), so mask.shape always equals GEMINI_CROP_SIZE_
VOXELS -- there's no separate destination-chunk sub-windowing to test here
(see tests/ai_annotate/test_overlay_write.py for the volume-bounds clipping
that happens at write time instead).
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


def _patch_common(monkeypatch, raw_crop, recolored_image, volume_meta, crop_size_voxels):
    import cellmap_flow.image_data_interface as idi_module

    # Real generate_recolored_image always resizes its result back to the
    # input image's size (see gemini_backend.py) -- mimic that contract here
    # so extract_mask's output stays aligned with input_image/mask_for_preview.
    def _fake_generate(image, *a, **kw):
        return recolored_image.resize(image.size)

    monkeypatch.setattr(idi_module, "ImageDataInterface", lambda *a, **kw: _FakeIDI(raw_crop))
    monkeypatch.setattr(ai_annotate, "generate_recolored_image", _fake_generate)
    monkeypatch.setattr(ai_annotate, "_get_volume_metadata", lambda volume_id: volume_meta)
    # GEMINI_CROP_SIZE_VOXELS is a fixed default (512) independent of the
    # model's input_size, and is expressed in *output*-voxel-resolution
    # pixels -- shrink it here so test fixtures stay small. Callers pass
    # crop_size_voxels already scaled by input_voxel_size/output_voxel_size
    # so the resulting fetched-crop shape (in input-voxel pixels) matches
    # the raw_crop fixture's shape.
    monkeypatch.setattr(ai_annotate, "GEMINI_CROP_SIZE_VOXELS", crop_size_voxels)


def test_run_ai_annotate_downsamples_mask_to_output_voxel_resolution(monkeypatch, tmp_path):
    # Output voxel size is 2x coarser than input voxel size, so the mask
    # (extracted at input resolution, over the FULL context crop) must be
    # downsampled 2x to land at output-voxel resolution.
    volume_meta = {
        "output_voxel_size": [4, 4, 4],
        "input_voxel_size": [2, 2, 2],
        "dataset_offset_nm": [0, 0, 0],
        "dataset_path": "fake-dataset",
        "ai_annotate_label_name": "test_organelle",
        "ai_annotate_gemini_model": "gemini-3-pro-image",
        "corrections_dir": str(tmp_path),
    }

    raw_crop = np.full((32, 32, 32), 50, dtype=np.uint8)

    # GEMINI_CROP_SIZE_VOXELS is expressed in output-voxel pixels; with
    # output_voxel_size=2x input_voxel_size, a value of 16 here yields a
    # 32x32x32 fetched-crop shape (in input-voxel pixels), matching raw_crop.
    # Color the left half of the recolored image red so we can tell which
    # half survives downsampling to the 16x16 output-voxel mask.
    recolored = np.zeros((32, 32, 3), dtype=np.uint8)
    recolored[:, :16] = (255, 0, 0)
    recolored_image = Image.fromarray(recolored, mode="RGB")

    _patch_common(monkeypatch, raw_crop, recolored_image, volume_meta, crop_size_voxels=16)

    point_nm = np.array([18.0, 34.0, 34.0])
    ai_annotate.run_ai_annotate(point_nm, 0, "vol-test", "annotate-1")

    staging_dir = ai_annotate._staging_dir(volume_meta, "annotate-1")
    mask = np.load(os.path.join(staging_dir, "mask.npy"))
    with open(os.path.join(staging_dir, "meta.json")) as f:
        meta = json.load(f)

    assert mask.shape == (16, 16)
    assert (mask[:, :8] == 255).all()
    assert (mask[:, 8:] == 0).all()
    assert meta["depth_axis"] == 0
    # context_offset_nm=[-14,2,2] -> in-plane write offset = round(-14/4, 2/4, 2/4)
    # with the depth-axis (0) component overridden to the click's own
    # absolute output-voxel index: round(18/4) = 4 (round-half-to-even).
    assert meta["write_offset_vox"] == [4, 0, 0]

    progress = ai_annotate._get_progress("vol-test")
    assert progress["status"] == "ready"
    assert progress["annotate_id"] == "annotate-1"


def test_run_ai_annotate_handles_non_default_depth_axis(monkeypatch, tmp_path):
    # depth_axis=1 means the click was made in an XZ-style view (y is the
    # sliced/depth axis, z and x are in-plane) -- this must not silently
    # fall back to treating z as depth.
    volume_meta = {
        "output_voxel_size": [4, 4, 4],
        "input_voxel_size": [2, 2, 2],
        "dataset_offset_nm": [0, 0, 0],
        "dataset_path": "fake-dataset",
        "ai_annotate_label_name": "test_organelle",
        "ai_annotate_gemini_model": "gemini-3-pro-image",
        "corrections_dir": str(tmp_path),
    }

    raw_crop = np.full((32, 32, 32), 50, dtype=np.uint8)

    # Full context slice (taken along axis=1) is 32x32 over (z, x); color
    # the left half (x range :16) red.
    recolored = np.zeros((32, 32, 3), dtype=np.uint8)
    recolored[:, :16] = (255, 0, 0)
    recolored_image = Image.fromarray(recolored, mode="RGB")

    _patch_common(monkeypatch, raw_crop, recolored_image, volume_meta, crop_size_voxels=16)

    point_nm = np.array([34.0, 34.0, 34.0])
    ai_annotate.run_ai_annotate(point_nm, 1, "vol-test", "annotate-3")

    staging_dir = ai_annotate._staging_dir(volume_meta, "annotate-3")
    mask = np.load(os.path.join(staging_dir, "mask.npy"))
    with open(os.path.join(staging_dir, "meta.json")) as f:
        meta = json.load(f)

    assert mask.shape == (16, 16)
    assert (mask[:, :8] == 255).all()
    assert (mask[:, 8:] == 0).all()
    assert meta["depth_axis"] == 1
    # context_offset_nm=[2,2,2] -> in-plane write offset = round(2/4, 2/4) = [0,0]
    # with the depth-axis (1) component overridden to round(34/4) = 8.
    assert meta["write_offset_vox"] == [0, 8, 0]


def test_run_ai_annotate_mask_shape_matches_crop_size_regardless_of_scale(monkeypatch, tmp_path):
    # With no separate "destination chunk" anymore -- the write region is
    # exactly the context crop -- mask shape always equals GEMINI_CROP_SIZE_
    # VOXELS (in output-voxel pixels), even for a tiny crop, with no special
    # clip/pad logic needed here (that only happens at write time now --
    # see test_overlay_write.py).
    volume_meta = {
        "output_voxel_size": [4, 4, 4],
        "input_voxel_size": [2, 2, 2],
        "dataset_offset_nm": [0, 0, 0],
        "dataset_path": "fake-dataset",
        "ai_annotate_label_name": "test_organelle",
        "ai_annotate_gemini_model": "gemini-3-pro-image",
        "corrections_dir": str(tmp_path),
    }
    raw_crop = np.full((4, 4, 4), 50, dtype=np.uint8)
    recolored_image = Image.new("RGB", (4, 4), (255, 0, 0))
    _patch_common(monkeypatch, raw_crop, recolored_image, volume_meta, crop_size_voxels=2)

    point_nm = np.array([18.0, 34.0, 34.0])
    ai_annotate.run_ai_annotate(point_nm, 0, "vol-test", "annotate-2")

    staging_dir = ai_annotate._staging_dir(volume_meta, "annotate-2")
    mask = np.load(os.path.join(staging_dir, "mask.npy"))

    assert mask.shape == (2, 2)
    assert (mask == 255).all()


def test_depth_axis_from_layout():
    # Unrecognized/multi-panel layouts (None, "4panel", "3d") fall back to
    # the depth axis of the identity/unrotated cross-section orientation,
    # which for our declared ["z", "y", "x"] axis order is "x" -- see
    # depth_axis_from_layout's docstring for the derivation.
    assert ai_annotate.depth_axis_from_layout(None) == 2
    assert ai_annotate.depth_axis_from_layout("4panel") == 2
    assert ai_annotate.depth_axis_from_layout("3d") == 2
    assert ai_annotate.depth_axis_from_layout("xy") == 2
    assert ai_annotate.depth_axis_from_layout("xy-3d") == 2
    assert ai_annotate.depth_axis_from_layout("xz") == 1
    assert ai_annotate.depth_axis_from_layout("yz") == 0
    assert ai_annotate.depth_axis_from_layout({"type": "yz"}) == 0
