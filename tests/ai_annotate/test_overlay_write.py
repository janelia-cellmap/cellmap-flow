"""Tests for overlay.py's AI-annotate mask-write index computation -- the
whole reviewed context crop is written on Accept (see ai_annotate.py's
_compute_context_write_region), which can extend past the annotation
volume's actual bounds near a dataset edge; _compute_ai_mask_write_index
is what clips that down to a valid zarr index.
"""

import numpy as np
import pytest

from cellmap_flow.dashboard.routes.finetune.overlay import _compute_ai_mask_write_index, _label_instances


def test_write_index_in_bounds():
    mask = np.full((4, 4), 255, dtype=np.uint8)
    idx, mask_clipped = _compute_ai_mask_write_index(
        arr_shape=(10, 10, 10), write_offset_vox=(3, 2, 2), depth_axis=0, mask_2d=mask
    )
    assert idx == (3, slice(2, 6), slice(2, 6))
    assert mask_clipped.shape == (4, 4)
    assert (mask_clipped == 255).all()


def test_write_index_clips_to_volume_bounds():
    # Context crop extends 2 voxels past the volume's edge on axis 1.
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[:2] = 255
    idx, mask_clipped = _compute_ai_mask_write_index(
        arr_shape=(10, 10, 10), write_offset_vox=(3, 8, 2), depth_axis=0, mask_2d=mask
    )
    assert idx == (3, slice(8, 10), slice(2, 6))
    assert mask_clipped.shape == (2, 4)
    assert (mask_clipped == mask[:2, :]).all()


def test_write_index_raises_when_entirely_out_of_bounds():
    mask = np.ones((4, 4), dtype=np.uint8)
    with pytest.raises(ValueError):
        _compute_ai_mask_write_index(
            arr_shape=(10, 10, 10), write_offset_vox=(3, 20, 2), depth_axis=0, mask_2d=mask
        )


def test_label_instances_assigns_distinct_ids_to_separated_blobs():
    # Two blobs separated by a black (background) column -- e.g. Gemini
    # honoring _INSTANCE_SEPARATION_INSTRUCTION -- must come out as two
    # distinct instance ids, not one shared foreground id.
    fg_mask = np.zeros((4, 9), dtype=bool)
    fg_mask[:, :4] = True  # left blob
    fg_mask[:, 5:] = True  # right blob, column 4 stays background

    labels, n = _label_instances(fg_mask, first_id=2)

    assert n == 2
    left_ids = set(np.unique(labels[:, :4]))
    right_ids = set(np.unique(labels[:, 5:]))
    assert left_ids <= {2, 3}
    assert right_ids <= {2, 3}
    assert left_ids != right_ids
    assert (labels[:, 4] == 0).all()


def test_label_instances_empty_mask_returns_zero_instances():
    labels, n = _label_instances(np.zeros((4, 4), dtype=bool))
    assert n == 0
    assert (labels == 0).all()


def test_label_instances_collapses_on_uint8_overflow(monkeypatch, caplog):
    import cellmap_flow.dashboard.routes.finetune.overlay as overlay_module

    fg_mask = np.ones((4, 4), dtype=bool)

    def fake_cc_label(mask):
        # Pretend the mask decomposed into 300 instances (would overflow
        # uint8's 255-value budget after adding first_id=2).
        return np.ones_like(mask, dtype=np.int32), 300

    monkeypatch.setattr("scipy.ndimage.label", fake_cc_label)

    labels, n = overlay_module._label_instances(fg_mask, first_id=2)

    assert n == 1
    assert (labels[fg_mask] == 2).all()
