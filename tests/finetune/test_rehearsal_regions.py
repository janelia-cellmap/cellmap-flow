"""Good regions must reach training, and must lose to a scribble drawn in one.

A good region is a box the user looked at and certified the model already
handles. Neither existing pool can sample one: both index on foreground
voxels and a good region has none by construction, so without a third pool
marking regions writes a file and changes nothing about a run.
"""

import json
import os

import numpy as np
import torch
import pytest
import zarr

from cellmap_flow.finetune.virtual_dataset import (
    VirtualPatchDataset,
    load_good_regions_for,
)

MULTISCALES = [{
    "version": "0.4",
    "axes": [{"name": a, "type": "space", "unit": "nanometer"} for a in "zyx"],
    "datasets": [{"path": "s0", "coordinateTransformations": [
        {"type": "scale", "scale": [16.0, 16.0, 16.0]},
        {"type": "translation", "translation": [0.0, 0.0, 0.0]},
    ]}],
}]


@pytest.fixture
def volume(tmp_path):
    """A 32^3 volume with a small scribble, and matching raw."""
    raw_path = str(tmp_path / "raw.zarr")
    r = zarr.open_group(raw_path, mode="w")
    r.create_dataset("s0", shape=(32, 32, 32), dtype="uint8", chunks=(16, 16, 16))
    r["s0"][:] = 128
    r.attrs["multiscales"] = MULTISCALES

    vol_path = str(tmp_path / "vol.zarr")
    v = zarr.open_group(vol_path, mode="w")
    v.create_group("annotation").create_dataset(
        "s0", shape=(32, 32, 32), chunks=(16, 16, 16), dtype="uint8", fill_value=0
    )
    arr = np.zeros((32, 32, 32), dtype=np.uint8)
    arr[2:6, 2:6, 2:6] = 2          # a scribble, far from the good region
    v["annotation"]["s0"][:] = arr
    v.attrs["dataset_offset_nm"] = [0.0, 0.0, 0.0]
    v["annotation"].attrs["multiscales"] = MULTISCALES
    return raw_path, vol_path


def _common(raw_path, vol_path):
    return dict(
        volume_zarr_path=vol_path,
        raw_dataset_path=raw_path,
        input_size_voxels=(8, 8, 8),
        output_size_voxels=(4, 4, 4),
        input_voxel_size_nm=(16, 16, 16),
        output_voxel_size_nm=(16, 16, 16),
        patches_per_epoch=64,
        seed=0,
    )


def _region(centre_voxel, size_voxels=4, voxel_nm=16.0, **extra):
    """A box centred on a voxel, expressed the way the dashboard saves it."""
    size_nm = size_voxels * voxel_nm
    centre_nm = np.array(centre_voxel, dtype=float) * voxel_nm
    return {
        "id": extra.pop("id", "r1"),
        "label": "good-1",
        "offset_nm": (centre_nm - size_nm / 2).tolist(),
        "shape_nm": [size_nm] * 3,
        **extra,
    }


def test_without_good_regions_nothing_changes(volume):
    """The 2-tuple contract has to survive, or every existing run breaks."""
    ds = VirtualPatchDataset(**_common(*volume))
    assert ds.emits_anchor is False
    assert len(ds[0]) == 2


def test_good_regions_are_actually_sampled(volume):
    """Both FG pools index on foreground, so only a third pool reaches here."""
    ds = VirtualPatchDataset(
        good_regions=[_region((24, 24, 24))],
        rehearsal_fraction=0.5,
        **_common(*volume),
    )
    assert ds.emits_anchor is True

    anchored = 0
    for i in range(200):
        raw, ann, anchor = ds[i]
        assert raw.shape == (1, 8, 8, 8) and ann.shape == (1, 4, 4, 4)
        assert anchor.shape == ann.shape
        if float(anchor.sum()) > 0:
            anchored += 1
    # Half of 200, with room for sampling noise.
    assert 60 < anchored < 140, f"expected ~100 rehearsal patches, got {anchored}"


def test_a_rehearsal_patch_anchors_every_unannotated_voxel(volume):
    ds = VirtualPatchDataset(
        good_regions=[_region((24, 24, 24))],
        rehearsal_fraction=1.0,
        **_common(*volume),
    )
    _, ann, anchor = ds[0]
    # Region is empty of annotations, so the whole patch anchors.
    assert float(ann.sum()) == 0.0
    assert float(anchor.sum()) == anchor.numel()


def test_a_scribble_inside_a_good_region_beats_the_anchor(volume):
    """Marking a region must not freeze in an error you later paint over."""
    raw_path, vol_path = volume
    # Put a scribble right where the good region is.
    arr = zarr.open(os.path.join(vol_path, "annotation", "s0"), mode="a")
    arr[24, 24, 24] = 2

    ds = VirtualPatchDataset(
        good_regions=[_region((24, 24, 24))],
        rehearsal_fraction=1.0,
        **_common(raw_path, vol_path),
    )
    _, ann, anchor = ds[0]

    annotated = ann.numpy() > 0
    assert annotated.any(), "fixture should place a scribble inside the region"
    # Where you painted, your label wins and the teacher is not consulted.
    assert not anchor.numpy()[annotated].any()
    # Everywhere else in the region still anchors.
    assert anchor.numpy()[~annotated].all()


def test_patches_from_the_other_pools_carry_no_anchor(volume):
    """Unlabeled next to a scribble means 'not got to it', not 'correct'."""
    ds = VirtualPatchDataset(
        good_regions=[_region((24, 24, 24))],
        rehearsal_fraction=0.0,
        **_common(*volume),
    )
    # rehearsal_fraction=0 means no anchoring at all, so the dataset falls
    # back to its plain contract rather than emitting a dead all-zero mask.
    assert ds.emits_anchor is False
    assert len(ds[0]) == 2


def test_a_region_outside_the_volume_is_dropped(volume):
    """A region marked against another dataset would anchor on pure zeros."""
    ds = VirtualPatchDataset(
        good_regions=[_region((24, 24, 24)), _region((9999, 9999, 9999), id="bad")],
        rehearsal_fraction=1.0,
        **_common(*volume),
    )
    assert ds._rehearsal_centers.shape[0] == 1


def test_malformed_regions_do_not_take_the_run_down(volume):
    ds = VirtualPatchDataset(
        good_regions=[{"id": "junk"}, _region((24, 24, 24))],
        rehearsal_fraction=1.0,
        **_common(*volume),
    )
    assert ds._rehearsal_centers.shape[0] == 1


def test_regions_are_read_next_to_the_corrections_dir(tmp_path):
    """Read at training time, so regions marked after import still count."""
    session = tmp_path / "20260101_000000"
    corrections = session / "corrections"
    corrections.mkdir(parents=True)
    regions = [_region((1, 1, 1))]
    (session / "good_regions.json").write_text(json.dumps(regions))

    assert load_good_regions_for(str(corrections)) == regions
    assert load_good_regions_for(None) == []
    assert load_good_regions_for(str(tmp_path / "nope")) == []


def test_a_corrupt_regions_file_is_not_fatal(tmp_path):
    session = tmp_path / "20260101_000000"
    corrections = session / "corrections"
    corrections.mkdir(parents=True)
    (session / "good_regions.json").write_text("{not json")
    assert load_good_regions_for(str(corrections)) == []


# ---------------------------------------------------------------------------
# Trainer side
# ---------------------------------------------------------------------------


class _AnchorDataset(torch.utils.data.Dataset):
    """Stands in for a VirtualPatchDataset that has good regions."""

    def __init__(self, emits_anchor=True):
        self.emits_anchor = emits_anchor

    def __len__(self):
        return 2

    def __getitem__(self, i):
        raw = torch.zeros(1, 4, 4, 4)
        ann = torch.zeros(1, 4, 4, 4)
        if not self.emits_anchor:
            return raw, ann
        return raw, ann, torch.ones(1, 4, 4, 4)


def _trainer(tmp_path, dataset, distillation_lambda):
    from cellmap_flow.finetune.lora_trainer import LoRAFinetuner

    return LoRAFinetuner(
        torch.nn.Conv3d(1, 1, 1),
        torch.utils.data.DataLoader(dataset, batch_size=1),
        output_dir=str(tmp_path / "run"),
        num_epochs=1,
        device="cpu",
        use_mixed_precision=False,
        distillation_lambda=distillation_lambda,
    )


def test_marked_regions_do_not_silently_train_nothing(tmp_path):
    """lambda=0 plus good regions would make every anchor contribute zero.

    The supervised loss never touches an unannotated voxel, so a rehearsal
    patch with no teacher term is a patch that does nothing at all -- the
    run would look healthy and the regions would have had no effect.
    """
    t = _trainer(tmp_path, _AnchorDataset(emits_anchor=True), distillation_lambda=0.0)
    assert t._anchors_available is True
    assert t.distillation_lambda == 1.0


def test_an_explicit_lambda_is_left_alone(tmp_path):
    t = _trainer(tmp_path, _AnchorDataset(emits_anchor=True), distillation_lambda=0.4)
    assert t.distillation_lambda == 0.4


def test_without_anchors_lambda_is_untouched(tmp_path):
    t = _trainer(tmp_path, _AnchorDataset(emits_anchor=False), distillation_lambda=0.0)
    assert t._anchors_available is False
    assert t.distillation_lambda == 0.0
