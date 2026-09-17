"""A browser-painted annotation volume must be trainable by the new path.

create_dataloader chooses its dataset solely by whether _virtual_sources.json
exists in the corrections dir: present means VirtualPatchDataset, which is the
only dataset that honours good regions; absent means the legacy per-chunk
CorrectionDataset, which ignores them entirely. Only the YAML importer used to
write that sentinel, so painting scribbles in the browser and marking regions
good produced a run that quietly trained on neither.
"""

import json
import os

import pytest

from cellmap_flow.dashboard.routes.finetune import common, training
from cellmap_flow.finetune.virtual_dataset import (
    VIRTUAL_MANIFEST_FILENAME,
    read_manifest,
)


def _volume(corrections_dir, **overrides):
    volume = {
        "zarr_path": os.path.join(corrections_dir, "vol-abc.zarr"),
        "dataset_path": "/nrs/raw.zarr/em/s0",
        "input_size": [178, 178, 178],
        "output_size": [56, 56, 56],
        "input_voxel_size": [16, 16, 16],
        "output_voxel_size": [16, 16, 16],
        "corrections_dir": corrections_dir,
    }
    volume.update(overrides)
    return volume


@pytest.fixture
def corrections(tmp_path, monkeypatch):
    d = tmp_path / "session" / "corrections"
    d.mkdir(parents=True)
    monkeypatch.setattr(common.g, "input_norms", [], raising=False)
    monkeypatch.setattr(common.g, "postprocess", [], raising=False)
    return str(d)


def test_a_volume_record_yields_a_loadable_manifest(corrections):
    path = common.write_volume_manifest(_volume(corrections))
    assert path is not None
    assert os.path.basename(path) == VIRTUAL_MANIFEST_FILENAME

    manifest = read_manifest(corrections)
    assert manifest["kind"] == "volume_zarr_v1"
    assert manifest["volume_zarr_path"].endswith("vol-abc.zarr")
    assert manifest["raw_dataset_path"] == "/nrs/raw.zarr/em/s0"
    assert manifest["input_size_voxels"] == [178, 178, 178]
    assert manifest["output_size_voxels"] == [56, 56, 56]
    # None means "one patch per populated chunk" -- cover what was painted.
    assert manifest["patches_per_epoch"] is None

    # dataset_from_manifest reads these by name; a rename here would silently
    # put the trainer back on the legacy path.
    for key in (
        "volume_zarr_path",
        "raw_dataset_path",
        "input_size_voxels",
        "output_size_voxels",
        "input_voxel_size_nm",
        "output_voxel_size_nm",
    ):
        assert key in manifest


@pytest.mark.parametrize(
    "missing",
    ["zarr_path", "dataset_path", "input_size", "output_size", "output_voxel_size"],
)
def test_an_incomplete_volume_writes_nothing(corrections, missing):
    """Better the old path than a manifest the trainer chokes on."""
    volume = _volume(corrections)
    volume[missing] = None
    assert common.write_volume_manifest(volume) is None
    assert not os.path.exists(os.path.join(corrections, VIRTUAL_MANIFEST_FILENAME))


def test_backfill_writes_for_a_session_that_predates_the_manifest(
    corrections, monkeypatch
):
    """The bug in the field: a real volume, no sentinel, good regions ignored."""
    monkeypatch.setattr(
        training.g, "annotation_volumes", {"vol-abc": _volume(corrections)},
        raising=False,
    )
    assert read_manifest(corrections) is None

    manifest = training._backfill_manifest(corrections)
    assert manifest is not None
    assert manifest["kind"] == "volume_zarr_v1"
    assert read_manifest(corrections) is not None


def test_backfill_declines_when_no_volume_belongs_to_that_session(
    corrections, monkeypatch, tmp_path
):
    other = tmp_path / "elsewhere" / "corrections"
    other.mkdir(parents=True)
    monkeypatch.setattr(
        training.g, "annotation_volumes", {"vol-xyz": _volume(str(other))},
        raising=False,
    )
    assert training._backfill_manifest(corrections) is None
    assert not os.path.exists(os.path.join(corrections, VIRTUAL_MANIFEST_FILENAME))


def test_good_regions_live_beside_the_corrections_dir_not_inside_it(corrections):
    """load_good_regions_for looks one level up; keep the writer in step."""
    from cellmap_flow.dashboard.routes.finetune import good_regions as gr
    from cellmap_flow.finetune.virtual_dataset import (
        GOOD_REGIONS_FILENAME,
        load_good_regions_for,
    )

    session_dir = os.path.dirname(corrections)
    with open(os.path.join(session_dir, GOOD_REGIONS_FILENAME), "w") as f:
        json.dump(
            [{"id": "a", "label": "good-1",
              "offset_nm": [0, 0, 0], "shape_nm": [896.0, 896.0, 896.0]}],
            f,
        )

    assert len(load_good_regions_for(corrections)) == 1
    # And the dashboard writes to that same place.
    assert os.path.basename(gr.GOOD_REGIONS_FILENAME) == GOOD_REGIONS_FILENAME


class TestRehearsalFractionOverride:
    """0 must stay distinct from blank: one turns rehearsal off for a run,
    the other leaves whatever the manifest says alone."""

    def test_blank_and_missing_leave_the_manifest_alone(self):
        assert training._parse_rehearsal_fraction_override({}) == (False, None)
        assert training._parse_rehearsal_fraction_override(
            {"rehearsal_fraction": ""}
        ) == (False, None)
        assert training._parse_rehearsal_fraction_override(
            {"rehearsal_fraction": None}
        ) == (False, None)

    def test_zero_is_a_real_choice(self):
        assert training._parse_rehearsal_fraction_override(
            {"rehearsal_fraction": 0}
        ) == (True, 0.0)

    @pytest.mark.parametrize("value,expected", [(0.25, 0.25), ("0.5", 0.5), (1, 1.0)])
    def test_valid_values_pass_through(self, value, expected):
        assert training._parse_rehearsal_fraction_override(
            {"rehearsal_fraction": value}
        ) == (True, expected)

    @pytest.mark.parametrize("value", [-0.1, 1.5, "abc"])
    def test_out_of_range_is_rejected(self, value):
        with pytest.raises(ValueError):
            training._parse_rehearsal_fraction_override({"rehearsal_fraction": value})


def test_the_override_reaches_the_manifest(corrections, monkeypatch):
    common.write_volume_manifest(_volume(corrections))
    manifest = read_manifest(corrections)
    assert "rehearsal_fraction" not in manifest

    training._refresh_virtual_manifest_for_training(
        corrections, manifest, {"rehearsal_fraction": 0.5}, "submit"
    )
    assert read_manifest(corrections)["rehearsal_fraction"] == 0.5

    # And turning it off for a run is persisted as 0, not dropped.
    training._refresh_virtual_manifest_for_training(
        corrections, read_manifest(corrections), {"rehearsal_fraction": 0}, "submit"
    )
    assert read_manifest(corrections)["rehearsal_fraction"] == 0.0


class TestGoodRegionReporting:
    """Turning rehearsal off is not the same as a good region being broken.

    Both ended up in one branch that warned "none usable", so setting
    rehearsal to 0 -- a deliberate choice -- read in the log exactly like a
    good region that had fallen outside the annotation volume.
    """

    def _dataset(self, rehearsal_fraction, effective=0.0, centers=None):
        from cellmap_flow.finetune.virtual_dataset import VirtualPatchDataset

        ds = VirtualPatchDataset.__new__(VirtualPatchDataset)
        ds.good_regions = [{"id": "g1"}]
        ds.rehearsal_fraction = rehearsal_fraction
        ds._effective_rehearsal_fraction = effective
        ds._rehearsal_centers = centers
        return ds

    def _records(self, ds, caplog):
        import logging

        from cellmap_flow.finetune import virtual_dataset as vd

        caplog.clear()
        with caplog.at_level(logging.INFO, logger=vd.__name__):
            ds._log_rehearsal_status()
        return caplog.records

    def test_rehearsal_zero_is_reported_as_a_choice_not_a_fault(self, caplog):
        records = self._records(self._dataset(0.0), caplog)
        assert [r.levelname for r in records] == ["INFO"]
        assert "set to 0" in records[0].getMessage()

    def test_a_region_outside_the_volume_is_still_a_warning(self, caplog):
        records = self._records(self._dataset(None), caplog)
        assert [r.levelname for r in records] == ["WARNING"]
        assert "landed inside" in records[0].getMessage()

    def test_working_anchors_say_what_share_of_patches_they_get(self, caplog):
        import numpy as np

        ds = self._dataset(None, effective=0.25, centers=np.zeros((1, 3)))
        records = self._records(ds, caplog)
        assert [r.levelname for r in records] == ["INFO"]
        assert "25% of patches" in records[0].getMessage()

    def test_no_good_regions_means_no_message_at_all(self, caplog):
        ds = self._dataset(None)
        ds.good_regions = []
        assert self._records(ds, caplog) == []
