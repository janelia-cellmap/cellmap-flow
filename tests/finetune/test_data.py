"""cellmap_flow.finetune.data: VirtualPatchDataset through the patches it draws
(where they are centred, what they hold, what good regions add)."""

import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from cellmap_flow.finetune.session.volume import create_volume_zarr, plan_volume
from cellmap_flow.finetune.data import VirtualPatchDataset, dataset_from_manifest
from cellmap_flow.finetune.data.reader import PatchReader


def _labels(size, **boxes):
    """A size^3 annotation (0 unannotated, 1 background, 2+ foreground) with ``boxes``: name=(value, slices)."""
    labels = np.zeros((size,) * 3, np.uint8)
    for value, where in boxes.values():
        labels[where] = value
    return labels


BLOCK = np.s_[4:28, 4:28, 4:28]
NORMS = {"MinMaxNormalizer": {"min_value": 0, "max_value": 255, "invert": False},
         "LambdaNormalizer": {"expression": "x*2-1"}}


@pytest.mark.parametrize("norms", [pytest.param(NORMS, id="keyed by name, as a yaml has it"),
                                   pytest.param([{"name": k, **v} for k, v in NORMS.items()],
                                                id="a list, as the dashboard sends it")])
def test_patches_are_normalized_as_inference_sees_them_in_a_spawned_worker_too(annotation_volume, norms):
    """The trainer is its own process, whose process_chain() is empty: it trained on
    raw uint8 while inference fed [-1, 1]. A spawned loader worker gets a pickled
    dataset, which a Lambda normalizer holding a lambda did not survive. The
    dashboard sends the chain as a list, which dict() turned into no normalizers."""
    volume = annotation_volume(_labels(32, fg=(2, BLOCK)))
    raw, _ = volume.dataset()[0]
    assert 110 < float(raw.min()) and float(raw.max()) < 140  # 128, as stored
    raw, _ = pickle.loads(pickle.dumps(volume.dataset(input_norm_config=norms)))[0]
    assert float(raw.abs().max()) < 0.05  # 128 / 255 * 2 - 1


def test_the_trainer_reads_the_raw_channel_the_chain_picks(ome_pyramid):
    """The trainer is its own process, whose process_chain() is empty: it read
    channel 0 of a multi-channel raw whatever channel the session's
    ChannelSelector picked for inference."""
    raw = ome_pyramid(((16, 8),), channels=2)  # 16^3 voxels of 16 nm, corner 0; each holds z + 1
    zarr.open(raw + "/s0", mode="r+")[1] += 100
    reader = PatchReader("unused", raw, (4,) * 3, (4,) * 3, (16.0,) * 3, (16.0,) * 3, np.zeros(3), np.full(3, 16),
                         input_norm_config=[{"name": "ChannelSelector", "channel": 1}])
    # Around output voxel 8: raw voxels 6 to 9, which hold z + 1 + 100 in channel 1.
    assert reader.raw(np.full(3, 8.0))[:, 0, 0].tolist() == [107, 108, 109, 110]


FAR_APART = _labels(64, fg=(2, np.s_[2:4, 2:4, 2:4]), fix=(1, np.s_[50:52, 50:52, 50:52]))
SCRIBBLE = _labels(32, s=(2, np.s_[2:6, 2:6, 2:6]))


def _drawn(dataset, n=16):
    """The annotation patches of ``n`` draws."""
    return [dataset[i][1] for i in range(n)]


def test_a_background_fix_far_from_any_foreground_is_drawn(annotation_volume):
    """Painting background where the model hallucinates: only foreground was a
    patch centre, so a fix further than half a patch from any was never in one."""
    dataset = annotation_volume(FAR_APART).dataset()
    assert any(1 in ann and 2 not in ann for ann in _drawn(dataset))


def test_both_classes_are_counted_wherever_they_are_annotated(annotation_volume):
    """What the single-class warning goes by. Neither pool holds a crop's
    background, and the painted pool is only what lies outside every crop."""
    labels = _labels(32, crop=(1, np.s_[:8, :8, :8]), inside=(2, np.s_[2:4, 2:4, 2:4]),
                     painted=(1, np.s_[20:22, 20:22, 20:22]))
    crop = {"annotation_offset_voxels": [0, 0, 0], "annotation_shape_voxels": [8, 8, 8]}
    assert annotation_volume(labels, crops=[crop]).dataset().class_voxels == (8**3 - 8 + 8, 8)


def _region(centre_voxel, id="r1"):
    """A 4-voxel box around ``centre_voxel``, the way the dashboard saves one."""
    return {"id": id, "label": "good", "offset_nm": [centre_voxel * 16.0 - 32.0] * 3, "shape_nm": [64.0] * 3}


def test_good_regions_get_their_share_of_the_patches(annotation_volume):
    """A third pool: the other two index on foreground, which a good region has
    none of, so marking one changed nothing about a run."""
    dataset = annotation_volume(SCRIBBLE).dataset(good_regions=[_region(24)], rehearsal_fraction=0.5,
                                                  patches_per_epoch=64)
    assert 16 < sum(bool(dataset[i][2].any()) for i in range(64)) < 48


@pytest.mark.parametrize("regions_file, beside, anchored", [
    pytest.param(json.dumps([_region(24)]), True, True, id="beside the corrections"),
    pytest.param(json.dumps([_region(24)]), False, False, id="no corrections dir given"),
    pytest.param("{not json", True, False, id="a corrupt file is not fatal"),
])
def test_the_trainer_reads_the_regions_marked_beside_its_corrections(annotation_volume, tmp_path,
                                                                     regions_file, beside, anchored):
    """Read when training starts, so regions marked after the import count."""
    volume = annotation_volume(SCRIBBLE)
    corrections = tmp_path / "20260101_000000" / "corrections"
    corrections.mkdir(parents=True)
    (corrections.parent / "good_regions.json").write_text(regions_file)
    manifest = {"kind": "volume_zarr_v1", "volume_zarr_path": volume.path, "raw_dataset_path": volume.raw,
                "input_size_voxels": [8] * 3, "output_size_voxels": [4] * 3,
                "input_voxel_size_nm": [16.0] * 3, "output_voxel_size_nm": [16.0] * 3, "anchor_fraction": 0.0}
    assert dataset_from_manifest(manifest, str(corrections) if beside else None).emits_anchor == anchored


@pytest.fixture
def janelia_raw(tmp_path, ome_zarr):
    """A 32^3 s0 at 8 nm whose voxels hold z + 1, and s1 at 16 nm; every level's corner is -4 nm."""
    levels = []
    for i, (scale, translation) in enumerate([(8.0, 0.0), (16.0, 4.0)]):
        n = 32 >> i
        z = np.arange(1, n + 1, dtype=np.uint8)[:, None, None]
        levels.append((f"s{i}", np.broadcast_to(z, (n, n, n)).copy(), scale, translation))
    return ome_zarr(tmp_path / "raw.zarr", *levels, chunks=(8, 8, 8))


def _paired_dataset(tmp_path, raw, output_size):
    """A volume planned over ``raw`` for a model of that output size, twice
    that input size, 8 nm input and 16 nm output voxels, and a dataset over both."""
    input_voxel_size, output_voxel_size = (8.0,) * 3, (16.0,) * 3
    model = SimpleNamespace(input_shape=[2 * output_size] * 3, output_shape=[output_size] * 3,
                            input_voxel_size=input_voxel_size, output_voxel_size=output_voxel_size)
    path = create_volume_zarr(str(tmp_path / "vol.zarr"), plan_volume(raw, model), dataset_path=raw, model_name="m")
    return path, lambda: VirtualPatchDataset(
        volume_zarr_path=path, raw_dataset_path=raw, input_size_voxels=(2 * output_size,) * 3,
        output_size_voxels=(output_size,) * 3, input_voxel_size_nm=input_voxel_size,
        output_voxel_size_nm=output_voxel_size, patches_per_epoch=1, jitter_voxels=(0, 0, 0), anchor_fraction=0.0)


@pytest.mark.parametrize("output_size", [pytest.param(2, id="even output size"),
                                         pytest.param(3, id="odd output size")])
def test_a_label_is_paired_with_the_raw_it_covers(tmp_path, janelia_raw, output_size):
    """A volume's dataset_offset_nm is its OME translation, voxel 0's centre, as
    Neuroglancer drew it; the trainer read it, and the raw's, as corners, so
    over a Janelia pyramid every label met raw a fraction of a voxel away."""
    path, dataset = _paired_dataset(tmp_path, janelia_raw, output_size)
    zarr.open_group(path, mode="r+")["annotation/s0"][5, 8, 8] = 2
    raw, ann = dataset()[0]
    z0 = 5 - int(np.argwhere(ann[0].numpy() == 2)[0][0])  # where the patch starts, in annotation voxels
    # Annotation voxel z covers raw voxels 2z and 2z + 1, which hold 2z + 1 and 2z + 2.
    assert raw[0, :, 0, 0].tolist() == list(range(2 * z0 + 1, 2 * z0 + 2 * output_size + 1))


# A 2D network's patch, one slice of 8 x 8, as Cellpose's is one of 256 x 256.
SLICE = dict(input_size_voxels=(1, 8, 8), output_size_voxels=(1, 8, 8))


def _where_raw():
    """A 32^3 raw whose voxel (z, y, x) holds z * 1024 + y * 32 + x: a patch says where each pixel came from."""
    z, y, x = np.indices((32,) * 3)
    return (z * 1024 + y * 32 + x).astype(np.uint16)


def _painted_slice(depth_axis):
    """Paint one voxel thick across ``depth_axis``, at 10: background with a foreground square in it."""
    labels = np.zeros((32,) * 3, np.uint8)
    for value, extent in ((1, slice(8, 24)), (2, slice(12, 20))):
        where = [extent] * 3
        where[depth_axis] = 10
        labels[tuple(where)] = value
    return labels


@pytest.mark.parametrize("depth_axis", [pytest.param(0, id="x-y"), pytest.param(1, id="z-x"),
                                        pytest.param(2, id="z-y")])
def test_a_one_slice_patch_is_cut_in_the_plane_its_paint_was_drawn_in(annotation_volume, depth_axis):
    """Paint drawn in a z–y slice is one voxel thick in x, and the x–y patches
    a 2D network trained on held a line of it a voxel wide: Cellpose learned
    flows along y alone, and segmented strips."""
    labels = _painted_slice(depth_axis)
    dataset = annotation_volume(labels, raw=_where_raw()).dataset(**SLICE, patches_per_epoch=32)
    rows, columns = (axis for axis in range(3) if axis != depth_axis)
    for i in range(32):
        raw, ann = (t[0, 0].numpy().astype(int) for t in dataset[i])
        where = np.stack([raw // 1024, raw // 32 % 32, raw % 32])
        assert (where[depth_axis] == 10).all(), "on the painted slice, never jittered off it"
        assert (np.diff(where[rows], axis=0) == 1).all() and (np.diff(where[columns], axis=1) == 1).all()
        assert (ann == labels[tuple(where)]).all(), "each label paired with the raw it covers"
        assert (ann > 0).sum() > 8, "paint, not a line of it"
