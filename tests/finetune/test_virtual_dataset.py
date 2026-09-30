"""VirtualPatchDataset, through the patches it draws: where they are centred,
what they hold, what good regions add, and the loaders it is read through."""

import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import zarr
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from cellmap_flow.finetune.session.volume import create_volume_zarr, plan_volume
from cellmap_flow.finetune.virtual_dataset import (
    VirtualPatchDataset,
    dataset_from_manifest,
    make_training_loader,
    rebuild_loader,
)


def _labels(size, **boxes):
    """A size^3 annotation (0 unannotated, 1 background, 2+ foreground) with ``boxes``: name=(value, slices)."""
    labels = np.zeros((size,) * 3, np.uint8)
    for value, where in boxes.values():
        labels[where] = value
    return labels


BLOCK = np.s_[4:28, 4:28, 4:28]
NORMS = {"MinMaxNormalizer": {"min_value": 0, "max_value": 255, "invert": False},
         "LambdaNormalizer": {"expression": "x*2-1"}}


@pytest.mark.parametrize("norms", [NORMS, [{"name": k, **v} for k, v in NORMS.items()]], ids=["dict", "list"])
def test_patches_are_normalized_as_inference_sees_them_in_a_spawned_worker_too(annotation_volume, norms):
    """The trainer is its own process, whose g.input_norms is empty: it trained on
    raw uint8 while inference fed [-1, 1]. A spawned loader worker gets a pickled
    dataset, which a Lambda normalizer holding a lambda did not survive. The
    dashboard sends the chain as a list, which dict() turned into no normalizers."""
    volume = annotation_volume(_labels(32, fg=(2, BLOCK)))
    raw, _ = volume.dataset()[0]
    assert 110 < float(raw.min()) and float(raw.max()) < 140  # 128, as stored
    raw, _ = pickle.loads(pickle.dumps(volume.dataset(input_norm_config=norms)))[0]
    assert float(raw.abs().max()) < 0.05  # 128 / 255 * 2 - 1


def test_consecutive_patches_are_different_draws(annotation_volume):
    """Reseeding on every __getitem__ made every patch the same one."""
    raw = np.random.default_rng(0).integers(0, 255, (32,) * 3, dtype=np.uint8)
    dataset = annotation_volume(_labels(32, fg=(2, BLOCK)), raw=raw).dataset(patches_per_epoch=8)
    assert len({dataset[i][0].numpy().tobytes() for i in range(8)}) > 1


CROP16 = {"annotation_offset_voxels": [0, 0, 0], "annotation_shape_voxels": [16, 16, 16]}


@pytest.mark.parametrize("labels, crops, length, drawn", [
    # One patch per populated chunk and epoch, by default.
    (_labels(48, a=(2, np.s_[1, 1, 1]), b=(2, np.s_[17, 17, 17]), c=(2, np.s_[33, 33, 33])), [], 3,
     lambda anns: all(2 in ann for ann in anns)),
    # Painting background where the model hallucinates: only foreground was a
    # centre, so a fix far from any was never in a patch.
    (_labels(64, fg=(2, np.s_[2:4, 2:4, 2:4]), fix=(1, np.s_[50:52, 50:52, 50:52])), [], 2,
     lambda anns: any(1 in ann and 2 not in ann for ann in anns)),
    # ... and a session that painted only background raised "no foreground voxels".
    (_labels(32, fix=(1, np.s_[20:24, 20:24, 20:24])), [], 1, lambda anns: all(1 in ann for ann in anns)),
    # An imported crop is centred on its foreground, not its background.
    (_labels(32, crop=(1, np.s_[0:16, 0:16, 0:16]), fg=(2, np.s_[4:8, 4:8, 4:8])), [CROP16], 1,
     lambda anns: all(2 in ann for ann in anns)),
    (_labels(32, crop=(1, np.s_[0:16, 0:16, 0:16])), [CROP16], 1, lambda anns: all(1 in ann for ann in anns)),
    (_labels(32), [], None, None),
], ids=["foreground", "a background fix", "background only", "a crop", "a background crop", "nothing"])
def test_every_kind_of_annotation_is_a_patch_centre(annotation_volume, labels, crops, length, drawn):
    volume = annotation_volume(labels, crops=crops)
    if length is None:
        with pytest.raises(ValueError):
            volume.dataset()
        return
    dataset = volume.dataset()
    assert len(dataset) == length
    assert drawn([dataset[i][1] for i in range(16)])


def test_the_scribbles_get_their_share_of_the_patches(annotation_volume):
    """Voxel-uniform sampling drew all but a handful of patches from a big
    imported crop and ignored the scribbles painted beside it: each pool gets
    half. With one pool only, an explicit ratio does not draw from the empty one."""
    labels = _labels(64, crop=(3, np.s_[0:32, 0:32, 0:32]), scribble=(2, np.s_[40:42, 40:42, 40:42]))
    crop = {"annotation_offset_voxels": [0, 0, 0], "annotation_shape_voxels": [32, 32, 32]}
    dataset = annotation_volume(labels, crops=[crop]).dataset(patches_per_epoch=64)
    assert 16 < sum(2 in dataset[i][1] for i in range(64)) < 48
    alone = annotation_volume(_labels(32, s=(2, np.s_[8:24, 8:24, 8:24])), name="alone")
    dataset = alone.dataset(dense_to_sparse_ratio=0.5, patches_per_epoch=4)
    assert all(2 in dataset[i][1] for i in range(4))


def _region(centre_voxel, id="r1"):
    """A 4-voxel box around ``centre_voxel``, the way the dashboard saves one."""
    return {"id": id, "label": "good", "offset_nm": [centre_voxel * 16.0 - 32.0] * 3, "shape_nm": [64.0] * 3}


@pytest.mark.parametrize("regions, fraction, painted, anchored", [
    ([], None, False, None),  # no region: the (raw, annotation) of every existing run
    ([_region(24)], 0.0, False, None),  # rehearsal switched off for this run
    ([_region(9999, "elsewhere")], 1.0, False, None),  # marked on another dataset: it would anchor zeros
    ([{"id": "junk"}], 1.0, False, None),  # malformed: dropped, not fatal
    # A third pool: the other two index on foreground, which a good region has none of.
    ([_region(24)], 0.5, False, 0.5),
    ([_region(24), _region(9999, "elsewhere")], 1.0, True, 1.0),
], ids=["none", "switched off", "outside the volume", "malformed", "half", "painted in"])
def test_good_regions_anchor_what_was_not_painted(annotation_volume, regions, fraction, painted, anchored):
    """A rehearsal patch holds the student to the teacher wherever nothing was
    painted; a scribble inside a region wins over the anchor, so a mistake
    noticed there can still be painted over."""
    labels = _labels(32, s=(2, np.s_[2:6, 2:6, 2:6]))
    if painted:
        labels[24, 24, 24] = 2
    dataset = annotation_volume(labels).dataset(good_regions=regions, rehearsal_fraction=fraction,
                                                patches_per_epoch=64)
    assert dataset.emits_anchor == (anchored is not None)
    if anchored is None:
        assert len(dataset[0]) == 2
        return
    patches = [dataset[i] for i in range(64)]
    assert sum(bool(anchor.any()) for _, _, anchor in patches) / 64 == pytest.approx(anchored, abs=0.2)
    for _, ann, anchor in patches:
        assert not anchor.any() or torch.equal(anchor, (ann == 0).float())
    assert not painted or any(ann.max() == 2 for _, ann, anchor in patches if anchor.any())


@pytest.mark.parametrize("regions_file, beside, anchored", [
    (json.dumps([_region(24)]), True, True),
    (json.dumps([_region(24)]), False, False),  # no corrections dir given: no regions
    ("{not json", True, False),  # a corrupt file is not fatal
])
def test_the_trainer_reads_the_regions_marked_beside_its_corrections(annotation_volume, tmp_path,
                                                                     regions_file, beside, anchored):
    """Read when training starts, so regions marked after the import count."""
    volume = annotation_volume(_labels(32, s=(2, np.s_[2:6, 2:6, 2:6])))
    corrections = tmp_path / "20260101_000000" / "corrections"
    corrections.mkdir(parents=True)
    (corrections.parent / "good_regions.json").write_text(regions_file)
    manifest = {"kind": "volume_zarr_v1", "volume_zarr_path": volume.path, "raw_dataset_path": volume.raw,
                "input_size_voxels": [8] * 3, "output_size_voxels": [4] * 3,
                "input_voxel_size_nm": [16.0] * 3, "output_voxel_size_nm": [16.0] * 3}
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


def _paired_dataset(tmp_path, raw, output_size, **kw):
    """A volume planned over ``raw`` for a model of that output size, and a dataset over both."""
    model = SimpleNamespace(input_shape=[2 * output_size] * 3, output_shape=[output_size] * 3,
                            input_voxel_size=(8.0,) * 3, output_voxel_size=(16.0,) * 3)
    path = create_volume_zarr(str(tmp_path / "vol.zarr"), plan_volume(raw, model), dataset_path=raw, model_name="m")
    return path, lambda: VirtualPatchDataset(
        volume_zarr_path=path, raw_dataset_path=raw, input_size_voxels=(2 * output_size,) * 3,
        output_size_voxels=(output_size,) * 3, input_voxel_size_nm=(8.0,) * 3,
        output_voxel_size_nm=(16.0,) * 3, patches_per_epoch=1, jitter_voxels=(0, 0, 0), **kw)


@pytest.mark.parametrize("output_size", [2, 3])
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


def test_a_good_region_patch_is_whole_voxels_paired_with_its_raw(tmp_path, janelia_raw):
    """A region's centre falls anywhere; the patch around it is still whole
    annotation voxels, with the raw read around that same patch."""
    path, dataset = _paired_dataset(
        tmp_path, janelia_raw, 2, rehearsal_fraction=1.0,
        good_regions=[{"id": "r", "offset_nm": [63.0] * 3, "shape_nm": [32.0] * 3}],  # 3 nm off the grid
    )
    zarr.open_group(path, mode="r+")["annotation/s0"][14, 14, 14] = 2  # something to train on, elsewhere
    raw, _, anchor = dataset()[0]
    assert float(anchor.sum()) == anchor.numel()
    # [63, 95) nm is annotation voxel 4.19 on (corner -4): the patch starts at voxel 4, raw 8.
    assert raw[0, :, 0, 0].tolist() == [9, 10, 11, 12]


@pytest.mark.parametrize("raw, norms, holds", [
    # Clipped to [0, dtype max]: signed and float raw lost their negative half.
    (np.random.default_rng(1).uniform(-1, 1, (32,) * 3).astype(np.float32), None, lambda p: (p < 0).mean() > 0.4),
    (np.random.default_rng(1).integers(-500, 500, (32,) * 3).astype(np.int16), None, lambda p: (p < 0).mean() > 0.4),
    # The noise is 1% of the normalizer's window, not of uint16's 0-65535.
    (np.full((32,) * 3, 2000, np.uint16), {"MinMaxNormalizer": {"min_value": 0, "max_value": 4000}},
     lambda p: 0.005 < (p - p.mean()).std() < 0.02 and 0 <= p.min() and p.max() <= 1),
    (np.random.default_rng(1).integers(200, 256, (32,) * 3).astype(np.uint8), None,
     lambda p: 0 <= p.min() and p.max() <= 255),
], ids=["float", "int16", "uint16 in a window", "uint8"])
def test_intensity_augmentation_stays_in_the_datas_own_range(annotation_volume, raw, norms, holds):
    dataset = annotation_volume(_labels(32, fg=(2, BLOCK)), raw=raw).dataset(
        augment=True, input_norm_config=norms, patches_per_epoch=4)
    assert all(holds(dataset[i][0].numpy()) for i in range(4))


def test_the_training_loader_keeps_its_workers_and_so_their_draws():
    """Workers re-spawned each epoch start from a fresh pickle of the dataset, its
    RNG unset: every epoch drew the same patches. The OOM fallback rebuilt the
    loader without persistent_workers, so every epoch after an OOM did."""
    data = TensorDataset(torch.rand(8, 1, 4, 4, 4), torch.ones(8, 1, 4, 4, 4))
    loader = make_training_loader(data, batch_size=4, num_workers=2)
    assert loader.persistent_workers and loader.multiprocessing_context is not None
    halved = rebuild_loader(loader, 2)
    assert (halved.batch_size, halved.num_workers, halved.persistent_workers, halved.pin_memory) == (2, 2, True, True)
    assert halved.multiprocessing_context is loader.multiprocessing_context and halved.dataset is data
    assert isinstance(rebuild_loader(DataLoader(data, shuffle=True), 2).sampler, RandomSampler)
    assert isinstance(rebuild_loader(DataLoader(data), 2).sampler, SequentialSampler)
