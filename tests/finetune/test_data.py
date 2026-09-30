"""cellmap_flow.finetune.data: VirtualPatchDataset through the patches it draws
(where they are centred, what they hold, what good regions add), and the
loaders it is read through."""

import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import zarr
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from cellmap_flow.finetune.session.volume import create_volume_zarr, plan_volume
from cellmap_flow.finetune.data import (
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


@pytest.mark.parametrize("norms", [pytest.param(NORMS, id="keyed by name, as a yaml has it"),
                                   pytest.param([{"name": k, **v} for k, v in NORMS.items()],
                                                id="a list, as the dashboard sends it")])
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
FAR_APART = _labels(64, fg=(2, np.s_[2:4, 2:4, 2:4]), fix=(1, np.s_[50:52, 50:52, 50:52]))
SCRIBBLE = _labels(32, s=(2, np.s_[2:6, 2:6, 2:6]))


def _drawn(dataset, n=16):
    """The annotation patches of ``n`` draws."""
    return [dataset[i][1] for i in range(n)]


@pytest.mark.parametrize("labels, patches", [
    pytest.param(_labels(48, a=(2, np.s_[1, 1, 1]), b=(2, np.s_[17, 17, 17]), c=(2, np.s_[33, 33, 33])), 3,
                 id="three chunks of foreground"),
    pytest.param(FAR_APART, 2, id="a background fix counts too"),
    # A session that painted only background raised "no foreground voxels".
    pytest.param(_labels(32, fix=(1, np.s_[20:24, 20:24, 20:24])), 1, id="background only"),
])
def test_an_epoch_is_a_patch_per_annotated_chunk(annotation_volume, labels, patches):
    """patches_per_epoch None, the default: what was painted, about once an epoch."""
    assert len(annotation_volume(labels).dataset()) == patches


def test_a_background_fix_far_from_any_foreground_is_drawn(annotation_volume):
    """Painting background where the model hallucinates: only foreground was a
    patch centre, so a fix further than half a patch from any was never in one."""
    dataset = annotation_volume(FAR_APART).dataset()
    assert any(1 in ann and 2 not in ann for ann in _drawn(dataset))


@pytest.mark.parametrize("labels, label", [
    pytest.param(_labels(32, crop=(1, np.s_[0:16, 0:16, 0:16]), fg=(2, np.s_[4:8, 4:8, 4:8])), 2,
                 id="around its foreground, not its background"),
    pytest.param(_labels(32, crop=(1, np.s_[0:16, 0:16, 0:16])), 1, id="a background crop trains too"),
])
def test_an_imported_crop_is_drawn_around_its_annotation(annotation_volume, labels, label):
    dataset = annotation_volume(labels, crops=[CROP16]).dataset()
    assert all(label in ann for ann in _drawn(dataset))


def test_a_volume_with_nothing_annotated_is_refused(annotation_volume):
    with pytest.raises(ValueError):
        annotation_volume(_labels(32)).dataset()


def test_the_scribbles_get_half_the_patches_beside_a_big_crop(annotation_volume):
    """Voxel-uniform sampling drew all but a handful of patches from a big
    imported crop, and ignored the scribbles painted beside it."""
    labels = _labels(64, crop=(3, np.s_[0:32, 0:32, 0:32]), scribble=(2, np.s_[40:42, 40:42, 40:42]))
    crop = {"annotation_offset_voxels": [0, 0, 0], "annotation_shape_voxels": [32, 32, 32]}
    dataset = annotation_volume(labels, crops=[crop]).dataset(patches_per_epoch=64)
    assert 16 < sum(2 in ann for ann in _drawn(dataset, 64)) < 48


def test_a_pool_ratio_never_draws_from_an_empty_pool(annotation_volume):
    """Scribbles only, and a ratio asking for half the patches from crops: all come from the scribbles."""
    dataset = annotation_volume(SCRIBBLE).dataset(dense_to_sparse_ratio=0.5, patches_per_epoch=4)
    assert all(2 in ann for ann in _drawn(dataset, 4))


def _region(centre_voxel, id="r1"):
    """A 4-voxel box around ``centre_voxel``, the way the dashboard saves one."""
    return {"id": id, "label": "good", "offset_nm": [centre_voxel * 16.0 - 32.0] * 3, "shape_nm": [64.0] * 3}


@pytest.mark.parametrize("regions, fraction", [
    pytest.param([], None, id="no regions"),  # the (raw, annotation) of every existing run
    pytest.param([_region(24)], 0.0, id="rehearsal switched off for this run"),
    pytest.param([_region(9999, "elsewhere")], 1.0, id="a region outside the volume"),  # it would anchor zeros
    pytest.param([{"id": "junk"}], 1.0, id="a malformed region"),  # dropped, not fatal
])
def test_without_a_usable_good_region_patches_carry_no_anchor(annotation_volume, regions, fraction):
    dataset = annotation_volume(SCRIBBLE).dataset(good_regions=regions, rehearsal_fraction=fraction)
    assert not dataset.emits_anchor and len(dataset[0]) == 2


def test_good_regions_get_their_share_of_the_patches(annotation_volume):
    """A third pool: the other two index on foreground, which a good region has
    none of, so marking one changed nothing about a run."""
    dataset = annotation_volume(SCRIBBLE).dataset(good_regions=[_region(24)], rehearsal_fraction=0.5,
                                                  patches_per_epoch=64)
    assert 16 < sum(bool(dataset[i][2].any()) for i in range(64)) < 48


def test_a_rehearsal_patch_anchors_every_voxel_that_was_not_painted(annotation_volume):
    """It holds the student to the teacher there; a scribble inside the region
    wins over the anchor, so a mistake noticed there can still be painted over."""
    labels = _labels(32, s=(2, np.s_[2:6, 2:6, 2:6]), in_the_region=(2, np.s_[24, 24, 24]))
    dataset = annotation_volume(labels).dataset(good_regions=[_region(24)], rehearsal_fraction=1.0,
                                                patches_per_epoch=8)
    for _, ann, anchor in (dataset[i] for i in range(8)):
        assert ann.max() == 2 and torch.equal(anchor, (ann == 0).float())


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
    pytest.param(np.random.default_rng(1).uniform(-1, 1, (32,) * 3).astype(np.float32), None,
                 lambda patch: (patch < 0).mean() > 0.4, id="float raw keeps its negative half"),
    pytest.param(np.random.default_rng(1).integers(-500, 500, (32,) * 3).astype(np.int16), None,
                 lambda patch: (patch < 0).mean() > 0.4, id="signed raw keeps its negative values"),
    # The noise was 1% of uint16's 0-65535, a sixth of data using 0-4000.
    pytest.param(np.full((32,) * 3, 2000, np.uint16), {"MinMaxNormalizer": {"min_value": 0, "max_value": 4000}},
                 lambda patch: 0.005 < (patch - patch.mean()).std() < 0.02 and 0 <= patch.min() <= patch.max() <= 1,
                 id="noise is 1% of the normalizer's window"),
    pytest.param(np.random.default_rng(1).integers(200, 256, (32,) * 3).astype(np.uint8), None,
                 lambda patch: 0 <= patch.min() and patch.max() <= 255, id="uint8 stays within 0-255"),
])
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
