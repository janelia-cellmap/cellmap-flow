"""The patches a finetune trains on, pinned draw for draw.

One synthetic session holds everything the dataset samples from: an
imported crop (the dense pool, its foreground), scribbles of foreground and
of background beside it and a chunk painted only background (the sparse
pool), good regions (one usable, one outside the volume, one malformed), and
chunks written but left unannotated. Two configurations of it are drawn from
by two loader workers, 64 draws each:

- ``plain``: no augmentation, no normalization, the pool and rehearsal
  shares left to their defaults. The raw holds each voxel's own index, so a
  draw is recorded as the raw voxel at its patch's centre: where it was cut.
- ``augmented``: flips, rotations, brightness and noise, the dashboard's
  normalization chain, and explicit pool and rehearsal shares.

Each worker gets the dataset the way a spawned one does, as a pickle, and
draws in this process with torch's worker info set to its id (real spawned
workers cost seconds each, and hang in some sandboxes).

Pinned: the default epoch length, the patch shapes, the centre of every
plain draw and whether it was a rehearsal draw, and a hash of every tensor
of every draw in order, per worker. The draws depend on numpy's generator
and float32 arithmetic, not on torch, so one record holds for every platform.
"""

import hashlib
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch.utils.data._utils.worker as torch_worker

SIZE = 48  # annotation voxels per side at 16 nm, in 16^3 chunks
DRAWS = 64  # per worker


def _labels():
    """0 unannotated, 1 background, 2+ foreground."""
    labels = np.zeros((SIZE,) * 3, np.uint8)
    labels[0:16, 0:16, 0:16] = 1  # the imported crop: background ...
    labels[4:10, 4:10, 4:10] = 3  # ... around its foreground
    labels[30, 30, 20:28] = 2  # a foreground scribble
    labels[30, 33, 18:30] = 1  # a background scribble beside it
    labels[40:42, 4:6, 40:42] = 1  # a chunk painted only background
    return labels


CROP = {"annotation_offset_voxels": [0, 0, 0], "annotation_shape_voxels": [16, 16, 16]}
REGIONS = [
    {"id": "good", "offset_nm": [24 * 16.0 - 32.0] * 3, "shape_nm": [64.0] * 3},
    {"id": "elsewhere", "offset_nm": [99999.0] * 3, "shape_nm": [64.0] * 3},
    {"id": "junk"},
]
NORMS = [{"name": "MinMaxNormalizer", "min_value": 0, "max_value": 255, "invert": False},
         {"name": "LambdaNormalizer", "expression": "x*2-1"}]

# config: (raw, dataset settings)
CONFIGS = {
    "plain": ("index", dict(good_regions=REGIONS)),
    "augmented": ("noise", dict(good_regions=REGIONS, augment=True, input_norm_config=NORMS,
                                dense_to_sparse_ratio=0.7, rehearsal_fraction=0.4)),
}


def _raw(kind):
    if kind == "index":  # each voxel's own index, 1-based (0 is padding), exact in float32
        return np.arange(1, SIZE**3 + 1, dtype=np.float32).reshape((SIZE,) * 3)
    return np.random.default_rng(7).integers(0, 256, (SIZE,) * 3, dtype=np.uint8)


def _worker_draws(dataset, worker_id, monkeypatch):
    """What loader worker ``worker_id`` draws: its own pickled copy, read with its id set."""
    copy = pickle.loads(pickle.dumps(dataset))
    monkeypatch.setattr(torch_worker, "_worker_info", SimpleNamespace(id=worker_id, num_workers=2))
    draws = [copy[i] for i in range(worker_id, 2 * DRAWS, 2)]
    monkeypatch.setattr(torch_worker, "_worker_info", None)
    return draws


def _hash(draws):
    h = hashlib.sha256()
    for draw in draws:
        for t in draw:
            h.update(f"{t.dtype}:{tuple(t.shape)}".encode())
            h.update(t.numpy().tobytes())
    return h.hexdigest()[:16]


def _where(raw, anchor):
    """The raw voxel at the patch's centre as z,y,x, with R for a rehearsal draw."""
    centre = int(raw[(0, *(n // 2 for n in raw.shape[1:]))]) - 1
    zyx = ",".join(str(int(v)) for v in np.unravel_index(centre, (SIZE,) * 3))
    return zyx + (" R" if anchor.any() else "")


# config: what it draws with the pools built in chunk order, the same on every
# machine; before that, the pools followed os.listdir and differed between filesystems.
RECORDED = {
    "plain": dict(
        epoch=4,
        draws=["6a3506e1465ecfa5", "52566bcdbfe22ee2"],
        where=[
            [
                "4,8,3", "41,4,41", "9,8,7", "3,6,4", "7,4,5", "30,30,27", "42,4,42", "30,30,22",
                "10,6,3", "6,9,6", "25,25,25 R", "9,7,4", "25,25,25 R", "9,8,6", "31,29,21", "25,25,25 R",
                "25,25,25 R", "7,7,8", "5,4,5", "7,9,6", "31,32,25", "6,6,5", "40,4,40", "29,31,24",
                "9,10,5", "8,5,5", "8,8,9", "7,8,8", "25,25,25 R", "6,9,3", "7,5,6", "5,7,5",
                "6,7,4", "25,25,25 R", "31,29,26", "7,10,8", "4,5,9", "8,8,7", "25,25,25 R", "10,3,8",
                "5,7,4", "31,30,19", "5,9,6", "8,9,7", "31,32,28", "31,30,25", "29,29,25", "8,5,7",
                "30,31,26", "25,25,25 R", "31,31,25", "7,8,9", "25,25,25 R", "30,30,20", "25,25,25 R", "9,7,7",
                "31,29,25", "6,6,5", "29,31,21", "6,9,7", "25,25,25 R", "9,9,7", "5,10,6", "41,5,41",
            ],
            [
                "29,34,25", "29,29,20", "25,25,25 R", "25,25,25 R", "25,25,25 R", "41,4,40", "10,4,5", "29,32,19",
                "25,25,25 R", "31,32,25", "25,25,25 R", "29,30,26", "31,29,27", "29,33,21", "41,6,40", "3,3,9",
                "5,6,7", "8,6,9", "9,4,6", "5,6,7", "29,29,25", "25,25,25 R", "41,4,40", "31,32,21",
                "25,25,25 R", "5,8,4", "6,9,6", "5,6,5", "30,33,19", "25,25,25 R", "30,34,19", "25,25,25 R",
                "25,25,25 R", "30,31,26", "7,4,8", "25,25,25 R", "30,30,21", "30,30,21", "31,31,20", "25,25,25 R",
                "8,5,7", "6,4,9", "25,25,25 R", "25,25,25 R", "9,6,6", "31,29,25", "41,3,42", "30,30,25",
                "25,25,25 R", "25,25,25 R", "25,25,25 R", "25,25,25 R", "25,25,25 R", "41,5,42", "25,25,25 R", "4,6,6",
                "9,5,9", "3,9,6", "25,25,25 R", "25,25,25 R", "29,29,26", "29,29,21", "25,25,25 R", "25,25,25 R",
            ],
        ],
    ),
    "augmented": dict(epoch=4, draws=["2fdd11bbffae032f", "6c6f0a6079c3f3c3"]),
}


@pytest.mark.parametrize("config", CONFIGS)
def test_the_dataset_draws_what_it_always_did(annotation_volume, monkeypatch, config):
    raw, settings = CONFIGS[config]
    volume = annotation_volume(_labels(), raw=_raw(raw), crops=[CROP])
    epoch = len(volume.dataset(**settings))  # patches_per_epoch None: one per annotated chunk

    dataset = volume.dataset(patches_per_epoch=2 * DRAWS, **settings)
    workers = [_worker_draws(dataset, k, monkeypatch) for k in range(2)]
    shapes = {tuple(tuple(t.shape) for t in draw) for draws in workers for draw in draws}
    assert shapes == {((1, 8, 8, 8), (1, 4, 4, 4), (1, 4, 4, 4))}  # good regions: (raw, annotation, anchor)

    got = {"epoch": epoch, "draws": [_hash(draws) for draws in workers]}
    if config == "plain":
        got["where"] = [[_where(raw, anchor) for raw, _, anchor in draws] for draws in workers]
    assert got == RECORDED[config]
