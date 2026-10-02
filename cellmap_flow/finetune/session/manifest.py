"""The files a session is trained from, besides the volume itself.

``_virtual_sources.json`` in ``corrections/`` points the trainer at the
annotation volume and carries the patch geometry and normalization;
``good_regions.json``, one level up, lists the regions the user marked as
already right. ``has_painted_annotations`` says whether a volume holds
scribbles as well as imported crops.
"""

import json
import logging
import os
from typing import Optional

import numpy as np
import zarr

from cellmap_flow.io.geometry import list_populated_chunks

logger = logging.getLogger(__name__)

VIRTUAL_MANIFEST_FILENAME = "_virtual_sources.json"
GOOD_REGIONS_FILENAME = "good_regions.json"



def write_manifest(corrections_dir: str, manifest: dict) -> str:
    """Persist a manifest sentinel that ``create_dataloader`` looks for."""
    os.makedirs(corrections_dir, exist_ok=True)
    path = os.path.join(corrections_dir, VIRTUAL_MANIFEST_FILENAME)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return path


def read_manifest(corrections_dir: str) -> Optional[dict]:
    """Return the manifest if present, else ``None``."""
    path = os.path.join(corrections_dir, VIRTUAL_MANIFEST_FILENAME)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def good_regions_path(corrections_dir: Optional[str]) -> Optional[str]:
    """Where the good regions of the session owning ``corrections_dir`` live:
    beside it, in the session directory. None without a corrections dir."""
    if not corrections_dir:
        return None
    return os.path.join(os.path.dirname(str(corrections_dir).rstrip("/")), GOOD_REGIONS_FILENAME)


def load_good_regions(path: Optional[str]) -> list:
    """The regions stored at ``path``; [] when there are none or they can't be read."""
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            regions = json.load(f)
    except (OSError, ValueError) as e:
        logger.warning(f"Could not read good regions from {path}: {e}")
        return []
    if not isinstance(regions, list):
        logger.warning(f"Ignoring good regions at {path}: expected a list.")
        return []
    return regions


def save_good_regions(path: str, regions: list) -> bool:
    """Replace the regions at ``path`` atomically; False if it could not be written."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w") as f:
            json.dump(regions, f, indent=2)
        os.replace(tmp, path)
        return True
    except OSError as e:
        logger.error(f"Could not save good regions to {path}: {e}")
        return False


def load_good_regions_for(corrections_dir: Optional[str]) -> list:
    """Read the session's good regions, if any were marked.

    Deliberately read here rather than snapshotted into the manifest: the
    manifest is written when crops are imported, and regions get marked
    afterwards, for as long as the user keeps browsing. Reading at training
    time means the run uses every region marked up to the moment it started.
    """
    path = good_regions_path(corrections_dir)
    regions = load_good_regions(path)
    if regions:
        logger.info(f"Loaded {len(regions)} good region(s) from {path}")
    return regions


def voxels_inside_any_bbox(
    voxels: np.ndarray, bbox_offsets: np.ndarray, bbox_ends: np.ndarray
) -> np.ndarray:
    """Return a boolean mask: ``True`` where ``voxels[i]`` lies inside any
    ``[bbox_offsets[j], bbox_ends[j])`` half-open box.

    voxels: (N, 3) int. bbox_offsets, bbox_ends: (M, 3) int. Vectorized
    over both: builds an (N, M) inside-test matrix and reduces along M.
    For typical M ~ 1-10 the temporary stays small.
    """
    if voxels.shape[0] == 0 or bbox_offsets.shape[0] == 0:
        return np.zeros(voxels.shape[0], dtype=bool)
    # (N, M, 3) broadcast: voxels[:, None, :] vs bbox_offsets[None, :, :]
    ge = np.all(voxels[:, None, :] >= bbox_offsets[None, :, :], axis=-1)
    lt = np.all(voxels[:, None, :] < bbox_ends[None, :, :], axis=-1)
    return np.any(ge & lt, axis=-1)


def has_painted_annotations(volume_zarr_path: str) -> bool:
    """Whether the volume holds annotations outside its imported crops.

    Those are painted: scribbles, sparse by construction, with unannotated
    voxels all around them. Chunks entirely inside a crop are not read.
    """
    s0_path = os.path.join(volume_zarr_path, "annotation", "s0")
    try:
        with open(os.path.join(volume_zarr_path, ".zattrs")) as f:
            imported = json.load(f).get("imported_crops", []) or []
        arr = zarr.open(s0_path, mode="r")
        chunks = list_populated_chunks(s0_path)
    except (OSError, ValueError, KeyError) as e:
        logger.debug(f"Could not look for painted annotations in {volume_zarr_path}: {e}")
        return False
    if imported:
        bbox_offsets = np.array([c["annotation_offset_voxels"] for c in imported], dtype=np.int64)
        bbox_ends = bbox_offsets + np.array([c["annotation_shape_voxels"] for c in imported], dtype=np.int64)
    else:
        bbox_offsets = bbox_ends = np.zeros((0, 3), dtype=np.int64)
    chunk_shape = np.array(arr.chunks, dtype=np.int64)
    for index in chunks:
        origin = np.array(index, dtype=np.int64) * chunk_shape
        end = origin + chunk_shape
        if bbox_offsets.shape[0] and np.any(
            np.all(origin >= bbox_offsets, axis=1) & np.all(end <= bbox_ends, axis=1)
        ):
            continue  # all of it inside one crop
        annotated = np.argwhere(arr.blocks[index] >= 1).astype(np.int64) + origin
        if not annotated.size:
            continue
        if not bbox_offsets.shape[0] or not voxels_inside_any_bbox(annotated, bbox_offsets, bbox_ends).all():
            return True
    return False
