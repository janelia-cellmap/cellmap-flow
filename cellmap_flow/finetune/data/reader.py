"""Reading a finetune patch: the annotation, and the raw around the same place.

Annotation voxel v spans ``corner_nm + v * output_voxel_size`` (the corner
from session.volume.volume_corner_nm). A patch is whole annotation voxels,
``[c - size/2, c + size/2)`` around its centre c, and the raw is read around
that same patch, so every label is paired with the raw it covers.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import zarr

logger = logging.getLogger(__name__)


def input_normalizers(input_norm_config) -> list:
    """The normalizer objects of an ``input_norm`` config; [] when it builds none.

    The config arrives in either shape: the name-keyed dict a YAML gives
    (e.g. {"MinMaxNormalizer": {...}, "LambdaNormalizer": {...}}), or the
    list of {"name": ..., ...} dicts the dashboard POSTs -- a list on
    purpose, because jsonify sorts dict keys and the order of these steps
    changes what they compute.

    Do not coerce it with dict(). Over a list whose entries have exactly two
    keys it does not raise; it silently returns {"name": "expression"},
    which builds no normalizers at all, and the trainer would then train on
    raw uint8 while inference sees [-1, 1]. get_normalizations() understands
    both shapes, so it gets the config untouched.
    """
    if not input_norm_config:
        return []
    try:
        from cellmap_flow.norm.input_normalize import get_normalizations

        return get_normalizations(input_norm_config)
    except Exception as e:
        logger.error(
            f"Failed to build input normalizers from config "
            f"{input_norm_config!r}: {e}. Patches will be unnormalized."
        )
        return []


class PatchReader:
    """Reads the annotation and raw patches of one volume, and normalizes the raw.

    The raw is normalized here, with the session's own chain: the dashboard
    normalizes raw through its ``process_chain()`` before inference, but the
    trainer is a separate process whose chain is empty, so without this it
    would train on raw uint8 while inference sees normalized [-1, 1].

    The arrays are opened on first read. In training that is in a loader
    worker, after the dataset was pickled to it: no open zarr or
    tensorstore handle is ever pickled.
    """

    def __init__(
        self,
        volume_zarr_path: str,
        raw_dataset_path: str,
        input_size_voxels,
        output_size_voxels,
        input_voxel_size_nm,
        output_voxel_size_nm,
        corner_nm: np.ndarray,
        shape_voxels: np.ndarray,
        input_norm_config=None,
    ):
        self.volume_zarr_path = volume_zarr_path
        self.raw_dataset_path = raw_dataset_path
        self.input_size = np.array(input_size_voxels, dtype=int)
        self.output_size = np.array(output_size_voxels, dtype=int)
        self.input_voxel_size = np.array(input_voxel_size_nm, dtype=float)
        self.output_voxel_size = np.array(output_voxel_size_nm, dtype=float)
        self.corner_nm = corner_nm
        self.shape_voxels = shape_voxels
        self.input_norm_config = input_norm_config or {}
        self.normalizers = input_normalizers(self.input_norm_config)
        self._volume_arr = None
        self._raw_idi = None

        if not self.normalizers and self.input_norm_config:
            logger.warning(
                "input_norm_config provided but produced no normalizers; "
                "raw patches will be returned unnormalized."
            )
        if self.normalizers:
            logger.info(
                f"VirtualPatchDataset: applying {len(self.normalizers)} "
                f"input normalizer(s) per patch: "
                f"{[type(n).__name__ for n in self.normalizers]}"
            )
        else:
            logger.warning(
                "VirtualPatchDataset: no input normalizers configured. "
                "Raw patches will be returned in their native dtype/range. "
                "If inference normalizes to [-1, 1] (typical), the trained "
                "model will see different inputs at train vs inference time."
            )

    def snap(self, centre_voxels: np.ndarray) -> np.ndarray:
        """The nearest centre whose patch, ``[c - size/2, c + size/2)``, is whole voxels.

        Without it the annotation patch would be cut at int(c - size/2)
        while the raw was read around c itself: half a voxel apart for an
        odd output size, and for a good region, whose centre falls anywhere.
        """
        half = self.output_size / 2
        return np.floor(centre_voxels - half + 0.5) + half

    def annotation(self, centre_voxels: np.ndarray) -> np.ndarray:
        """The annotation patch around a snapped centre, in the volume's own dtype.

        Out-of-bounds voxels are 0, unannotated, which the trainer's loss
        masks out when ``mask_unannotated`` is on.

        The dtype is the volume's because an instance-correction volume is
        uint16 or uint32 (instance id + 1): a uint8 patch wrapped its ids
        modulo 256, so instance 255 became unannotated, 256 background, and
        257 merged with instance 1. The dataset casts the patch to float32,
        which holds every id below 2**24 exactly.
        """
        out_size = self.output_size
        lo = (centre_voxels - out_size / 2).astype(int)
        hi = lo + out_size

        clip_lo = np.maximum(lo, 0)
        clip_hi = np.minimum(hi, self.shape_voxels)
        valid = np.all(clip_hi > clip_lo)

        if self._volume_arr is None:
            self._volume_arr = zarr.open(
                os.path.join(self.volume_zarr_path, "annotation", "s0"), mode="r"
            )
        patch = np.zeros(out_size, dtype=self._volume_arr.dtype)
        if valid:
            src_slices = tuple(slice(int(c), int(d)) for c, d in zip(clip_lo, clip_hi))
            dst_slices = tuple(
                slice(int(c - l), int(d - l))
                for c, d, l in zip(clip_lo, clip_hi, lo)
            )
            patch[dst_slices] = self._volume_arr[src_slices]
        return patch

    def raw(self, centre_voxels: np.ndarray) -> np.ndarray:
        """The ``input_size`` raw patch centred on the annotation patch, as stored.

        Read with ``normalize=False``, so no process-wide norms apply;
        ``normalize`` applies the session's own afterwards.

        The patch is a box of raw voxels: ``input_size`` of them, from the
        one the patch's lower edge falls in (Grid.world_to_box's rule). It
        used to go through a whole-nm Roi, whose Coordinate truncated the
        edge and the shape, so at 5.24 nm or 10.48 nm the patch came out a
        voxel short and started a voxel early. Where the edge is whole nm
        on a whole-nm grid (the 8 and 16 nm levels) both give the same box.
        """
        from cellmap_flow.image_data_interface import ImageDataInterface
        from cellmap_flow.io.geometry import Box
        from cellmap_flow.io.metadata import snap_integral
        from cellmap_flow.io.source import read_padded

        if self._raw_idi is None:
            self._raw_idi = ImageDataInterface(
                self.raw_dataset_path,
                voxel_size=self.input_voxel_size,
                normalize=False,
            )
        raw_voxel_size = np.asarray(self._raw_idi.voxel_size, dtype=float)
        raw_corner_nm = np.asarray(self._raw_idi.offset, dtype=float)
        centre_nm = self.corner_nm + centre_voxels * self.output_voxel_size
        lower_edge_nm = centre_nm - self.input_size * raw_voxel_size / 2
        # snap_integral: float noise (39.9999999 voxels) counts as the whole
        # number it is.
        begin = np.floor(snap_integral((lower_edge_nm - raw_corner_nm) / raw_voxel_size))
        box = Box(tuple(int(b) for b in begin), tuple(int(s) for s in self.input_size))
        # .ts is the selected channel, unnormalized (normalize=False); out of
        # the array is padded with 0, as to_ndarray_ts pads.
        return read_padded(self._raw_idi.ts, box)

    def normalize(self, raw: np.ndarray) -> np.ndarray:
        """``raw`` through the session's normalizers, in order, as apply_norms() does for inference."""
        for norm in self.normalizers:
            raw = norm(raw)
        return raw
