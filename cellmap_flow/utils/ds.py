# %%
import json
import logging
import os
import re
from typing import Optional, Sequence, Union

import h5py
import numpy as np
import s3fs
import tensorstore as ts
import zarr
from funlib.geometry import Coordinate, Roi
from skimage.measure import block_reduce
from zarr.n5 import N5FSStore

from cellmap_flow.globals import g


def generate_singlescale_metadata(
    arr_name: str,
    voxel_size: list,
    translation: list,
    units: str,
    axes: list,
):
    z_attrs: dict = {"multiscales": [{}]}

    # Create axes with proper types - channel axis should have type "channel"
    axes_list = []
    for axis, unit in zip(axes, units):
        if axis in ["c", "c^"]:
            axes_list.append({"name": axis, "type": "channel"})
        else:
            axes_list.append({"name": axis, "type": "space", "unit": unit})

    z_attrs["multiscales"][0]["axes"] = axes_list

    # Set coordinateTransformations scale to match dimensionality
    scale_transform = [1.0] * len(axes)
    z_attrs["multiscales"][0]["coordinateTransformations"] = [
        {"scale": scale_transform, "type": "scale"}
    ]

    z_attrs["multiscales"][0]["datasets"] = [
        {
            "coordinateTransformations": [
                {"scale": list(voxel_size), "type": "scale"},
                {"translation": list(translation), "type": "translation"},
            ],
            "path": arr_name,
        }
    ]

    z_attrs["multiscales"][0]["name"] = ""
    z_attrs["multiscales"][0]["version"] = "0.4"

    return z_attrs


def get_scale_info(zarr_grp):
    attrs = zarr_grp.attrs
    ms = attrs["multiscales"][0]

    # Determine which axes are spatial so we can skip channel axes
    axes = ms.get("axes", [])
    spatial_indices = [i for i, a in enumerate(axes) if a.get("type") == "space"]
    # If no axes metadata, assume all dimensions are spatial
    if not spatial_indices:
        spatial_indices = None

    resolutions = {}
    offsets = {}
    shapes = {}
    for scale in ms["datasets"]:
        transforms = scale["coordinateTransformations"]
        full_res = transforms[0]["scale"]
        # Translation is optional (e.g. s0 often has only scale)
        full_translation = next(
            (t["translation"] for t in transforms if t["type"] == "translation"),
            [0.0] * len(full_res),
        )
        full_shape = zarr_grp[scale["path"]].shape

        if spatial_indices is not None:
            resolutions[scale["path"]] = [full_res[i] for i in spatial_indices]
            offsets[scale["path"]] = [full_translation[i] for i in spatial_indices]
            shapes[scale["path"]] = tuple(full_shape[i] for i in spatial_indices)
        else:
            resolutions[scale["path"]] = full_res
            offsets[scale["path"]] = full_translation
            shapes[scale["path"]] = full_shape
    return offsets, resolutions, shapes


def get_array_path_if_needed(zarr_grp_path, target_resolution):
    try:
        _ = get_ds_info(zarr_grp_path)
        # If successful, it's a dataset path
        return zarr_grp_path
    except Exception as e:
        if ".zarr" not in zarr_grp_path and not _is_zarr_container(zarr_grp_path):
            raise RuntimeError(
                f"Failed to open dataset at {zarr_grp_path}: {e}\n Multiscale is only supported for zarr groups. Please provide a valid dataset path."
            )
        # Otherwise, it's a group path; find the appropriate scale
        target_scale, _, _ = find_target_scale(zarr_grp_path, target_resolution)
        return _join_path(zarr_grp_path, target_scale)


def find_target_scale(zarr_grp_path, target_resolution):
    try:
        zarr_grp = _open_zarr(zarr_grp_path, mode="r")
    except Exception as e:
        raise RuntimeError(f"Failed to open zarr group at {zarr_grp_path}: {e}")
    offsets, resolutions, shapes = get_scale_info(zarr_grp)
    target_scale = None
    for scale, res in resolutions.items():
        if Coordinate(res) == Coordinate(target_resolution):
            target_scale = scale
            break
    if target_scale is None:
        msg = f"Zarr {zarr_grp.store.path}, {zarr_grp.path} does not contain array with sampling {target_resolution}"
        raise ValueError(msg)
    return target_scale, offsets[target_scale], shapes[target_scale]


def find_closest_scale(zarr_grp_path, target_resolution):
    zarr_grp = _open_zarr(zarr_grp_path, mode="r")
    offsets, resolutions, shapes = get_scale_info(zarr_grp)
    target_scale = None
    last_scale = None
    for scale, res in resolutions.items():
        if last_scale is None:
            last_scale = scale
        if Coordinate(res) == Coordinate(target_resolution):
            target_scale = scale
            break
        elif any((r > t for r, t in zip(res, target_resolution))):
            target_scale = last_scale
            break
        last_scale = scale
    if target_scale is None:
        target_scale = last_scale
    return target_scale, offsets[target_scale], shapes[target_scale]


# Ensure tensorstore does not attempt to use GCE credentials
os.environ["GCE_METADATA_ROOT"] = "metadata.google.internal.invalid"

# Much below taken from flyemflows: https://github.com/janelia-flyem/flyemflows/blob/master/flyemflows/util/util.py
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def ends_with_scale(string):
    pattern = (
        r"s\d+$"  # Matches 's' followed by one or more digits at the end of the string
    )
    return bool(re.search(pattern, string))


def _normalize_path(path: str) -> str:
    """Remove shell-escape backslashes from a filesystem path.

    Users often copy-paste paths from a terminal where spaces are escaped
    (e.g. ``/path/to/file\\ name.zarr``).  YAML preserves the literal
    backslashes, but the filesystem expects plain spaces.
    """
    if _is_remote_path(path):
        return path
    return path.replace("\\ ", " ")


def _is_remote_path(path: str) -> bool:
    return path.startswith("http://") or path.startswith("https://")


# --- Zarr v3 read support -------------------------------------------------
#
# cellmap-flow's metadata-introspection path historically used zarr-python 2,
# which only understands the v2 spec (.zarray, .zgroup, .zattrs). v3 stores
# carry a single zarr.json per node and zarr-python 2 fails to read them.
#
# We can't install zarr-python 3 in this environment because funlib.persistence
# pins zarr<3 (its block-scheduling layer hasn't been ported). Instead, the
# helpers below introduce a thin read-only proxy that parses zarr.json
# directly. Data reads still go through tensorstore (which supports v3 +
# sharding natively); only the metadata path needed work.


def _zarr_format(path: str) -> Optional[int]:
    """Return ``2``, ``3``, or ``None`` based on which zarr-spec files exist
    at ``path``.

    Prefers a valid v3 ``zarr.json`` (with ``zarr_format=3``) over v2
    markers when both are present. zarr-python 2's ``zarr.open(path)``
    defaults to mode ``"a"``, which silently writes a stray ``.zgroup``
    against a v3 store; honoring the explicit v3 metadata is the friendlier
    semantics.

    Remote (http/https) paths are reported as v2 unconditionally — current
    callers don't have any remote v3 stores, and adding remote v3 detection
    would require an extra fetch we don't need yet.
    """
    if _is_remote_path(path):
        return 2
    path = _normalize_path(path)
    if not os.path.isdir(path):
        return None
    zj = os.path.join(path, "zarr.json")
    if os.path.exists(zj):
        try:
            with open(zj) as f:
                meta = json.load(f)
            if meta.get("zarr_format") == 3:
                return 3
        except (OSError, json.JSONDecodeError):
            pass
    if (
        os.path.exists(os.path.join(path, ".zarray"))
        or os.path.exists(os.path.join(path, ".zgroup"))
    ):
        return 2
    return None


class ZarrV3Node:
    """Read-only metadata proxy over a single v3 ``zarr.json``.

    Duck-types the subset of zarr-python 2's Group/Array interface that
    cellmap-flow's read paths actually use: ``attrs``, ``shape``, ``chunks``,
    ``dtype``, ``keys()``, ``__getitem__``, ``path``. Data access still goes
    through tensorstore via :func:`open_ds_tensorstore` — this proxy is
    metadata-only.

    For sharded arrays, ``chunks`` returns the *inner* chunk shape (the
    compression/IO unit), not the outer shard shape. The outer shard shape
    is available via :attr:`shard_shape` when needed.
    """

    def __init__(self, path: str):
        path = _normalize_path(path)
        zj = os.path.join(path, "zarr.json")
        if not os.path.exists(zj):
            raise FileNotFoundError(f"No zarr.json at {path}")
        with open(zj) as f:
            self._meta = json.load(f)
        if self._meta.get("zarr_format") != 3:
            raise ValueError(
                f"{zj} is not zarr_format=3 (got {self._meta.get('zarr_format')!r})"
            )
        self.path = path

    @property
    def is_group(self) -> bool:
        return self._meta.get("node_type") == "group"

    @property
    def attrs(self) -> dict:
        return self._meta.get("attributes", {})

    @property
    def shape(self) -> tuple:
        if self.is_group:
            raise AttributeError(f"{self.path} is a v3 group; has no shape")
        return tuple(self._meta["shape"])

    @property
    def chunks(self) -> tuple:
        """Inner chunk shape (compression/IO unit).

        For sharded arrays, returns the codec's inner chunk_shape, NOT the
        outer chunk_grid shape (which is the shard size). For unsharded
        arrays, returns the chunk_grid shape directly.
        """
        if self.is_group:
            raise AttributeError(f"{self.path} is a v3 group; has no chunks")
        for codec in self._meta.get("codecs", []):
            if codec.get("name") == "sharding_indexed":
                return tuple(codec["configuration"]["chunk_shape"])
        return tuple(self._meta["chunk_grid"]["configuration"]["chunk_shape"])

    @property
    def shard_shape(self) -> Optional[tuple]:
        """Outer shard shape for sharded arrays; ``None`` for unsharded."""
        if self.is_group:
            return None
        for codec in self._meta.get("codecs", []):
            if codec.get("name") == "sharding_indexed":
                return tuple(self._meta["chunk_grid"]["configuration"]["chunk_shape"])
        return None

    @property
    def dtype(self):
        return np.dtype(self._meta["data_type"])

    @property
    def name(self) -> str:
        return os.path.basename(os.path.normpath(self.path))

    def keys(self):
        """List child node names (groups only).

        A child is a subdirectory containing a ``zarr.json``. Hidden entries
        and non-zarr subdirs are skipped.
        """
        if not self.is_group:
            raise AttributeError(f"{self.path} is a v3 array; has no keys()")
        out = []
        try:
            entries = sorted(os.listdir(self.path))
        except OSError:
            return out
        for entry in entries:
            child = os.path.join(self.path, entry)
            if os.path.isdir(child) and os.path.exists(
                os.path.join(child, "zarr.json")
            ):
                out.append(entry)
        return out

    def __getitem__(self, key: str) -> "ZarrV3Node":
        child_path = os.path.join(self.path, key)
        if not os.path.exists(os.path.join(child_path, "zarr.json")):
            raise KeyError(key)
        return ZarrV3Node(child_path)

    def __repr__(self) -> str:
        kind = "group" if self.is_group else "array"
        return f"<ZarrV3Node {kind} at {self.path}>"


def _is_zarr_group(node) -> bool:
    """Structural check accepting either a zarr-python 2 Group or a
    :class:`ZarrV3Node` group. Use in place of
    ``isinstance(node, zarr.hierarchy.Group)``."""
    if isinstance(node, ZarrV3Node):
        return node.is_group
    return isinstance(node, zarr.hierarchy.Group)


def _is_zarr_root(node) -> bool:
    """Structural check for the top of a zarr store.

    For v2, the root is signalled by an empty path within the store.
    For v3, the root is when the proxy's parent directory is not itself a
    v3 container.
    """
    if isinstance(node, ZarrV3Node):
        parent = os.path.dirname(os.path.normpath(node.path))
        return _zarr_format(parent) != 3
    return node.path == ""


def _open_zarr(path, mode="r"):
    """Open a zarr v2 or v3 dataset.

    Local paths dispatch on which spec marker is present:
    - ``.zarray`` / ``.zgroup`` → zarr-python 2
    - ``zarr.json`` (v3) → :class:`ZarrV3Node` (read-only proxy)

    Remote (http/https) paths always go through zarr-python 2 + fsspec.
    """
    path = _normalize_path(path)
    if _is_remote_path(path):
        import fsspec

        return zarr.open(fsspec.get_mapper(path), mode=mode)
    if _zarr_format(path) == 3:
        if mode != "r":
            raise NotImplementedError(
                f"v3 zarr writes via _open_zarr not supported (path={path}, mode={mode!r})"
            )
        return ZarrV3Node(path)
    return zarr.open(path, mode=mode)


def _join_path(base, *parts):
    """Join path components, handling URLs correctly."""
    if _is_remote_path(base):
        return "/".join([base.rstrip("/"), *parts])
    return os.path.join(base, *parts)


def _is_zarr_container(path: str) -> bool:
    """Check if a local path is a zarr container by looking for zarr metadata files.

    Works for zarr directories that don't have a .zarr extension. Detects
    both v2 (``.zarray`` / ``.zgroup`` / ``.zattrs``) and v3 (``zarr.json``).
    """
    if _is_remote_path(path):
        return False
    return os.path.isdir(path) and (
        os.path.exists(os.path.join(path, ".zgroup"))
        or os.path.exists(os.path.join(path, ".zarray"))
        or os.path.exists(os.path.join(path, ".zattrs"))
        or os.path.exists(os.path.join(path, "zarr.json"))
    )


def split_dataset_path(dataset_path, scale=None) -> tuple[str, str]:
    """Split the dataset path into the filename and dataset

    Args:
        dataset_path ('str'): Path to the dataset
        scale ('int'): Scale to use, if present

    Returns:
        Tuple of filename and dataset
    """

    has_zarr = ".zarr" in dataset_path
    has_n5 = ".n5" in dataset_path

    if has_zarr or has_n5:
        # split at .zarr or .n5, whichever comes last
        splitter = (
            ".zarr"
            if dataset_path.rfind(".zarr") > dataset_path.rfind(".n5")
            else ".n5"
        )

        filename, dataset = dataset_path.rsplit(splitter, 1)
        if dataset.startswith("/"):
            dataset = dataset[1:]
        # include scale if present
        if scale is not None:
            dataset += f"/s{scale}"

        return filename + splitter, dataset

    # No .zarr or .n5 extension — walk up the path to find a zarr group container.
    if _is_remote_path(dataset_path):
        raise RuntimeError(
            f"Remote URL must contain .zarr or .n5 in the path: {dataset_path}"
        )
    # Prefer group roots over leaf arrays while walking upward. Detect both
    # zarr v2 (.zgroup/.zarray) and v3 (zarr.json).
    path = os.path.normpath(dataset_path)
    parts = []
    fallback = None  # track first array-only match as fallback
    while path and path != os.path.dirname(path):
        if os.path.isdir(path):
            zarr_json = os.path.join(path, "zarr.json")
            if os.path.exists(zarr_json):
                try:
                    with open(zarr_json) as f:
                        meta = json.load(f)
                except (OSError, json.JSONDecodeError):
                    meta = {}
                if meta.get("zarr_format") == 3:
                    if meta.get("node_type") == "group":
                        dataset = "/".join(reversed(parts))
                        if scale is not None:
                            dataset = f"{dataset}/s{scale}" if dataset else f"s{scale}"
                        return path, dataset
                    if fallback is None:
                        fallback = (path, list(parts))

            if os.path.exists(os.path.join(path, ".zgroup")):
                dataset = "/".join(reversed(parts))
                if scale is not None:
                    dataset = f"{dataset}/s{scale}" if dataset else f"s{scale}"
                return path, dataset
            if fallback is None and os.path.exists(
                os.path.join(path, ".zarray")
            ):
                fallback = (path, list(parts))
        path, part = os.path.split(path)
        parts.append(part)

    if fallback is not None:
        fb_path, fb_parts = fallback
        dataset = "/".join(reversed(fb_parts))
        if scale is not None:
            dataset = f"{dataset}/s{scale}" if dataset else f"s{scale}"
        return fb_path, dataset

    raise RuntimeError(
        f"Could not find a zarr or n5 container in path: {dataset_path}"
    )


def apply_norms(data):
    if hasattr(data, "read"):
        data = data.read().result()
    # logger.error("norm time")
    for norm in g.input_norms:
        # logger.error(f"applying norm: {norm}")
        data = norm(data)
    return data


class LazyNormalization:
    def __init__(self, ts_dataset):
        self.ts_dataset = ts_dataset

    def __getitem__(self, index):
        result = self.ts_dataset[index]
        return apply_norms(result)

    def __getattr__(self, attr):
        at = getattr(self.ts_dataset, attr)
        if attr == "dtype":
            if len(g.input_norms) > 0:
                return np.dtype(g.input_norms[-1].dtype)
            return np.dtype(at.numpy_dtype)
        return at


def _detect_filetype(dataset_path: str) -> str:
    """Detect the tensorstore driver: ``zarr`` (v2), ``zarr3``, or ``n5``.

    For local paths, inspects the filesystem so v2 and v3 (which share the
    ``.zarr`` extension) can be distinguished. Falls back to extension
    parsing for remote paths and for local paths that don't yet exist.
    """
    # n5 detection via extension (existing behavior; no v3 equivalent)
    if ".n5" in dataset_path:
        if ".zarr" not in dataset_path or dataset_path.rfind(".n5") > dataset_path.rfind(".zarr"):
            return "n5"

    if _is_remote_path(dataset_path):
        return "zarr"  # remote v3 not supported yet — current callers don't have any

    # Filesystem detection at the given path
    normalized = os.path.normpath(dataset_path)
    fmt = _zarr_format(normalized)
    if fmt == 3:
        return "zarr3"
    if fmt == 2:
        return "zarr"

    # Walk up looking for any zarr marker (handles paths like .zarr/0/sub)
    path = normalized
    while path and path != os.path.dirname(path):
        if _is_zarr_container(path):
            f = _zarr_format(path)
            return "zarr3" if f == 3 else "zarr"
        path = os.path.dirname(path)

    return "zarr"


def _clean_zarr_compressor(dataset_path: str):
    """Return .zarray metadata with unsupported compressor fields removed.

    Tensorstore is strict about compressor metadata and rejects extra fields
    added by newer numcodecs versions, such as ``checksum``.
    """
    zarray_path = os.path.join(os.path.normpath(dataset_path), ".zarray")
    if not os.path.isfile(zarray_path):
        return None
    try:
        with open(zarray_path) as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    compressor = meta.get("compressor")
    if not isinstance(compressor, dict):
        return None

    known_fields = {
        "zstd": {"id", "level"},
        "zlib": {"id", "level"},
        "gzip": {"id", "level"},
        "bz2": {"id", "level"},
        "blosc": {"id", "cname", "clevel", "shuffle", "blocksize"},
    }
    allowed = known_fields.get(compressor.get("id", ""))
    if allowed is None:
        return None

    extra_keys = set(compressor.keys()) - allowed
    if not extra_keys:
        return None

    logger.info(
        "Stripping unsupported compressor fields %s for tensorstore compatibility",
        extra_keys,
    )
    meta["compressor"] = {k: v for k, v in compressor.items() if k in allowed}
    return meta


def open_ds_tensorstore(
    dataset_path: str, mode="r", concurrency_limit=None, normalize=True
):
    # open with zarr or n5 depending on extension
    filetype = _detect_filetype(dataset_path)
    extra_args = {}

    if dataset_path.startswith("precomputed://"):
        # precomputed:// URLs point to neuroglancer precomputed format
        raw_path = "/" + dataset_path[len("precomputed://"):].lstrip("/")
        if ends_with_scale(raw_path):
            scale_index = int(raw_path.rsplit("/s")[1])
            raw_path = raw_path.rsplit("/s")[0]
        else:
            scale_index = 0
        filetype = "neuroglancer_precomputed"
        kvstore = {
            "driver": "file",
            "path": os.path.normpath(raw_path),
        }
        extra_args = {"scale_index": scale_index}
    elif dataset_path.startswith("http://") or dataset_path.startswith("https://"):
        kvstore = {
            "driver": "http",
            "base_url": dataset_path.rstrip("/"),
            "path": "",
        }
    elif dataset_path.startswith("s3://"):
        kvstore = {
            "driver": "s3",
            "bucket": dataset_path.split("/")[2],
            "path": "/".join(dataset_path.split("/")[3:]),
            "aws_credentials": {
                "anonymous": True,
            },
        }
    elif dataset_path.startswith("gs://"):
        # check if path ends with s#int
        if ends_with_scale(dataset_path):
            scale_index = int(dataset_path.rsplit("/s")[1])
            dataset_path = dataset_path.rsplit("/s")[0]
        else:
            scale_index = 0
        filetype = "neuroglancer_precomputed"
        kvstore = dataset_path
        extra_args = {"scale_index": scale_index}
    else:
        kvstore = {
            "driver": "file",
            "path": os.path.normpath(dataset_path),
        }

    assume_metadata = False
    if (
        filetype == "zarr"
        and isinstance(kvstore, dict)
        and kvstore.get("driver") == "file"
    ):
        cleaned_metadata = _clean_zarr_compressor(kvstore["path"])
        if cleaned_metadata is not None:
            extra_args["metadata"] = cleaned_metadata
            assume_metadata = True

    if concurrency_limit:
        spec = {
            "driver": filetype,
            "context": {
                "data_copy_concurrency": {"limit": concurrency_limit},
                "file_io_concurrency": {"limit": concurrency_limit},
            },
            "kvstore": kvstore,
            **extra_args,
        }
    else:
        spec = {"driver": filetype, "kvstore": kvstore, **extra_args}

    open_kwargs = {"open": True, "assume_metadata": True} if assume_metadata else {}
    if mode == "r":
        dataset_future = ts.open(spec, read=True, write=False, **open_kwargs)
    else:
        dataset_future = ts.open(spec, read=False, write=True, **open_kwargs)

    try:
        ts_dataset = dataset_future.result()
        if ts_dataset.ndim > 3:
            from cellmap_flow.norm.input_normalize import ChannelSelector

            channel = 0
            for norm in g.input_norms:
                if isinstance(norm, ChannelSelector):
                    channel = norm.channel
                    break
            ts_dataset = ts_dataset[channel]
    except ValueError as e:
        if "extra members" in str(e) and filetype == "zarr":
            # Some zarr files have extra fields (e.g. "checksum") in the
            # compressor metadata that tensorstore doesn't recognize.
            # Fix by providing the metadata explicitly without the extra fields.
            cleaned_metadata = None
            if isinstance(kvstore, dict) and kvstore.get("driver") == "file":
                cleaned_metadata = _clean_zarr_compressor(kvstore["path"])
            if cleaned_metadata is None:
                raise
            spec["metadata"] = cleaned_metadata
            if mode == "r":
                dataset_future = ts.open(
                    spec, read=True, write=False, assume_metadata=True
                )
            else:
                dataset_future = ts.open(
                    spec, read=False, write=True, assume_metadata=True
                )
            ts_dataset = dataset_future.result()
        else:
            raise

    # return ts_dataset
    if normalize:
        return LazyNormalization(ts_dataset)
    return ts_dataset


def to_ndarray_tensorstore(
    dataset,
    roi=None,
    voxel_size=None,
    offset=None,
    output_voxel_size=None,
    axes_names=["z", "y", "x"],
    custom_fill_value=None,
):
    """Read a region of a tensorstore dataset and return it as a numpy array

    Args:
        dataset ('tensorstore.dataset'): Tensorstore dataset
        roi ('funlib.geometry.Roi'): Region of interest to read

    Returns:
        Numpy array of the region
    """

    if roi is None:
        with ts.Transaction() as txn:
            return dataset.with_transaction(txn).read().result()

    if offset is None:
        offset = Coordinate(np.zeros(roi.dims, dtype=int))

    if output_voxel_size is None:
        output_voxel_size = voxel_size

    rescale_factor = 1
    if voxel_size != output_voxel_size:
        # in the case where there is a mismatch in voxel sizes, we may need to extra pad to ensure that the output is a multiple of the output voxel size
        original_roi = roi
        roi = original_roi.snap_to_grid(voxel_size)
        rescale_factor = voxel_size[0] / output_voxel_size[0]
        snapped_offset = (original_roi.begin - roi.begin) / output_voxel_size
        snapped_end = (original_roi.end - roi.begin) / output_voxel_size
        snapped_slices = tuple(
            slice(snapped_offset[i], snapped_end[i]) for i in range(3)
        )

    roi -= offset
    roi /= voxel_size

    # Specify the range
    roi_slices = roi.to_slices()

    domain = dataset.domain
    # Compute the valid range
    valid_slices = tuple(
        slice(max(s.start, inclusive_min), min(s.stop, exclusive_max))
        for s, inclusive_min, exclusive_max in zip(
            roi_slices, domain.inclusive_min, domain.exclusive_max
        )
    )

    # Create an array to hold the requested data, filled with a default value (e.g., zeros)
    # output_shape = [s.stop - s.start for s in roi_slices]

    if not dataset.fill_value:
        fill_value = 0
    if custom_fill_value:
        fill_value = custom_fill_value
    with ts.Transaction() as txn:
        data = dataset.with_transaction(txn)[valid_slices].read().result()
        # logger.error("norm time")
        for norm in g.input_norms:
            # logger.error(f"Applying norm: {norm}")
            data = norm(data)
    pad_width = [
        [valid_slice.start - s.start, s.stop - valid_slice.stop]
        for s, valid_slice in zip(roi_slices, valid_slices)
    ]
    if np.any(np.array(pad_width)):
        if fill_value == "edge":
            data = np.pad(
                data,
                pad_width=pad_width,
                mode="edge",
            )
        else:
            data = np.pad(
                data,
                pad_width=pad_width,
                mode="constant",
                constant_values=fill_value,
            )

    if rescale_factor > 1:
        rescale_factor = int(voxel_size[0] / output_voxel_size[0])
        data = np.kron(data, np.ones((rescale_factor, rescale_factor, rescale_factor), dtype=data.dtype))
        data = data[snapped_slices]

    elif rescale_factor < 1:
        data = block_reduce(data, block_size=int(1 / rescale_factor), func=np.median)
        data = data[snapped_slices]

    return data


def get_url(node: Union[zarr.Group, zarr.Array]) -> str:
    store = node.store
    if hasattr(store, "path"):
        if hasattr(store, "fs"):
            if isinstance(store.fs.protocol, Sequence):
                protocol = store.fs.protocol[0]
            else:
                protocol = store.fs.protocol
        else:
            protocol = "file"

        # fsstore keeps the protocol in the path, but not s3store
        if "://" in store.path:
            store_path = store.path.split("://")[-1]
        else:
            store_path = store.path
        return f"{protocol}://{store_path}"
    else:
        raise ValueError(
            f"The store associated with this object has type {type(store)}, which "
            "cannot be resolved to a url"
        )


def separate_store_path(store, path):
    """
    sometimes you can pass a total os path to node, leading to
    an empty('') node.path attribute.
    the correct way is to separate path to container(.n5, .zarr)
    from path to array within a container.

    Args:
        store (string): path to store
        path (string): path array/group (.n5 or .zarr)

    Returns:
        (string, string): returns regularized store and group/array path
    """
    new_store, path_prefix = os.path.split(store)
    if ".zarr" in path_prefix or ".n5" in path_prefix:
        return store, path
    # For extensionless zarr containers, check for zarr metadata on disk.
    # Strip file:// protocol prefix for filesystem check.
    local_path = store
    if local_path.startswith("file://"):
        local_path = local_path[len("file://"):]
    if os.path.exists(os.path.join(local_path, ".zgroup")) or os.path.exists(
        os.path.join(local_path, ".zarray")
    ):
        return store, path
    if new_store == store:
        # Reached the root without finding a container
        raise RuntimeError(f"Could not find zarr/n5 container in path: {store}")
    return separate_store_path(new_store, os.path.join(path_prefix, path))


def access_parent(node):
    """
    Get the parent (zarr.Group) of an input zarr array(ds).


    Args:
        node: zarr-python 2 ``zarr.core.Array`` / ``zarr.hierarchy.Group``
              OR v3 :class:`ZarrV3Node`

    Raises:
        RuntimeError: returned if the node array is in the parent group,
        or the group itself is the root group

    Returns:
        Parent group containing input group/array (same type as input)
    """
    if isinstance(node, ZarrV3Node):
        parent_path = os.path.dirname(os.path.normpath(node.path))
        if _zarr_format(parent_path) != 3:
            raise RuntimeError(
                f"{node.name} is at the root of the {node.path} v3 store."
            )
        return ZarrV3Node(parent_path)

    path = get_url(node)

    store_path, node_path = separate_store_path(path, node.path)
    if node_path == "":
        raise RuntimeError(f"{node.name} is in the root group of the {path} store.")
    else:
        if store_path.endswith(".n5"):
            store_path = N5FSStore(store_path)
        return zarr.open(store=store_path, path=os.path.split(node_path)[0], mode="r")


def check_for_multiscale(group):
    """check if multiscale attribute exists in the input group and for any parent level group

    Args:
        group: zarr-python 2 ``zarr.hierarchy.Group`` or v3 :class:`ZarrV3Node`

    Returns:
        tuple({}, group): (multiscales attribute body, zarr group where multiscales was found)
    """
    multiscales = group.attrs.get("multiscales", None)

    if multiscales:
        return (multiscales, group)

    if _is_zarr_root(group):
        return (multiscales, group)

    return check_for_multiscale(access_parent(group))


# check if voxel_size value is present in .zatts other than in multiscale attribute
def check_for_voxel_size(array, order):
    """checks specific attributes(resolution, scale,
        pixelResolution["dimensions"], transform["scale"]) for voxel size
        value in the parent directory of the input array

    Args:
        array (zarr.core.Array): array to check
        order (string): colexicographical/lexicographical order
    Raises:
        ValueError: raises value error if no voxel_size value is found

    Returns:
       [float] : returns physical size of the voxel (unitless)
    """

    voxel_size = None
    parent_group = access_parent(array)
    for item in [array, parent_group]:

        if "resolution" in item.attrs:
            return item.attrs["resolution"]
        elif "scale" in item.attrs:
            return item.attrs["scale"]
        elif "pixelResolution" in item.attrs:
            downsampling_factors = [1, 1, 1]
            if "downsamplingFactors" in item.attrs:
                downsampling_factors = item.attrs["downsamplingFactors"]
            if "dimensions" not in item.attrs["pixelResolution"]:
                base_resolution = item.attrs["pixelResolution"]
            else:
                base_resolution = item.attrs["pixelResolution"]["dimensions"]
            final_resolution = list(
                np.array(base_resolution) * np.array(downsampling_factors)
            )
            return final_resolution
        elif "transform" in item.attrs:
            # Davis saves transforms in C order regardless of underlying
            # memory format (i.e. n5 or zarr). May be explicitly provided
            # as transform.ordering
            transform_order = item.attrs["transform"].get("ordering", "C")
            voxel_size = item.attrs["transform"]["scale"]
            if transform_order != order:
                voxel_size = voxel_size[::-1]
            return voxel_size

    return voxel_size


# check if offset value is present in .zatts other than in multiscales
def check_for_offset(array, order):
    """checks specific attributes(offset, transform["translate"]) for offset
        value in the parent directory of the input array

    Args:
        array (zarr.core.Array): array to check
        order (string): colexicographical/lexicographical order
    Raises:
        ValueError: raises value error if no offset value is found

    Returns:
       [float] : returns offset of the voxel (unitless) in respect to
                the center of the coordinate system
    """
    offset = None
    parent_group = access_parent(array)
    for item in [array, parent_group]:

        if "offset" in item.attrs:
            offset = item.attrs["offset"]
            return offset

        elif "transform" in item.attrs:
            transform_order = item.attrs["transform"].get("ordering", "C")
            offset = item.attrs["transform"]["translate"]
            if transform_order != order:
                offset = offset[::-1]
            return offset

    return offset


def check_for_units(array, order):
    """checks specific attributes(units, pixelResolution["unit"] transform["units"])
        for units(nm, cm, etc.) value in the parent directory of the input array

    Args:
        array (zarr.core.Array): array to check
        order (string): colexicographical/lexicographical order
    Raises:
        ValueError: raises value error if no units value is found

    Returns:
       [string] : returns units for the voxel_size
    """

    units = None
    parent_group = access_parent(array)
    for item in [array, parent_group]:

        if "units" in item.attrs:
            return item.attrs["units"]
        elif (
            "pixelResolution" in item.attrs and "unit" in item.attrs["pixelResolution"]
        ):
            unit = item.attrs["pixelResolution"]["unit"]
            return [unit for _ in range(len(array.shape))]
        elif "transform" in item.attrs:
            # Davis saves transforms in C order regardless of underlying
            # memory format (i.e. n5 or zarr). May be explicitly provided
            # as transform.ordering
            transform_order = item.attrs["transform"].get("ordering", "C")
            units = item.attrs["transform"]["units"]
            if transform_order != order:
                units = units[::-1]
            return units

    if units is None:
        Warning(
            f"No units attribute was found for {type(array.store)} store. Using pixels."
        )
        return "pixels"


def check_for_attrs_multiscale(ds, multiscale_group, multiscales):
    """checks multiscale attribute of the .zarr or .n5 group
        for voxel_size(scale), offset(translation) and units values

    Args:
        ds (zarr.core.Array): input zarr Array
        multiscale_group (zarr.hierarchy.Group): the group attrs
                                                that contains multiscale
        multiscales ({}): dictionary that contains all the info necessary
                            to create multiscale resolution pyramid

    Returns:
        ([float],[float],[string]): returns (voxel_size, offset, physical units)
    """

    voxel_size = None
    offset = None
    units = None

    if multiscales is not None:
        logger.info("Found multiscales attributes")
        scale = os.path.relpath(
            separate_store_path(get_url(ds), ds.path)[1], multiscale_group.path
        )
        if isinstance(ds.store, (zarr.n5.N5Store, zarr.n5.N5FSStore)):
            for level in multiscales[0]["datasets"]:
                if level["path"] == scale:

                    voxel_size = level["transform"]["scale"]
                    offset = level["transform"]["translate"]
                    units = level["transform"]["units"]
                    return voxel_size, offset, units
        # for zarr store
        else:
            units = [item["unit"] for item in multiscales[0]["axes"]]
            for level in multiscales[0]["datasets"]:
                if level["path"].lstrip("/") == scale:
                    for attr in level["coordinateTransformations"]:
                        if attr["type"] == "scale":
                            voxel_size = attr["scale"]
                        elif attr["type"] == "translation":
                            offset = attr["translation"]
                    return voxel_size, offset, units

    return voxel_size, offset, units


def _read_attrs(ds, order="C"):
    """check n5/zarr metadata and returns voxel_size, offset, physical units,
        for the input zarr array(ds)

    Args:
        ds (zarr.core.Array): input zarr array
        order (str, optional): _description_. Defaults to "C".

    Raises:
        TypeError: incorrect data type of the input(ds) array.
        ValueError: returns value error if no multiscale attribute was found
    Returns:
        _type_: _description_
    """
    voxel_size = None
    offset = None
    units = None
    multiscales = None

    if not isinstance(ds, zarr.core.Array):
        raise TypeError(
            f"{os.path.join(ds.store.path, ds.path)} is not zarr.core.Array"
        )

    # check recursively for multiscales attribute in the zarr store tree
    multiscales, multiscale_group = check_for_multiscale(group=access_parent(ds))

    # check for attributes in .zarr group multiscale
    if not isinstance(ds.store, (zarr.n5.N5Store, zarr.n5.N5FSStore)):
        if multiscales:
            voxel_size, offset, units = check_for_attrs_multiscale(
                ds, multiscale_group, multiscales
            )

    # if multiscale attribute is missing
    if voxel_size is None:
        voxel_size = check_for_voxel_size(ds, order)
    if offset is None:
        offset = check_for_offset(ds, order)
    if units is None:
        units = check_for_units(ds, order)

    dims = len(ds.shape)
    dims = dims if dims <= 3 else 3

    if voxel_size is not None and offset is not None and units is not None:
        if order == "F" or isinstance(ds.store, (zarr.n5.N5Store, zarr.n5.N5FSStore)):
            return voxel_size[::-1], offset[::-1], units[::-1]
        else:
            return voxel_size, offset, units

    # if no voxel offset are found in transform, offset or scale, check in n5 multiscale attribute:
    if (
        isinstance(ds.store, (zarr.n5.N5Store, zarr.n5.N5FSStore))
        and multiscales != False
    ):

        voxel_size, offset, units = check_for_attrs_multiscale(
            ds, multiscale_group, multiscales
        )

    # return default value if an attribute was not found
    if voxel_size is None:
        voxel_size = (1,) * dims
        Warning(f"No voxel_size attribute was found. Using {voxel_size} as default.")
    if offset is None:
        offset = (0,) * dims
        Warning(f"No offset attribute was found. Using {offset} as default.")
    if units is None:
        units = "pixels"
        Warning(f"No units attribute was found. Using {units} as default.")

    if order == "F":
        return voxel_size[::-1], offset[::-1], units[::-1]
    else:
        return voxel_size, offset, units


def regularize_offset(voxel_size_float, offset_float):
    """
        offset is not a multiple of voxel_size. This is often due to someone defining
        offset to the point source of each array element i.e. the center of the rendered
        voxel, vs the offset to the corner of the voxel.
        apparently this can be a heated discussion. See here for arguments against
        the convention we are using: http://alvyray.com/Memos/CG/Microsoft/6_pixel.pdf

    Args:
        voxel_size_float ([float]): float voxel size list
        offset_float ([float]): float offset list
    Returns:
        (Coordinate, Coordinate)): returned offset size that is multiple of voxel size
    """
    voxel_size, offset = Coordinate(voxel_size_float), Coordinate(offset_float)

    if voxel_size is not None and (offset / voxel_size) * voxel_size != offset:

        logger.debug(
            f"Offset: {offset} being rounded to nearest voxel size: {voxel_size}"
        )
        offset = (
            (Coordinate(offset) + (Coordinate(voxel_size) / 2)) / Coordinate(voxel_size)
        ) * Coordinate(voxel_size)
        logger.debug(f"Rounded offset: {offset}")

    return Coordinate(voxel_size), Coordinate(offset)


def _read_voxel_size_offset(ds, order="C"):

    voxel_size, offset, units = _read_attrs(ds, order)
    for idx, unit in enumerate(units):
        if unit == "um":
            voxel_size[idx] = voxel_size[idx] * 1000
            offset[idx] = offset[idx] * 1000

    return regularize_offset(voxel_size, offset)


def get_ds_info(path: str, mode: str = "r"):
    """Open a Zarr, N5, or HDF5 dataset as an :class:`Array`. If the
    dataset has attributes ``resolution`` and ``offset``, those will be
    used to determine the meta-information of the returned array.

    Args:

        filename:

            The name of the container "file" (which is a directory for zarr and
            N5).

        ds_name:

            The name of the dataset to open.

    Returns:

        A :class:`Array` pointing to the dataset.
    """

    path = _normalize_path(path)
    axes_names = ["x", "y", "z"]
    if path.startswith("s3://"):
        ts_info = open_ds_tensorstore(path)
        shape = ts_info.shape
        path, filename = split_dataset_path(path)
        filename, scale = filename.rsplit("/s")
        scale = int(scale)
        fs = s3fs.S3FileSystem(
            anon=True
        )  # Set anon=True if you don't need authentication
        store = s3fs.S3Map(root=path, s3=fs)
        zarr_dataset = zarr.open(
            store,
            mode="r",
        )
        multiscale_attrs = zarr_dataset[filename].attrs.asdict()
        if "multiscales" in multiscale_attrs:
            multiscales = multiscale_attrs["multiscales"][0]
            axes = [axis["name"] for axis in multiscales["axes"]]
            for scale_info in multiscale_attrs["multiscales"][0]["datasets"]:
                if scale_info["path"] == f"s{scale}":
                    voxel_size = Coordinate(
                        scale_info["coordinateTransformations"][0]["scale"]
                    )
        axes_names = axes[:3]

        chunk_shape = Coordinate(ts_info.chunk_layout.read_chunk.shape)
        roi = Roi((0, 0, 0), Coordinate(shape) * voxel_size)
        return voxel_size, chunk_shape, shape, roi, axes_names

    elif path.startswith("gs://"):
        ts_info = open_ds_tensorstore(path)
        shape = ts_info.shape
        voxel_size = Coordinate(
            (d.to_json()[0] if d is not None else 1 for d in ts_info.dimension_units)
        )
        axes_names = list(ts_info.spec().transform.input_labels[:3])
        chunk_shape = Coordinate(ts_info.chunk_layout.read_chunk.shape)
        roi = Roi([0] * len(shape), Coordinate(shape) * voxel_size)
        file_type = "gs"
        return voxel_size, chunk_shape, shape, roi, axes_names, file_type

    elif path.startswith("precomputed://"):
        ts_info = open_ds_tensorstore(path)
        shape = ts_info.shape
        voxel_size = Coordinate(
            (d.to_json()[0] if d is not None else 1 for d in ts_info.dimension_units)
        )
        axes_names = list(ts_info.spec().transform.input_labels[:3])
        chunk_shape = Coordinate(ts_info.chunk_layout.read_chunk.shape)
        roi = Roi([0] * len(shape), Coordinate(shape) * voxel_size)
        return voxel_size, chunk_shape, shape, roi, axes_names, "precomputed"

    # v3 branch — local zarr_format=3 (sharded or not).
    # Detected by presence of zarr.json at the given path. v3's OME-NGFF
    # multiscales metadata lives in the group's zarr.json "attributes" field,
    # using the same coordinateTransformations schema as v2 .zattrs.
    if (
        not _is_remote_path(path)
        and _zarr_format(_normalize_path(path)) == 3
    ):
        ds = _open_zarr(path, mode="r")  # ZarrV3Node

        # Group root: navigate into the first scale via multiscales metadata,
        # or fall back to the first child array.
        if _is_zarr_group(ds):
            multiscales = ds.attrs.get("multiscales", None)
            if multiscales:
                ms = multiscales[0]
                first_dataset = ms["datasets"][0]
                ds = ds[first_dataset["path"]]

                axes = ms.get("axes", [])
                spatial_indices = [
                    i for i, a in enumerate(axes) if a.get("type") == "space"
                ]
                if not spatial_indices:
                    spatial_indices = list(range(len(ds.shape)))
                axes_names = (
                    [axes[i]["name"] for i in spatial_indices]
                    if axes
                    else ["z", "y", "x"]
                )

                scale_transform = first_dataset["coordinateTransformations"][0][
                    "scale"
                ]
                voxel_size = Coordinate(
                    scale_transform[i] for i in spatial_indices
                )
                translation = next(
                    (
                        t["translation"]
                        for t in first_dataset["coordinateTransformations"]
                        if t["type"] == "translation"
                    ),
                    [0.0] * len(scale_transform),
                )
                offset = Coordinate(translation[i] for i in spatial_indices)
                shape = Coordinate(ds.shape[i] for i in spatial_indices)
                chunk_shape = tuple(ds.chunks[i] for i in spatial_indices)
                roi = Roi(offset, voxel_size * shape)
                return voxel_size, chunk_shape, shape, roi, axes_names, "zarr3"

            # Group without multiscales — pick first child array
            for key in ds.keys():
                child = ds[key]
                if not _is_zarr_group(child):
                    ds = child
                    break

        # Array path: try the parent group's multiscales (OME-NGFF pattern)
        parent_path = os.path.dirname(os.path.normpath(_normalize_path(path)))
        if _zarr_format(parent_path) == 3:
            try:
                parent = ZarrV3Node(parent_path)
                if _is_zarr_group(parent):
                    multiscales = parent.attrs.get("multiscales", None)
                    if multiscales:
                        ms = multiscales[0]
                        axes = ms.get("axes", [])
                        spatial_indices = [
                            i for i, a in enumerate(axes) if a.get("type") == "space"
                        ]
                        if not spatial_indices:
                            spatial_indices = list(range(len(ds.shape)))

                        sub_path = os.path.basename(
                            os.path.normpath(_normalize_path(path))
                        )
                        dataset_entry = next(
                            (d for d in ms["datasets"] if d["path"] == sub_path),
                            ms["datasets"][0],
                        )
                        scale_transform = dataset_entry[
                            "coordinateTransformations"
                        ][0]["scale"]
                        voxel_size = Coordinate(
                            scale_transform[i] for i in spatial_indices
                        )
                        translation = next(
                            (
                                t["translation"]
                                for t in dataset_entry["coordinateTransformations"]
                                if t["type"] == "translation"
                            ),
                            [0.0] * len(scale_transform),
                        )
                        offset = Coordinate(
                            translation[i] for i in spatial_indices
                        )
                        axes_names = (
                            [axes[i]["name"] for i in spatial_indices]
                            if axes
                            else ["z", "y", "x"]
                        )
                        shape = Coordinate(ds.shape[i] for i in spatial_indices)
                        chunk_shape = tuple(ds.chunks[i] for i in spatial_indices)
                        roi = Roi(offset, voxel_size * shape)
                        return (
                            voxel_size,
                            chunk_shape,
                            shape,
                            roi,
                            axes_names,
                            "zarr3",
                        )
            except Exception as e:
                logger.warning(
                    "failed to read v3 parent multiscale metadata for %s: %s"
                    % (path, e)
                )

        # Fallback: no multiscales context — defaults
        dims = min(3, len(ds.shape))
        voxel_size = Coordinate((1,) * dims)
        offset = Coordinate((0,) * dims)
        shape = Coordinate(ds.shape[-dims:])
        chunk_shape = tuple(ds.chunks[-dims:])
        axes_names = ["z", "y", "x"][:dims]
        roi = Roi(offset, voxel_size * shape)
        return voxel_size, chunk_shape, shape, roi, axes_names, "zarr3"

    if _is_remote_path(path):
        ds = _open_zarr(path, mode="r")

        # If the URL points to a zarr Group (e.g. multiscale container),
        # read OME-Zarr multiscales metadata and navigate into the first array.
        if isinstance(ds, zarr.hierarchy.Group):
            multiscales = ds.attrs.get("multiscales", None)
            if multiscales:
                ms = multiscales[0]
                first_dataset = ms["datasets"][0]
                ds = ds[first_dataset["path"]]

                # Extract spatial axes info (skip channel axes)
                axes = ms.get("axes", [])
                spatial_indices = [
                    i for i, a in enumerate(axes) if a.get("type") == "space"
                ]
                axes_names = [axes[i]["name"] for i in spatial_indices]

                scale_transform = first_dataset["coordinateTransformations"][0]["scale"]
                voxel_size = Coordinate(scale_transform[i] for i in spatial_indices)

                translation = next(
                    (
                        t["translation"]
                        for t in first_dataset["coordinateTransformations"]
                        if t["type"] == "translation"
                    ),
                    [0.0] * len(scale_transform),
                )
                offset = Coordinate(translation[i] for i in spatial_indices)

                shape = Coordinate(ds.shape[i] for i in spatial_indices)
                chunk_shape = tuple(ds.chunks[i] for i in spatial_indices)
                roi = Roi(offset, voxel_size * shape)
                return voxel_size, chunk_shape, shape, roi, axes_names, "zarr"
            else:
                for key in sorted(ds.keys()):
                    if isinstance(ds[key], zarr.core.Array):
                        ds = ds[key]
                        break

        # The path points to a sub-array (e.g. .zarr/raw/s0). Try to read
        # multiscale metadata from the parent zarr Group.
        if ".zarr" in path or ".n5" in path:
            container, sub_path = split_dataset_path(path)
            if sub_path:
                try:
                    parent = _open_zarr(container, mode="r")
                    multiscales = parent.attrs.get("multiscales", None)
                    if multiscales:
                        ms = multiscales[0]
                        axes = ms.get("axes", [])
                        spatial_indices = [
                            i
                            for i, a in enumerate(axes)
                            if a.get("type") == "space"
                        ]
                        if not spatial_indices:
                            spatial_indices = list(range(len(ds.shape)))

                        # Find the matching dataset entry
                        dataset_entry = next(
                            (
                                d
                                for d in ms["datasets"]
                                if d["path"] == sub_path
                            ),
                            ms["datasets"][0],
                        )
                        scale_transform = dataset_entry[
                            "coordinateTransformations"
                        ][0]["scale"]
                        voxel_size = Coordinate(
                            scale_transform[i] for i in spatial_indices
                        )
                        translation = next(
                            (
                                t["translation"]
                                for t in dataset_entry[
                                    "coordinateTransformations"
                                ]
                                if t["type"] == "translation"
                            ),
                            [0.0] * len(scale_transform),
                        )
                        offset = Coordinate(
                            translation[i] for i in spatial_indices
                        )
                        axes_names = [axes[i]["name"] for i in spatial_indices] if axes else ["z", "y", "x"]
                        shape = Coordinate(
                            ds.shape[i] for i in spatial_indices
                        )
                        chunk_shape = tuple(
                            ds.chunks[i] for i in spatial_indices
                        )
                        roi = Roi(offset, voxel_size * shape)
                        return (
                            voxel_size,
                            chunk_shape,
                            shape,
                            roi,
                            axes_names,
                            "zarr",
                        )
                except Exception as e:
                    logger.warning(
                        "failed to read parent multiscale metadata for %s: %s"
                        % (path, e)
                    )

        # Fallback for remote arrays without multiscales metadata
        try:
            order = ds.attrs["order"]
        except KeyError:
            try:
                order = ds.order
            except Exception:
                logger.error("no order attribute found, set default C")
                order = "C"
        try:
            voxel_size, offset = _read_voxel_size_offset(ds, order)
        except Exception:
            logger.error(
                "failed to read voxel size and offset for %s, Will use default values"
                % path
            )
            voxel_size = Coordinate((1,) * 3)
            offset = Coordinate((0,) * 3)
        shape = Coordinate(ds.shape[-len(voxel_size) :])
        roi = Roi(offset, voxel_size * shape)
        chunk_shape = ds.chunks
        return voxel_size, chunk_shape, shape, roi, ["z", "y", "x"], "zarr"

    filename, ds_name = split_dataset_path(path)
    if filename.endswith(".zarr") or filename.endswith(".zip") or _is_zarr_container(filename):
        assert (
            not filename.endswith(".zip") or mode == "r"
        ), "Only reading supported for zarr ZipStore"

        logger.debug("opening zarr dataset %s in %s", ds_name, filename)
        try:
            ds = zarr.open(filename, mode=mode)
            if ds_name:
                ds = ds[ds_name]
        except Exception as e:
            logger.error("failed to open %s/%s" % (filename, ds_name))
            raise e

        try:
            order = ds.attrs["order"]
        except KeyError:
            try:
                order = ds.order
            except Exception:
                logger.error("no order attribute found in %s set default C" % ds_name)
                order = "C"
        try:
            voxel_size, offset = _read_voxel_size_offset(ds, order)
        except Exception as e:
            logger.error(
                "failed to read voxel size and offset for %s/%s, Will use default values"
                % (filename, ds_name)
            )
            voxel_size = Coordinate((1,) * 3)
            offset = Coordinate((0,) * 3)
        shape = Coordinate(ds.shape[-len(voxel_size) :])
        roi = Roi(offset, voxel_size * shape)

        chunk_shape = ds.chunks

        logger.debug("opened zarr dataset %s in %s", ds_name, filename)
        return voxel_size, chunk_shape, shape, roi, ["z", "y", "x"], "zarr"

    elif filename.endswith(".n5"):
        logger.debug("opening N5 dataset %s in %s", ds_name, filename)
        ds = zarr.open(N5FSStore(filename), mode=mode)[ds_name]

        voxel_size, offset = _read_voxel_size_offset(ds, "F")
        shape = Coordinate(ds.shape[-len(voxel_size) :])
        roi = Roi(offset, voxel_size * shape)

        chunk_shape = ds.chunks

        logger.debug("opened N5 dataset %s in %s", ds_name, filename)
        return voxel_size, chunk_shape, shape, roi, axes_names, "n5"

    elif filename.endswith(".h5") or filename.endswith(".hdf"):
        logger.debug("opening H5 dataset %s in %s", ds_name, filename)
        ds = h5py.File(filename, mode=mode)[ds_name]

        voxel_size, offset = _read_voxel_size_offset(ds, "C")
        shape = Coordinate(ds.shape[-len(voxel_size) :])
        roi = Roi(offset, voxel_size * shape)

        chunk_shape = ds.chunks

        logger.debug("opened H5 dataset %s in %s", ds_name, filename)
        return voxel_size, chunk_shape, shape, roi, axes_names, "h5"

    elif filename.endswith(".json"):
        logger.debug("found JSON container spec")
        with open(filename, "r") as f:
            spec = json.load(f)
        assert "container" in spec, "JSON spec must contain 'container' key"
        return get_ds_info(spec["container"], ds_name, mode=mode), "json"

    else:
        logger.error("don't know data format of %s in %s", ds_name, filename)
        raise RuntimeError("Unknown file format for %s" % filename)
