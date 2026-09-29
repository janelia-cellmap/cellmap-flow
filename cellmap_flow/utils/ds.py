# %%
import json
import logging
import os
from typing import Sequence, Union

import numpy as np
import tensorstore as ts
import zarr
from funlib.geometry import Coordinate
from skimage.measure import block_reduce
from zarr.n5 import N5FSStore

from cellmap_flow.globals import g
from cellmap_flow.io import metadata, multiscale, paths
from cellmap_flow.io.metadata import (  # noqa: F401  (kept names; see io.metadata)
    open_zarr as _open_zarr,
    regularize_offset,
)
from cellmap_flow.io.paths import (  # noqa: F401  (kept names; see io.paths)
    ends_with_scale,
    is_remote as _is_remote_path,
    is_zarr_container as _is_zarr_container,
    join as _join_path,
    normalize_path as _normalize_path,
)
from cellmap_flow.utils import zarr_v3


def generate_singlescale_metadata(
    arr_name: str,
    voxel_size: list,
    offset: list,
    units: str,
    axes: list,
):
    """OME-NGFF 0.4 multiscales attrs for one array whose voxel 0 has its
    lower corner at ``offset``.

    OME translation is the *centre* of voxel 0, so spatial axes get
    ``offset + voxel_size / 2``. Writing the corner there put every output
    half a voxel off in Neuroglancer and in any OME reader.
    """
    translation = [
        float(o) if axis in ("c", "c^") else float(o) + float(v) / 2
        for axis, o, v in zip(axes, offset, voxel_size)
    ]
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
    """``(offsets, resolutions, shapes)`` of an opened OME-Zarr v2 group,
    keyed by level path: spatial axes only, nanometers, offsets the corner
    of voxel 0 (see io.metadata)."""
    return zarr_v3.scale_info(metadata.levels_from_zarr_group(zarr_grp))


def find_closest_scale(zarr_grp_path, target_resolution):
    """``(level path, offset, shape)`` of the level of an OME-Zarr v2 group
    to read at ``target_resolution`` (see io.multiscale.select_level,
    "floor"); the finest level when it is None."""
    group = _open_zarr(zarr_grp_path, mode="r")
    levels = metadata.levels_from_zarr_group(group, zarr_grp_path)
    return zarr_v3.level_info(multiscale.select_level(levels, target_resolution))


# Ensure tensorstore does not attempt to use GCE credentials
os.environ["GCE_METADATA_ROOT"] = "metadata.google.internal.invalid"

# Much below taken from flyemflows: https://github.com/janelia-flyem/flyemflows/blob/master/flyemflows/util/util.py
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def split_dataset_path(dataset_path, scale=None) -> tuple[str, str]:
    """``(container, dataset inside it)``, as io.paths.split_container.

    ``scale``, if given, is appended to the dataset as ``s<scale>``.
    """
    filename, dataset, by_suffix = paths._split_container(dataset_path)
    if scale is not None:
        if by_suffix:
            dataset += f"/s{scale}"
        else:
            dataset = f"{dataset}/s{scale}" if dataset else f"s{scale}"
    return filename, dataset


def apply_norms(data, input_norms=None):
    """Read ``data`` if it is a tensorstore view and run it through the chain.

    ``input_norms=None`` means the process-wide ``g.input_norms``.
    """
    if hasattr(data, "read"):
        data = data.read().result()
    for norm in g.input_norms if input_norms is None else input_norms:
        data = norm(data)
    return data


_CHANNEL_LABELS = ("c", "c^", "channel")


def selected_channel(input_norms) -> int:
    """The input channel a chain asks for: its first ChannelSelector, else 0."""
    from cellmap_flow.norm.input_normalize import ChannelSelector

    for norm in input_norms or []:
        if isinstance(norm, ChannelSelector):
            return norm.channel
    return 0


def select_channel(ts_dataset, channel=0, spatial_ndim=3):
    """Index one channel out of a multichannel tensorstore.

    The channel axis is the one labelled c/c^/channel when the store labels its
    dimensions, and otherwise the first one (the OME-Zarr convention).
    Arrays with no more than ``spatial_ndim`` dimensions are returned as is.
    """
    if ts_dataset.ndim <= spatial_ndim:
        return ts_dataset
    labels = list(getattr(ts_dataset.domain, "labels", None) or [])
    axis = next((i for i, lab in enumerate(labels) if lab in _CHANNEL_LABELS), 0)
    if axis == 0:
        return ts_dataset[channel]
    return ts_dataset[ts.d[axis][channel]]


class LazyNormalization:
    """A tensorstore seen through the input chain, for neuroglancer to index.

    The channel and the normalizers are looked up on every access rather than
    fixed when the store was opened: a server that has already read one chunk
    must still follow a ChannelSelector that changes afterwards.

    ``input_norms=None`` follows the process-wide ``g.input_norms``.
    ``normalize=False`` selects the channel but applies no normalizers.
    """

    def __init__(self, ts_dataset, input_norms=None, normalize=True, spatial_ndim=3):
        self.ts_dataset = ts_dataset
        self.input_norms = input_norms
        self.normalize = normalize
        self.spatial_ndim = spatial_ndim

    def chain(self):
        return list(g.input_norms if self.input_norms is None else self.input_norms)

    def norms_to_apply(self):
        return self.chain() if self.normalize else []

    def selected(self):
        """The raw (unnormalized) tensorstore for the channel the chain selects."""
        return select_channel(
            self.ts_dataset, selected_channel(self.chain()), self.spatial_ndim
        )

    def __getitem__(self, index):
        result = self.selected()[index]
        if not self.normalize:
            return result
        return apply_norms(result, self.norms_to_apply())

    def __getattr__(self, attr):
        if attr in ("ts_dataset", "input_norms", "normalize", "spatial_ndim"):
            # Not set yet (e.g. while unpickling); don't recurse.
            raise AttributeError(attr)
        at = getattr(self.selected(), attr)
        if attr == "dtype":
            # The last step that declares a dtype decides; steps without one
            # (ChannelSelector) pass their input's through.
            for norm in reversed(self.norms_to_apply()):
                if norm.dtype is not None:
                    return np.dtype(norm.dtype)
            return np.dtype(at.numpy_dtype)
        return at


def _detect_filetype(dataset_path: str) -> str:
    """"n5" when the last container suffix in the path is .n5, else "zarr"."""
    return "n5" if paths.suffix_format(dataset_path) == "n5" else "zarr"


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

    if paths.is_precomputed(dataset_path):
        # precomputed:// (a local directory) and gs:// URLs are neuroglancer
        # precomputed volumes; a trailing /s<N> picks the scale.
        kvstore, scale_index = paths.precomputed_kvstore(dataset_path)
        filetype = "neuroglancer_precomputed"
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
    else:
        kvstore = {
            "driver": "file",
            "path": os.path.normpath(dataset_path),
        }

    is_v3 = (
        filetype == "zarr"
        and isinstance(kvstore, dict)
        and kvstore.get("driver") == "file"
        and zarr_v3.is_v3_container(kvstore["path"])
    )
    if is_v3:
        filetype = "zarr3"

    # tensorstore rejects compressor fields it doesn't know ("extra
    # members", e.g. numcodecs' zstd checksum), so such arrays are opened
    # with their metadata minus those fields.
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
    ts_dataset = dataset_future.result()

    if filetype in ("n5", "neuroglancer_precomputed"):
        # Both drivers expose Fortran order (x, y, z[, channel]); everything
        # else here -- the metadata readers, the ROI math, neuroglancer's axis
        # names -- is C order (z, y, x). Reading an N5 dataset without this
        # returned x/z-transposed data, and precomputed lost its x axis to the
        # channel selection.
        ts_dataset = ts_dataset[ts.d[:].transpose[::-1]]

    if normalize:
        return LazyNormalization(ts_dataset)
    # Unnormalized callers still get one spatial volume, as before; the channel
    # comes from the current chain.
    return select_channel(ts_dataset, selected_channel(g.input_norms))


def to_ndarray_tensorstore(
    dataset,
    roi=None,
    voxel_size=None,
    offset=None,
    output_voxel_size=None,
    axes_names=["z", "y", "x"],
    custom_fill_value=None,
    input_norms=None,
):
    """Read a region of a tensorstore dataset and return it as a numpy array

    Args:
        dataset ('tensorstore.dataset'): Tensorstore dataset, or the
            LazyNormalization view ``open_ds_tensorstore`` returns
        roi ('funlib.geometry.Roi'): Region of interest to read
        input_norms: normalizers to apply to what is read. ``None`` means the
            view's own chain for a LazyNormalization, and ``g.input_norms``
            for a bare tensorstore; pass ``[]`` for raw values.

    Returns:
        Numpy array of the region
    """
    if isinstance(dataset, LazyNormalization):
        if input_norms is None:
            input_norms = dataset.norms_to_apply()
        dataset = dataset.selected()
    elif input_norms is None:
        input_norms = g.input_norms

    if roi is None:
        with ts.Transaction() as txn:
            data = dataset.with_transaction(txn).read().result()
        for norm in input_norms:
            data = norm(data)
        return data

    if offset is None:
        offset = np.zeros(roi.dims)

    if output_voxel_size is None:
        output_voxel_size = voxel_size

    rescale_factor = 1
    if not zarr_v3.same_voxel_size(voxel_size, output_voxel_size):
        # in the case where there is a mismatch in voxel sizes, we may need to extra pad to ensure that the output is a multiple of the output voxel size
        voxel_size = Coordinate(voxel_size)
        output_voxel_size = Coordinate(output_voxel_size)
        original_roi = roi
        roi = original_roi.snap_to_grid(voxel_size)
        rescale_factor = voxel_size[0] / output_voxel_size[0]
        snapped_offset = (original_roi.begin - roi.begin) / output_voxel_size
        snapped_end = (original_roi.end - roi.begin) / output_voxel_size
        snapped_slices = tuple(
            slice(snapped_offset[i], snapped_end[i]) for i in range(3)
        )

    # World nm -> voxel indices, in floats: dividing a Roi by a Coordinate
    # first truncated a 5.24 nm voxel size to 5. Rounding is unchanged for
    # integer sizes (truncation, except that float noise like 9.9999999 is 10).
    voxel_size_f = np.asarray(voxel_size, dtype=float)
    begin = np.trunc(
        zarr_v3.snap_integral(
            (np.asarray(roi.begin, dtype=float) - np.asarray(offset, dtype=float))
            / voxel_size_f
        )
    )
    size = np.trunc(zarr_v3.snap_integral(np.asarray(roi.shape, dtype=float) / voxel_size_f))

    # Specify the range
    roi_slices = tuple(slice(int(b), int(b + s)) for b, s in zip(begin, size))

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

    # Padding for the part of the ROI outside the array. It was only assigned
    # when the array's own fill_value was falsy, so every border read of an
    # array with, say, fill_value=255 raised UnboundLocalError.
    fill_value = custom_fill_value if custom_fill_value else 0
    with ts.Transaction() as txn:
        data = dataset.with_transaction(txn)[valid_slices].read().result()
    for norm in input_norms:
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
        node (zarr.core.Array or zarr.hierarchy.Group): _description_

    Raises:
        RuntimeError: returned if the node array is in the parent group,
        or the group itself is the root group

    Returns:
        zarr.hierarchy.Group : parent group that contains input group/array
    """

    path = get_url(node)

    store_path, node_path = separate_store_path(path, node.path)
    if node_path == "":
        raise RuntimeError(f"{node.name} is in the root group of the {path} store.")
    else:
        if store_path.endswith(".n5"):
            store_path = N5FSStore(store_path)
        return zarr.open(store=store_path, path=os.path.split(node_path)[0], mode="r")


def _parent_or_none(node):
    """``access_parent(node)``, or None for a node at the root of its store.

    An array at the container root has no parent group, and access_parent
    raising there made the whole attribute lookup fall back to voxel size 1
    and offset 0, ignoring the array's own resolution/offset attrs.
    """
    try:
        return access_parent(node)
    except (RuntimeError, ValueError):
        return None


def _self_and_parent(array):
    parent = _parent_or_none(array)
    return [array] if parent is None else [array, parent]


def check_for_multiscale(group):
    """check if multiscale attribute exists in the input group and for any parent level group

    Args:
        group (zarr.hierarchy.Group): group to check

    Returns:
        tuple({}, zarr.hierarchy.Group): (multiscales attribute body, zarr group where multiscales was found)
    """
    multiscales = group.attrs.get("multiscales", None)

    if multiscales:
        return (multiscales, group)

    if group.path == "":
        return (multiscales, group)

    return check_for_multiscale(access_parent(group))


def _items(array):
    """The attributes of ``array`` and of its parent group, if it has one."""
    return [dict(item.attrs) for item in _self_and_parent(array)]


def check_for_voxel_size(array, order):
    """The voxel size in ``array``'s own or its parent's attributes (see
    io.metadata.n5_voxel_size), or None."""
    return metadata.n5_voxel_size(_items(array), order)


def check_for_offset(array, order):
    """The offset in ``array``'s own or its parent's attributes (see
    io.metadata.n5_offset), or None."""
    return metadata.n5_offset(_items(array), order)


def check_for_units(array, order):
    """The units in ``array``'s own or its parent's attributes, "pixels"
    without any (see io.metadata.n5_units)."""
    return metadata.n5_units(_items(array), order, len(array.shape))


def get_ds_info(path: str, mode: str = "r"):
    """Metadata of one array: ``(voxel_size, chunk_shape, shape, roi,
    axes_names, filetype)``.

    Spatial axes only, in C order (z, y, x), sizes and offsets in nanometers.
    Reads zarr v2 and v3 (local), zarr v2 over http(s) or anonymous s3, N5,
    and neuroglancer precomputed (local ``precomputed://`` or ``gs://``).

    ``voxel_size`` is a Coordinate when it is a whole number of nanometers
    and a tuple of floats otherwise (Coordinate would truncate 5.24 to 5);
    ``roi`` is the integer-nm box covering the array. read_ds_meta() has the
    exact float offset.
    """
    return zarr_v3.legacy_ds_info(read_ds_meta(path, mode), path)


def read_ds_meta(path: str, mode: str = "r"):
    """``(voxel_size, offset, chunk_shape, shape, axes_names, filetype)`` for
    one array, voxel size and offset as nanometer floats (see get_ds_info).

    ``mode`` is ignored: metadata is only ever read.
    """
    return zarr_v3.legacy_meta(metadata.read_array_meta(path))
