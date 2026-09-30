# %%
import logging
import os
from typing import Sequence, Union

import numpy as np
import tensorstore as ts
import zarr
from zarr.n5 import N5FSStore

from cellmap_flow.globals import g
from cellmap_flow.io import metadata, multiscale, ome, paths
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
    lower corner at ``offset`` (see io.ome.singlescale_attrs)."""
    return ome.singlescale_attrs(arr_name, voxel_size, offset, units, axes)


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
