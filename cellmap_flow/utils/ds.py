# %%
import json
import logging
import os
import re
from typing import Sequence, Union

import numpy as np
import tensorstore as ts
import zarr
from funlib.geometry import Coordinate
from skimage.measure import block_reduce
from zarr.n5 import N5FSStore

from cellmap_flow.globals import g
from cellmap_flow.io import paths
from cellmap_flow.io.paths import (  # noqa: F401  (kept names; see io.paths)
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
    attrs = zarr_grp.attrs
    ms = attrs["multiscales"][0]

    # Spatial axes only (skip channel axes), in nanometers. If there is no
    # axes metadata, assume all dimensions are spatial.
    spatial_indices, _, units = zarr_v3.spatial_axes(ms.get("axes", []))

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
            resolutions[scale["path"]] = zarr_v3.to_nm(
                [full_res[i] for i in spatial_indices], units
            )
            offsets[scale["path"]] = zarr_v3.ome_corner(
                zarr_v3.to_nm([full_translation[i] for i in spatial_indices], units),
                resolutions[scale["path"]],
            )
            shapes[scale["path"]] = tuple(full_shape[i] for i in spatial_indices)
        else:
            resolutions[scale["path"]] = full_res
            offsets[scale["path"]] = zarr_v3.ome_corner(full_translation, full_res)
            shapes[scale["path"]] = full_shape
    return offsets, resolutions, shapes


def find_closest_scale(zarr_grp_path, target_resolution):
    zarr_grp = _open_zarr(zarr_grp_path, mode="r")
    offsets, resolutions, shapes = get_scale_info(zarr_grp)
    if target_resolution is None:
        # No target: the finest (first) scale, as the v3 reader does.
        target_scale = next(iter(resolutions))
        return target_scale, offsets[target_scale], shapes[target_scale]
    target_scale = None
    last_scale = None
    for scale, res in resolutions.items():
        if last_scale is None:
            last_scale = scale
        if zarr_v3.same_voxel_size(res, target_resolution):
            target_scale = scale
            break
        elif zarr_v3.coarser_anywhere(res, target_resolution):
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


def _open_zarr(path, mode="r"):
    """Open a zarr dataset, handling HTTP/HTTPS and (anonymous) S3 URLs via fsspec."""
    path = _normalize_path(path)
    if _is_remote_path(path):
        import fsspec

        options = {"anon": True} if path.startswith("s3://") else {}
        return zarr.open(fsspec.get_mapper(path, **options), mode=mode)
    return zarr.open(path, mode=mode)


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
    for item in _self_and_parent(array):

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
    for item in _self_and_parent(array):

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
    for item in _self_and_parent(array):

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
            # Spatial axes only: a channel axis has no unit (the spec allows
            # that), and item["unit"] raising KeyError on it was swallowed
            # into voxel size 1 and offset 0 for every multichannel dataset.
            spatial, _, units = zarr_v3.spatial_axes(multiscales[0].get("axes", []))

            def pick(values):
                return list(values) if spatial is None else [values[i] for i in spatial]

            for level in multiscales[0]["datasets"]:
                if level["path"].lstrip("/") == scale:
                    for attr in level["coordinateTransformations"]:
                        if attr["type"] == "scale":
                            voxel_size = pick(attr["scale"])
                        elif attr["type"] == "translation":
                            offset = pick(attr["translation"])
                    if units is None and voxel_size is not None:
                        units = [None] * len(voxel_size)
                    if voxel_size is not None:
                        # translation is voxel 0's centre, and 0 when absent.
                        offset = zarr_v3.ome_corner(
                            offset if offset is not None else [0.0] * len(voxel_size),
                            voxel_size,
                        )
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
        (voxel_size, offset, units, offset_is_ome_corner). The last is True when
        the offset is a corner converted from an OME translation, which is
        exact and must not be rounded onto the voxel grid.
    """
    voxel_size = None
    offset = None
    units = None
    multiscales = None
    from_ome = False

    if not isinstance(ds, zarr.core.Array):
        raise TypeError(
            f"{os.path.join(ds.store.path, ds.path)} is not zarr.core.Array"
        )

    # check recursively for multiscales attribute in the zarr store tree (an
    # array at the root of its store has no group above it to look in)
    parent = _parent_or_none(ds)
    if parent is None:
        multiscales, multiscale_group = None, None
    else:
        multiscales, multiscale_group = check_for_multiscale(group=parent)

    # check for attributes in .zarr group multiscale
    if not isinstance(ds.store, (zarr.n5.N5Store, zarr.n5.N5FSStore)):
        if multiscales:
            voxel_size, offset, units = check_for_attrs_multiscale(
                ds, multiscale_group, multiscales
            )
            from_ome = voxel_size is not None

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
            return voxel_size[::-1], offset[::-1], units[::-1], from_ome
        else:
            return voxel_size, offset, units, from_ome

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
        return voxel_size[::-1], offset[::-1], units[::-1], from_ome
    else:
        return voxel_size, offset, units, from_ome


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
        (Coordinate, Coordinate)): returned offset size that is multiple of voxel size.
        For a non-integer voxel size, two tuples of floats instead (the same
        rounding, without truncating the voxel size to an integer first).
    """
    snapped_voxel_size = zarr_v3.snap_integral(voxel_size_float)
    if not zarr_v3.is_integral(snapped_voxel_size):
        vs = snapped_voxel_size
        off = zarr_v3.snap_integral(offset_float)
        if not np.allclose(np.round(off / vs) * vs, off):
            logger.debug(f"Offset: {off} being rounded to nearest voxel size: {vs}")
            off = zarr_v3.snap_integral(np.trunc((off + vs / 2) / vs) * vs)
        return tuple(float(v) for v in vs), tuple(float(v) for v in off)

    voxel_size = Coordinate(int(v) for v in snapped_voxel_size)
    offset = Coordinate(offset_float)

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

    voxel_size, offset, units, from_ome = _read_attrs(ds, order)
    if isinstance(units, str) or units is None:
        units = [units] * len(voxel_size)
    # Everything downstream is in nanometers. Only the literal "um" was
    # converted, so OME-NGFF's "micrometer" came through as 0.004 "nm",
    # truncated to 0, and a divide by zero fell back to voxel size 1.
    voxel_size = zarr_v3.to_nm(voxel_size, units)
    offset = zarr_v3.to_nm(offset, units)

    if from_ome:
        # An OME corner is exact, and usually not a multiple of the voxel size
        # (-4 nm at 8 nm for Janelia data); rounding it onto the voxel grid
        # would undo the centre-to-corner conversion.
        return (
            tuple(float(v) for v in zarr_v3.snap_integral(voxel_size)),
            tuple(float(v) for v in zarr_v3.snap_integral(offset)),
        )
    return regularize_offset(voxel_size, offset)


def _ome_level_info(group, leaf, ds):
    """Metadata of array ``ds``, the ``leaf`` dataset of OME group ``group``.

    ``(voxel_size, offset, chunk_shape, shape, axes_names, "zarr")`` with
    nanometer floats, spatial axes only; None when ``group`` has no
    multiscales.
    """
    multiscales = group.attrs.get("multiscales", None)
    if not multiscales:
        return None
    ms = multiscales[0]
    spatial_indices, spatial_names, units = zarr_v3.spatial_axes(ms.get("axes", []))
    if spatial_indices is None:
        spatial_indices = list(range(len(ds.shape)))
        spatial_names = ["z", "y", "x"][-len(spatial_indices):]
        units = None

    dataset_entry = next(
        (d for d in ms["datasets"] if d["path"].strip("/") == leaf),
        ms["datasets"][0],
    )
    transforms = dataset_entry["coordinateTransformations"]
    scale_transform = next(t["scale"] for t in transforms if t["type"] == "scale")
    translation = next(
        (t["translation"] for t in transforms if t["type"] == "translation"),
        [0.0] * len(scale_transform),
    )
    voxel_size = zarr_v3.to_nm([scale_transform[i] for i in spatial_indices], units)
    offset = zarr_v3.ome_corner(
        zarr_v3.to_nm([translation[i] for i in spatial_indices], units), voxel_size
    )
    shape = tuple(ds.shape[i] for i in spatial_indices)
    chunk_shape = tuple(ds.chunks[i] for i in spatial_indices)
    return voxel_size, offset, chunk_shape, shape, list(spatial_names), "zarr"


def _attrs_info(ds, filetype="zarr", order=None):
    """Metadata of an array described by its own (or its parent's) attrs.

    ``order`` is the *axis* order of those attrs: "F" for N5. A zarr array's
    own ``order`` is its chunk memory layout, not an axis order, so it is not
    consulted (a Fortran-ordered array had its voxel size reversed); only an
    explicit ``order`` attribute is.
    """
    if order is None:
        order = ds.attrs.get("order", "C")
    try:
        voxel_size, offset = _read_voxel_size_offset(ds, order)
    except Exception as e:
        logger.error(
            "failed to read voxel size and offset for %s (%s), will use default values"
            % (getattr(ds, "path", ds), e)
        )
        voxel_size, offset = (1,) * 3, (0,) * 3
    n = len(voxel_size)
    return (
        [float(v) for v in voxel_size],
        [float(v) for v in offset],
        tuple(ds.chunks[-n:]),
        tuple(ds.shape[-n:]),
        ["z", "y", "x"][-n:],
        filetype,
    )


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
    one array, voxel size and offset as nanometer floats (see get_ds_info)."""

    path = _normalize_path(path)

    if path.startswith(("gs://", "precomputed://")):
        # open_ds_tensorstore puts these in C order and selects one channel.
        ts_info = open_ds_tensorstore(path)
        shape = tuple(ts_info.shape)
        voxel_size = [
            d.to_json()[0] * zarr_v3.nm_per_unit(d.to_json()[1]) if d is not None else 1.0
            for d in ts_info.dimension_units
        ]
        axes_names = list(ts_info.spec().transform.input_labels)
        chunk_shape = tuple(ts_info.chunk_layout.read_chunk.shape)
        file_type = "gs" if path.startswith("gs://") else "precomputed"
        return voxel_size, [0.0] * len(shape), chunk_shape, shape, axes_names, file_type

    if _is_remote_path(path):
        # http(s) and s3 alike: zarr v2 over fsspec.
        ds = _open_zarr(path, mode="r")

        # If the URL points to a zarr Group (e.g. multiscale container),
        # read OME-Zarr multiscales metadata and navigate into the first array.
        if isinstance(ds, zarr.hierarchy.Group):
            multiscales = ds.attrs.get("multiscales", None)
            if multiscales:
                leaf = multiscales[0]["datasets"][0]["path"]
                return _ome_level_info(ds, leaf.strip("/"), ds[leaf])
            for key in sorted(ds.keys()):
                if isinstance(ds[key], zarr.core.Array):
                    ds = ds[key]
                    break

        # The path points to a sub-array (e.g. .zarr/raw/s0). Its multiscales
        # live on the group right above it (raw), not necessarily the root.
        if ".zarr" in path or ".n5" in path:
            container, sub_path = split_dataset_path(path)
            if sub_path:
                sub_path = sub_path.strip("/")
                parent_path, _, leaf = sub_path.rpartition("/")
                candidates = [(parent_path, leaf)]
                if parent_path:
                    # Also a root-level multiscales naming "raw/s0" directly.
                    candidates.append(("", sub_path))
                for group_path, entry_path in candidates:
                    try:
                        group = _open_zarr(
                            _join_path(container, group_path) if group_path else container,
                            mode="r",
                        )
                        info = _ome_level_info(group, entry_path, ds)
                    except Exception as e:
                        logger.warning(
                            "failed to read parent multiscale metadata for %s: %s"
                            % (path, e)
                        )
                        continue
                    if info is not None:
                        return info

        # Fallback for remote arrays without multiscales metadata
        return _attrs_info(ds)

    v3_container = zarr_v3.find_v3_container(path)
    if v3_container is not None:
        return zarr_v3.read_ds_meta_v3(path)

    filename, ds_name = split_dataset_path(path)
    if filename.endswith(".zarr") or _is_zarr_container(filename):
        logger.debug("opening zarr dataset %s in %s", ds_name, filename)
        try:
            ds = zarr.open(filename, mode=mode)
            if ds_name:
                ds = ds[ds_name]
        except Exception as e:
            logger.error("failed to open %s/%s" % (filename, ds_name))
            raise e
        info = _attrs_info(ds)
        logger.debug("opened zarr dataset %s in %s", ds_name, filename)
        return info

    if filename.endswith(".n5"):
        logger.debug("opening N5 dataset %s in %s", ds_name, filename)
        ds = zarr.open(N5FSStore(filename), mode=mode)[ds_name]
        # N5 attributes are x, y, z; the zarr view and everything returned
        # here are z, y, x.
        info = _attrs_info(ds, filetype="n5", order="F")
        logger.debug("opened N5 dataset %s in %s", ds_name, filename)
        return info

    logger.error("don't know data format of %s in %s", ds_name, filename)
    raise RuntimeError("Unknown file format for %s" % filename)
