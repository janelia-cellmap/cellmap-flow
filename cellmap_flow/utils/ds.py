# %%
import zarr
from pathlib import Path
from zarr.util import TreeNode
from funlib.geometry import Coordinate
import logging
import tensorstore as ts
import numpy as np
from funlib.geometry import Coordinate
from funlib.geometry import Roi
import os
import s3fs
import re
import zarr
from skimage.measure import block_reduce
from funlib.geometry import Coordinate, Roi

import zarr
from zarr.n5 import N5FSStore
import h5py
import json
import logging
import os
from typing import Union, Sequence


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


def split_dataset_path(dataset_path, scale=None) -> tuple[str, str]:
    """Split the dataset path into the filename and dataset

    Args:
        dataset_path ('str'): Path to the dataset
        scale ('int'): Scale to use, if present

    Returns:
        Tuple of filename and dataset
    """

    # split at .zarr or .n5, whichever comes last
    splitter = (
        ".zarr" if dataset_path.rfind(".zarr") > dataset_path.rfind(".n5") else ".n5"
    )

    filename, dataset = dataset_path.split(splitter)
    if dataset.startswith("/"):
        dataset = dataset[1:]
    # include scale if present
    if scale is not None:
        dataset += f"/s{scale}"

    return filename + splitter, dataset


def open_ds_tensorstore(dataset_path: str, mode="r", concurrency_limit=None):
    # open with zarr or n5 depending on extension
    filetype = (
        "zarr" if dataset_path.rfind(".zarr") > dataset_path.rfind(".n5") else "n5"
    )
    extra_args = {}

    if dataset_path.startswith("s3://"):
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

    if mode == "r":
        dataset_future = ts.open(spec, read=True, write=False)
    else:
        dataset_future = ts.open(spec, read=False, write=True)

    if dataset_path.startswith("gs://"):
        # NOTE: Currently a hack since google store is for some reason stored as mutlichannel
        return dataset_future.result()[ts.d["channel"][0]]
    else:
        return dataset_future.result()


def to_ndarray_tensorstore(
    dataset,
    roi=None,
    voxel_size=None,
    offset=None,
    output_voxel_size=None,
    swap_axes=False,
    custom_fill_value=None,
):
    """Read a region of a tensorstore dataset and return it as a numpy array

    Args:
        dataset ('tensorstore.dataset'): Tensorstore dataset
        roi ('funlib.geometry.Roi'): Region of interest to read

    Returns:
        Numpy array of the region
    """
    if swap_axes:
        print("Swapping axes")
        if roi:
            roi = Roi(roi.begin[::-1], roi.shape[::-1])
        if offset:
            offset = Coordinate(offset[::-1])

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
    # else:
    #     padded_data = (
    #         np.ones(output_shape, dtype=dataset.dtype.numpy_dtype) * fill_value
    #     )
    #     padded_slices = tuple(
    #         slice(valid_slice.start - s.start, valid_slice.stop - s.start)
    #         for s, valid_slice in zip(roi_slices, valid_slices)
    #     )

    #     # Read the region of interest from the dataset
    #     padded_data[padded_slices] = dataset[valid_slices].read().result()

    if rescale_factor > 1:
        rescale_factor = voxel_size[0] / output_voxel_size[0]
        data = (
            data.repeat(rescale_factor, 0)
            .repeat(rescale_factor, 1)
            .repeat(rescale_factor, 2)
        )
        data = data[snapped_slices]

    elif rescale_factor < 1:
        data = block_reduce(data, block_size=int(1 / rescale_factor), func=np.median)
        data = data[snapped_slices]

    if swap_axes:
        data = np.swapaxes(data, 0, 2)

    return data


# def get_ds_info(path, ome_zarr=True):
#     swap_axes = False
#     if path.startswith("s3://"):
#         ts_info = open_ds_tensorstore(path)
#         shape = ts_info.shape
#         path, filename = split_dataset_path(path)
#         filename, scale = filename.rsplit("/s")
#         scale = int(scale)
#         fs = s3fs.S3FileSystem(
#             anon=True
#         )  # Set anon=True if you don't need authentication
#         store = s3fs.S3Map(root=path, s3=fs)
#         zarr_dataset = zarr.open(
#             store,
#             mode="r",
#         )
#         multiscale_attrs = zarr_dataset[filename].attrs.asdict()
#         if "multiscales" in multiscale_attrs:
#             multiscales = multiscale_attrs["multiscales"][0]
#             axes = [axis["name"] for axis in multiscales["axes"]]
#             for scale_info in multiscale_attrs["multiscales"][0]["datasets"]:
#                 if scale_info["path"] == f"s{scale}":
#                     voxel_size = Coordinate(
#                         scale_info["coordinateTransformations"][0]["scale"]
#                     )
#         if axes[:3] == ["x", "y", "z"]:
#             swap_axes = True
#         chunk_shape = Coordinate(ts_info.chunk_layout.read_chunk.shape)
#         roi = Roi((0, 0, 0), Coordinate(shape) * voxel_size)
#     elif path.startswith("gs://"):
#         ts_info = open_ds_tensorstore(path)
#         shape = ts_info.shape
#         voxel_size = Coordinate(
#             (d.to_json()[0] if d is not None else 1 for d in ts_info.dimension_units)
#         )
#         if ts_info.spec().transform.input_labels[:3] == ("x", "y", "z"):
#             swap_axes = True
#         chunk_shape = Coordinate(ts_info.chunk_layout.read_chunk.shape)
#         roi = Roi([0] * len(shape), Coordinate(shape) * voxel_size)
#     else:
#         # TODO change with just get metadata
#         ds = open_dataset(path, ome_zarr=ome_zarr)
#         voxel_size = ds.voxel_size
#         chunk_shape = ds.chunk_shape
#         roi = ds.roi
#         shape = ds.shape
#     if swap_axes:
#         voxel_size = Coordinate(voxel_size[::-1])
#         chunk_shape = Coordinate(chunk_shape[::-1])
#         shape = shape[::-1]
#         roi = Roi(roi.begin[::-1], roi.shape[::-1])
#     return voxel_size, chunk_shape, shape, roi, swap_axes


# def get_resolutions(group):
#     if type(group) is str:
#         group = zarr.open(group)
#     result = []
#     for t in TreeNode(z).get_children():
#         result.append(t.obj.name.lstrip("/"))

#     return result


# def get_right_resolution(group, resolution):

#     if type(group) is str:
#         group = Path(group)
#     if not isinstance(resolution, tuple):
#         raise ValueError(f"resolution must be a tuple got {resolution} ({type(resolution).__name__})")

#     if type(resolution) is tuple:
#         resolution = Coordinate(resolution)

#     if not isinstance(zarr.open(group), zarr.hierarchy.Group):
#         raise ValueError(f"{group} is not a zarr group")

#     subfolders = get_resolutions(group)
#     for s_r in subfolders:
#         # TODO
#         r = open_ome_ds(group, s_r).voxel_size
#         if r[0] > resolution[0]:
#             raise ValueError(f"{group} resolution {resolution} is larger than {s_r} {r}")
#         equals = all([r[i] == resolution[i] for i in range(len(r))])
#         if equals:
#             return group / s_r
#     raise ValueError(f"no resolution found for {resolution} in {group}")


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
    parent_group = access_parent(array)
    for item in [array, parent_group]:

        if "resolution" in item.attrs:
            return item.attrs["resolution"]
        elif "scale" in item.attrs:
            return item.attrs["scale"]
        elif "pixelResolution" in item.attrs:
            return item.attrs["pixelResolution"]["dimensions"]
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
        elif "pixelResolution" in item.attrs:
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

    return regularize_offset(voxel_size, offset)


def get_ds_info(path: str, mode: str = "r"):
    """Open a Zarr, N5, or HDF5 dataset as an :class:`Array`. If the
    dataset has attributes ``resolution`` and ``offset``, those will be
    used to determine the meta-information of the returned array.

    Args:

        filename:

            The name of the container "file" (which is a directory for Zarr and
            N5).

        ds_name:

            The name of the dataset to open.

    Returns:

        A :class:`Array` pointing to the dataset.
    """
    # TODO
    swap_axes = False

    filename, ds_name = split_dataset_path(path)

    if filename.endswith(".zarr") or filename.endswith(".zip"):
        assert (
            not filename.endswith(".zip") or mode == "r"
        ), "Only reading supported for zarr ZipStore"

        logger.debug("opening zarr dataset %s in %s", ds_name, filename)
        try:
            ds = zarr.open(filename, mode=mode)[ds_name]
        except Exception as e:
            logger.error("failed to open %s/%s" % (filename, ds_name))
            raise e

        try:
            order = ds.attrs["order"]
        except KeyError:
            order = ds.order
        voxel_size, offset = _read_voxel_size_offset(ds, order)
        shape = Coordinate(ds.shape[-len(voxel_size) :])
        roi = Roi(offset, voxel_size * shape)

        chunk_shape = ds.chunks

        logger.debug("opened zarr dataset %s in %s", ds_name, filename)
        return voxel_size, chunk_shape, shape, roi, swap_axes

    elif filename.endswith(".n5"):
        logger.debug("opening N5 dataset %s in %s", ds_name, filename)
        ds = zarr.open(N5FSStore(filename), mode=mode)[ds_name]

        voxel_size, offset = _read_voxel_size_offset(ds, "F")
        shape = Coordinate(ds.shape[-len(voxel_size) :])
        roi = Roi(offset, voxel_size * shape)

        chunk_shape = ds.chunks

        logger.debug("opened N5 dataset %s in %s", ds_name, filename)
        return voxel_size, chunk_shape, shape, roi, swap_axes

    elif filename.endswith(".h5") or filename.endswith(".hdf"):
        logger.debug("opening H5 dataset %s in %s", ds_name, filename)
        ds = h5py.File(filename, mode=mode)[ds_name]

        voxel_size, offset = _read_voxel_size_offset(ds, "C")
        shape = Coordinate(ds.shape[-len(voxel_size) :])
        roi = Roi(offset, voxel_size * shape)

        chunk_shape = ds.chunks

        logger.debug("opened H5 dataset %s in %s", ds_name, filename)
        return voxel_size, chunk_shape, shape, roi, swap_axes

    elif filename.endswith(".json"):
        logger.debug("found JSON container spec")
        with open(filename, "r") as f:
            spec = json.load(f)
        assert "container" in spec, "JSON spec must contain 'container' key"
        return get_ds_info(spec["container"], ds_name, mode=mode)

    else:
        logger.error("don't know data format of %s in %s", ds_name, filename)
        raise RuntimeError("Unknown file format for %s" % filename)
