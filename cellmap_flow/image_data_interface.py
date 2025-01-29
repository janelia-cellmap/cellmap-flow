# %%
import logging
import tensorstore as ts
import numpy as np
from funlib.geometry import Coordinate
from funlib.geometry import Roi
import os
import s3fs
import re
import zarr

# Ensure tensorstore does not attempt to use GCE credentials
os.environ["GCE_METADATA_ROOT"] = "metadata.google.internal.invalid"

from funlib.persistence import open_ds
from skimage.measure import block_reduce

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


def get_ds_info(path):
    swap_axes = False
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
        if axes[:3] == ["x", "y", "z"]:
            swap_axes = True
        chunk_shape = Coordinate(ts_info.chunk_layout.read_chunk.shape)
        roi = Roi((0, 0, 0), Coordinate(shape) * voxel_size)
    elif path.startswith("gs://"):
        ts_info = open_ds_tensorstore(path)
        shape = ts_info.shape
        voxel_size = Coordinate(
            (d.to_json()[0] if d is not None else 1 for d in ts_info.dimension_units)
        )
        if ts_info.spec().transform.input_labels[:3] == ("x", "y", "z"):
            swap_axes = True
        chunk_shape = Coordinate(ts_info.chunk_layout.read_chunk.shape)
        roi = Roi([0] * len(shape), Coordinate(shape) * voxel_size)
    else:
        path, filename = split_dataset_path(path)
        logger.info(f"Opening {path} {filename}")
        ds = open_ds(path, filename)
        voxel_size = ds.voxel_size
        chunk_shape = ds.chunk_shape
        roi = ds.roi
        shape = ds.shape
    if swap_axes:
        voxel_size = Coordinate(voxel_size[::-1])
        chunk_shape = Coordinate(chunk_shape[::-1])
        shape = shape[::-1]
        roi = Roi(roi.begin[::-1], roi.shape[::-1])
    return voxel_size, chunk_shape, shape, roi, swap_axes


class ImageDataInterface:
    def __init__(
        self,
        dataset_path,
        mode="r",
        output_voxel_size=None,
        custom_fill_value=None,
        concurrency_limit=1,
    ):
        self.path = dataset_path
        self.filetype = (
            "zarr" if dataset_path.rfind(".zarr") > dataset_path.rfind(".n5") else "n5"
        )
        self.swap_axes = self.filetype == "n5"
        self.ts = None
        self.voxel_size, self.chunk_shape, self.shape, self.roi, self.swap_axes = (
            get_ds_info(dataset_path)
        )
        self.offset = self.roi.offset
        self.custom_fill_value = custom_fill_value
        self.concurrency_limit = concurrency_limit
        if output_voxel_size is not None:
            self.output_voxel_size = output_voxel_size
        else:
            self.output_voxel_size = self.voxel_size

    def to_ndarray_ts(self, roi=None):
        if not self.ts:
            self.ts = open_ds_tensorstore(
                self.path, concurrency_limit=self.concurrency_limit
            )
        res = to_ndarray_tensorstore(
            self.ts,
            roi,
            self.voxel_size,
            self.offset,
            self.output_voxel_size,
            self.swap_axes,
            self.custom_fill_value,
        )
        self.ts = None
        return res
