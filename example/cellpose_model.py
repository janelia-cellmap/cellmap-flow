# %%
from funlib.geometry import Coordinate
import numpy as np
import cellpose.models

from cellmap_flow.image_data_interface import ImageDataInterface

input_voxel_size = Coordinate((426, 233, 233))
output_voxel_size = Coordinate((426, 233, 233))
padding = 10
read_shape = (
    Coordinate((70 + 2 * padding, 128 + 2 * padding, 128 + 2 * padding))
    * input_voxel_size
)
write_shape = Coordinate((70, 128, 128)) * output_voxel_size
context = (read_shape - write_shape) / 2

output_channels = 1
block_shape = np.array((70, 128, 128, output_channels))

# set model arguments
model_kwargs = {"gpu": True, "model_type": "cyto2"}  # , "net_avg": False}

# set evaluation arguments
eval_kwargs = {
    "diameter": 40,
    "anisotropy": input_voxel_size[0] / input_voxel_size[1],
    "z_axis": 0,
    "channels": [0, 0],
    "do_3D": True,
    "min_size": 15000,
    "cellprob_threshold": 1.0,
    # "net_avg": False,
}

model = cellpose.models.CellposeModel(**model_kwargs)


def process_chunk(idi: ImageDataInterface, input_roi):
    input_roi = input_roi.grow(context, context)
    data = idi.to_ndarray_ts(input_roi)
    output = model.eval(data, **eval_kwargs)[0]
    # trim output
    output = output[padding:-padding, padding:-padding, padding:-padding]
    # insert channel dim
    output = np.expand_dims(output, axis=0)
    return output


# # %%
# import cellmap_flow.image_data_interface
# import cellmap_flow.utils.ds
# from importlib import reload

# reload(cellmap_flow.utils.ds)
# reload(cellmap_flow.image_data_interface)
# from cellmap_flow.image_data_interface import ImageDataInterface
# from funlib.geometry import Roi

# idi = ImageDataInterface("/nrs/cellmap/ackermand/fromGreg.n5/c3/s0")
# output = process_chunk(idi, Roi((0, 0, 0), read_shape))

# %%
