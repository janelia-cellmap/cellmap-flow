# %%
import tensorflow.compat.v1 as tf
import os
import numpy as np
from funlib.geometry import Roi
import json
import time
from funlib.geometry import Coordinate


def load_eval_model(setup_dir, checkpoint, device=None):
    graph = tf.Graph()
    session = tf.Session(graph=graph)

    with graph.as_default():
        meta_graph_file = os.path.join(setup_dir, "config.meta")  # f"{checkpoint}.meta"
        saver = tf.train.import_meta_graph(meta_graph_file, clear_devices=True)
        saver.restore(session, os.path.join(setup_dir, checkpoint))
        all_names = list(n.name for n in tf.get_default_graph().as_graph_def().node)
        print([n for n in all_names if "Placeholder" in n])
    return session


def process_lsd(
    chunk,
    session,
    input_tensorname=["raw"],
    output_tensorname=["embedding"],
):
    inputs = [rescale_data(chunk, 158, 233)]
    output_data = session.run(
        {ot: ot for ot in output_tensorname},
        feed_dict={k: v for k, v in zip(input_tensorname, inputs)},
    )
    output_data = output_data[output_tensorname[0]].clip(0, 1) * 255.0
    return output_data.astype(np.uint8)


output_voxel_size = Coordinate((8, 8, 8))
voxel_size = Coordinate((8, 8, 8))
input_voxel_size = Coordinate((8, 8, 8))

read_shape_lsd = Coordinate((268, 268, 268)) * voxel_size
read_shape = read_shape_lsd
lsd_output_shape = read_shape_lsd - Coordinate(104, 104, 104) * voxel_size

read_shape_acrlsd = Coordinate((160, 160, 160)) * voxel_size
write_shape = Coordinate((56, 56, 56)) * output_voxel_size
context = (read_shape - write_shape) / 2

output_channels = 3
block_shape = np.array((56, 56, 56, output_channels))
model = None


# from https://github.com/neptunes5thmoon/lsd/blob/modern/lsd/tutorial/example_nets/fib25/lsd/inference.py
# /groups/saalfeld/home/heinrichl/Brew/miniconda/envs/fly-organelles-lsd/bin/python /groups/saalfeld/home/heinrichl/dev/fly-organelles-lsd/lsd/lsd/tutorial/example_nets/fib25/lsd/inference.py predict --setup_dir /nrs/saalfeld/heinrichl/fly_organelles/lsd/networks/fib25/lsd --checkpoint train_net_checkpoint_400000 -input_tensor raw -output_tensor embedding -no 1 -cs 0-10:embedding/s0 -vs 8 8 8 -oc jrc_fly-vnc-1.zarr -od 400000_contrast -ic /nrs/cellmap/data/jrc_fly-vnc-1/jrc_fly-vnc-1.zarr -id recon-1/em/fibsem-uint8/s1 --min-raw 158 --max-raw 233
setup_dir_lsd = "/nrs/cellmap/ackermand/downsample_larissas_networks/lsd"
checkpoint_lsd = "train_net_checkpoint_400000"
input_tensor_lsd = ["raw"]
output_tensor_lsd = ["embedding"]
session_lsd = load_eval_model(setup_dir_lsd, checkpoint_lsd)

setup_dir_acrlsd = "/nrs/cellmap/ackermand/downsample_larissas_networks/acrlsd"
checkpoint_acrlsd = "train_net_checkpoint_300000"
input_tensor_acrlsd = ["raw", "pretrained_lsd"]
output_tensor_acrlsd = ["affs"]
session_acrlsd = load_eval_model(setup_dir_acrlsd, checkpoint_acrlsd)


# /groups/saalfeld/home/heinrichl/Brew/miniconda/envs/fly-organelles-lsd/bin/python /groups/saalfeld/home/heinrichl/dev/fly-organelles-lsd/lsd/lsd/tutorial/example_nets/fib25/acrlsd/inference.py predict --setup_dir /nrs/saalfeld/heinrichl/fly_organelles/lsd/networks/fib25/acrlsd --checkpoint train_net_checkpoint_300000 -input_tensor raw -output_tensor affs -input_tensor pretrained_lsd -no 1 -cs 0-3:affs/s0 -vs 8 8 8 -oc jrc_fly-vnc-1.zarr -od 400000 -ic /nrs/cellmap/data/jrc_fly-vnc-1/jrc_fly-vnc-1.zarr -id recon-1/em/fibsem-uint8/s1 -ic /nrs/saalfeld/heinrichl/fly_organelles/lsd/networks/fib25/lsd/jrc_fly-vnc-1.zarr -id 400000/embedding/s0 --min-raw 158 --max-raw 233 --min-raw 0 --max-raw 255 --local
def process_lsd(
    chunk,
):
    input_tensorname, output_tensorname = get_tensor_names(
        setup_dir_lsd, input_tensor_lsd, output_tensor_lsd
    )
    inputs = [rescale_data(chunk, 158, 233)]
    output_data = session_lsd.run(
        {ot: ot for ot in output_tensorname},
        feed_dict={k: v for k, v in zip(input_tensorname, inputs)},
    )
    output_data = output_data[output_tensorname[0]].clip(0, 1) * 255.0
    return output_data.astype(np.uint8)


def rescale_data(input, minr, maxr):
    shift = minr
    scale = maxr - minr
    input_rescaled = (2.0 * (input.astype(np.float32) - shift) / scale) - 1.0
    return input_rescaled


def get_tensor_names(setup_dir, input_tensor, output_tensor):
    with open(os.path.join(setup_dir, "config.json"), "r") as f:
        net_config = json.load(f)
        output_tensorname = []
        input_tensorname = []
        for ot in output_tensor:
            output_tensorname.append(net_config[ot])
        for it in input_tensor:
            input_tensorname.append(net_config[it])

    return input_tensorname, output_tensorname


def process_chunk(idi, input_roi):
    # read in raw
    input_roi = input_roi.grow(context, context)
    chunk = idi.to_ndarray_ts(input_roi)

    # process lsd
    lsd_output = process_lsd(chunk)
    print(f"{lsd_output.shape=}")
    # trim lsd_output
    excess_voxels = (lsd_output_shape - read_shape_acrlsd) / (
        Coordinate(2, 2, 2) * voxel_size
    )
    slicer = np.s_[excess_voxels[0] : -excess_voxels[0]]
    lsd_output = lsd_output[:, slicer, slicer, slicer]
    print(f"{lsd_output.shape=}")

    # trim raw
    excess_voxels = (read_shape_lsd - read_shape_acrlsd) / (
        Coordinate(2, 2, 2) * voxel_size
    )
    print(excess_voxels)
    slicer = np.s_[excess_voxels[0] : -excess_voxels[0]]
    chunk = chunk[slicer, slicer, slicer]
    print(f"{chunk.shape=}")

    inputs = [
        rescale_data(chunk, 153, 233),
        rescale_data(lsd_output, 0, 255),
    ]

    input_tensorname, output_tensorname = get_tensor_names(
        setup_dir_acrlsd, input_tensor_acrlsd, output_tensor_acrlsd
    )

    output_data = session_acrlsd.run(
        {ot: ot for ot in output_tensorname},
        feed_dict={k: v for k, v in zip(input_tensorname, inputs)},
    )
    output_data = output_data[output_tensorname[0]].clip(0, 1) * 255.0
    return output_data.astype(np.uint8)