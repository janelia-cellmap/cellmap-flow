# %%
import tensorflow.compat.v1 as tf
import os
import numpy as np
import json
from funlib.geometry import Coordinate
from cellmap_flow.image_data_interface import ImageDataInterface
from funlib.geometry import Roi


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


output_voxel_size = Coordinate((8, 8, 8))
voxel_size = Coordinate((8, 8, 8))
input_voxel_size = Coordinate((8, 8, 8))

read_shape = Coordinate((268, 268, 268)) * voxel_size
write_shape = Coordinate((164, 164, 164)) * output_voxel_size
context = (read_shape - write_shape) / 2

output_channels = 10
block_shape = np.array((164, 164, 164, output_channels))
model = None

setup_dir = "/nrs/cellmap/ackermand/downsample_larissas_networks/lsd"
checkpoint_lsd = "train_net_checkpoint_400000"
input_tensor_lsd = ["raw"]
output_tensor_lsd = ["embedding"]
session = load_eval_model(setup_dir, checkpoint_lsd)


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


def process_lsd(
    chunk,
    session,
    input_tensorname,
    output_tensorname,
):
    inputs = [rescale_data(chunk, 158, 233)]
    output_data = session.run(
        {ot: ot for ot in output_tensorname},
        feed_dict={k: v for k, v in zip(input_tensorname, inputs)},
    )
    output_data = output_data[output_tensorname[0]].clip(0, 1) * 255.0
    return output_data.astype(np.uint8)


def process_chunk(idi: ImageDataInterface, input_roi: Roi):
    input_roi = input_roi.grow(context, context)
    chunk = idi.to_ndarray_ts(input_roi)
    input_tensorname, output_tensorname = get_tensor_names(
        setup_dir, input_tensor_lsd, output_tensor_lsd
    )
    return process_lsd(chunk, session, input_tensorname, output_tensorname)
