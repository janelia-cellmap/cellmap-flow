#%%
# TODO
setup_dir = ""
checkpoint = ""
input_tensor = ""
output_tensor = ""

input_dataset = ""
shifts = ""
scales = ""

# pip install fly-organelles
from daisy.coordinate import Coordinate
import numpy as np
voxel_size = (8, 8, 8)
read_shape = Coordinate((178, 178, 178)) * Coordinate(voxel_size)
write_shape = Coordinate((56, 56, 56)) * Coordinate(voxel_size)
output_voxel_size = Coordinate((8, 8, 8))


import tensorflow.compat.v1 as tf
import os
import json



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

with open(os.path.join(setup_dir, "config.json"), "r") as f:
    net_config = json.load(f)
    output_tensorname = []
    input_tensorname = []
    for ot in output_tensor:
        output_tensorname.append(net_config[ot])
    for it in input_tensor:
        input_tensorname.append(net_config[it])

session = load_eval_model(setup_dir,checkpoint)

def predict(read_roi,write_roi, config):
    inputs = []
    for in_dataset_iter, shift, scale in zip(input_dataset, shifts, scales):
        inputs.append(
            (2.0
            * (
                in_dataset_iter.to_ndarray(
                    roi=read_roi, fill_value=shift + scale
                ).astype(np.float32)
                - shift
            )
            / scale) - 1.0
        )

    output_data = config.session.run(
                    {ot: ot for ot in config.output_tensorname},
                    feed_dict={k:v for k, v in zip(config.input_tensorname, config.inputs)},
                )
    return output_data



#%%
import torch
from fly_organelles.model import StandardUnet
#%%
def load_eval_model(num_labels, checkpoint_path):
    model_backbone = StandardUnet(num_labels)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("device:", device)    
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=device)
    model_backbone.load_state_dict(checkpoint["model_state_dict"])
    model = torch.nn.Sequential(model_backbone, torch.nn.Sigmoid())
    model.to(device)
    model.eval()
    return model


classes = ["mito","er","nuc"," pm"," ves","ld"]
CHECKPOINT_PATH = "/nrs/saalfeld/heinrichl/fly_organelles/run07/model_checkpoint_700000"
# output_channels = len(classes) 
output_channels = 8
model = load_eval_model(output_channels, CHECKPOINT_PATH)
block_shape = np.array((56, 56, 56,output_channels))
# %%
# print("model loaded",model)
# %%
