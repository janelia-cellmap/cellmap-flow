#%%

from dacapo.experiments import Run
from dacapo.store.create_store import create_config_store, create_weights_store
from daisy.coordinate import Coordinate
import numpy as np

def get_dacapo_run_model(run_name, iteration):
    config_store = create_config_store()
    run_config = config_store.retrieve_run_config(run_name)
    run = Run(run_config)
    if iteration > 0:
        weights_store = create_weights_store()
        weights = weights_store.retrieve_weights(run, iteration)
        run.model.load_state_dict(weights.model)

    return run

model_name = "20241204_finetune_mito_affs_task_datasplit_v3_u21_kidney_mito_default_cache_8_1"
channels = ["mito"]
checkpoint= 700000
run = get_dacapo_run_model(model_name, checkpoint)
model = run.model

in_shape =  model.eval_input_shape
out_shape = model.compute_output_shape(in_shape)[1]

voxel_size = run.datasplit.train[0].raw.voxel_size
read_shape = Coordinate(in_shape) * Coordinate(voxel_size)
write_shape = Coordinate(out_shape) * Coordinate(voxel_size)
output_voxel_size = Coordinate(model.scale(voxel_size))

#%%
import torch
#%%



if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("device:", device)    

model.to(device)
model.eval()


output_channels = len(channels)  # 0:all_mem,1:organelle,2:mito,3:er,4:nucleus,5:pm,6:vs,7:ld
block_shape = np.array(tuple(out_shape) +(output_channels,))
# %%

