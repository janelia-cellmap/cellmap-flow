#%%
from daisy.coordinate import Coordinate
import numpy as np
import torch

channels = ['ecs', 'pm', 'mito', 'mito_mem', 'ves', 'ves_mem', 'endo', 'endo_mem', 'er', 'er_mem', 'eres', 'nuc', 'mt', 'mt_out']

# %%
from dacapo.experiments.tasks import DistanceTaskConfig

task_config = DistanceTaskConfig(
    name="tmp_cosem_distance",
    channels=channels ,
    clip_distance=40.0,
    tol_distance=40.0,
    scale_factor=80.0,
)

# %%
from dacapo.experiments.architectures import CNNectomeUNetConfig

architecture_config = CNNectomeUNetConfig(
    name="upsample_unet",
    input_shape=Coordinate(216, 216, 216),
    eval_shape_increase=Coordinate(72, 72, 72),
    fmaps_in=1,
    num_fmaps=12,
    fmaps_out=72,
    fmap_inc_factor=6,
    downsample_factors=[(2, 2, 2), (3, 3, 3), (3, 3, 3)],
    constant_upsample=True,
    upsample_factors=[(2, 2, 2)],
)

#%%
from dacapo.experiments.starts import CosemStartConfig

start_config = CosemStartConfig("setup04", "1820500")
# %%
from dacapo.experiments import RunConfig
from dacapo.experiments.run import Run


run_config = RunConfig(
    task_config=task_config,
    architecture_config=architecture_config,
    start_config=start_config,
)



run = Run(run_config)
model = run.model
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)
model.eval()

#%%
# pip install fly-organelles


output_voxel_size = Coordinate((4, 4, 4))
voxel_size = Coordinate((8, 8, 8))

read_shape = Coordinate((216, 216, 216)) * Coordinate(voxel_size)
write_shape = Coordinate((68, 68, 68)) * Coordinate(output_voxel_size)



output_channels = 14  # 0:all_mem,1:organelle,2:mito,3:er,4:nucleus,5:pm,6:vs,7:ld
block_shape = np.array((68, 68, 68,14))

# %%
start_config.
# %%
from dacapo.store.create_store import create_config_store
# %%
config_store = create_config_store()
# %%
config_store.retrieve_run_config_names