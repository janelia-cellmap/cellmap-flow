#%%
# pip install fly-organelles
from daisy.coordinate import Coordinate
import numpy as np

output_voxel_size = Coordinate((4, 4, 4))
voxel_size = Coordinate((8, 8, 8))

read_shape = Coordinate((216, 216, 216)) * Coordinate(voxel_size)
write_shape = Coordinate((68, 68, 68)) * Coordinate(output_voxel_size)


#%%
import torch
from dacapo.experiments.starts import CosemStartConfig
start_config = CosemStartConfig("setup04", "1820500")
# model = cosem_models.load_model('setup04/1820500')

#%%



output_channels = 14  # 0:all_mem,1:organelle,2:mito,3:er,4:nucleus,5:pm,6:vs,7:ld
block_shape = np.array((68, 68, 68,14))
# %%
start = start_config.start_type(start_config)
# %%
start.run
# %%
from cellmap_models import cosem
model = cosem.load_model("setup04")
# %%
model
# %%
start.initialize_weights(model)
# %%
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)
model.eval()