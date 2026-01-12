# %%
path = "gs://fish2-derived/em_sofima_240112_raw/"
from cellmap_flow.utils.ds import get_ds_info,open_ds_tensorstore

voxel_size, chunk_shape, shape, roi, axes_names, filetype = get_ds_info(path)
# %%
voxel_size, chunk_shape, shape, roi, axes_names, filetype
# %%
ds = open_ds_tensorstore(path)
# %%
ds
# %%
# ds.ts_dataset.__dict__.keys()
# %%
import yaml

p = "/groups/cellmap/cellmap/zouinkhim/cellmap-flow/check2.yaml"
with open(p, "r") as f:
    config = yaml.safe_load(f)
config["models"]
# %%
from cellmap_flow.models.models_config import ScriptModelConfig
# %%
p = "/groups/cellmap/cellmap/zouinkhim/cellmap-flow/model_config.py"
# %%
sc = ScriptModelConfig(script_path=p)
# %%
sc
# %%
sc.config
# %%
sc
# %%
sc.config.axes_names
# %%
paths = ["gs://fish2-derived/em_sofima_240112_raw/",
         "/nrs/cellmap/data/jrc_mus-salivary-1/jrc_mus-salivary-1.zarr/recon-1/em/fibsem-uint8/s0",
        #  "/nrs/cellmap/data/jrc_mus-salivary-1/jrc_mus-salivary-1.zarr/recon-1/em/fibsem-uint8/"
         ]
from cellmap_flow.utils.ds import get_ds_info,open_ds_tensorstore

for path in paths:
    voxel_size, chunk_shape, shape, roi, axes_names, filetype = get_ds_info(path)
    print(f"Path: {path}")
    print(f"Voxel size: {voxel_size}")
    print(f"Chunk shape: {chunk_shape}")
    print(f"Shape: {shape}")
    print(f"ROI: {roi}")
    print(f"Axes names: {axes_names}")
    print(f"Filetype: {filetype}")
    print("-" * 40)
# %%
from cellmap_flow.utils.ds import find_target_scale
p = "/nrs/cellmap/data/jrc_mus-salivary-1/jrc_mus-salivary-1.zarr/recon-1/em/fibsem-uint8/"

find_target_scale(p, [16,16,16])
# %%
sc.config.model
# %%
input_size = (36, 286, 286)
import torch
random_input = torch.randn((1, 1) + input_size).type(torch.float32).to("cuda")
# %%
model = sc.config.model.to("cuda")
# %%
output = model(random_input)
# %%
output.shape
# %%
sc._chunk_impl(None, None, 2, 2, 2, None)

# %%
from cellmap_flow.server import CellMapFlowServer
server = CellMapFlowServer(path, sc)
# %%
server._chunk_impl(None, None, 2, 2, 2, None)
# %%
