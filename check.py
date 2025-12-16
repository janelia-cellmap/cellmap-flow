# %%
path = "gs://fish2-derived/em_sofima_240112_raw/"
from cellmap_flow.utils.ds import get_ds_info,open_ds_tensorstore

voxel_size, chunk_shape, shape, roi, swap_axes = get_ds_info(path)
# %%
voxel_size, chunk_shape, shape, roi, swap_axes
# %%
ds = open_ds_tensorstore(path)
# %%
ds
# %%
ds.ts_dataset.__dict__.keys()
# %%
import yaml

p = "/groups/cellmap/cellmap/zouinkhim/cellmap-flow/check2.yaml"
with open(p, "r") as f:
    config = yaml.safe_load(f)
config["models"]
# %%
