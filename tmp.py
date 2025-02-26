#%%
from cellmap_flow.utils.data import CellMapModelConfig
from cellmap_flow.cli.server_cli import run_server

folder_path = "/groups/cellmap/cellmap/zouinkhim/models/v22_peroxisome_funetuning_best_v20_1e4_finetuned_distances_8nm_peroxisome_jrc_mus-livers_peroxisome_8nm_attention-upsample-unet_default_one_label_finetuning_0"
name = "hello"
data_path = "/nrs/cellmap/data/jrc_mus-pancreas-5/jrc_mus-pancreas-5.zarr/recon-1/em/fibsem-uint16/s0"
port = 0
certfile = None
keyfile = None
debug = True
model_config = CellMapModelConfig(folder_path=folder_path, name=name)
# run_server(model_config, data_path, debug, port, certfile, keyfile)
# %%
from cellmap_flow.server import CellMapFlowServer
server = CellMapFlowServer(data_path, model_config)
chunk_x = 2
chunk_y = 2
chunk_z = 2

server._chunk_impl(None, None, chunk_x, chunk_y, chunk_z, None)

print("Server check passed")
# %%
