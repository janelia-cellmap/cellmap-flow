#%%
from cellmap_flow.server import CellMapFlowServer
from cellmap_flow.utils.data import ModelConfig, BioModelConfig, DaCapoModelConfig, ScriptModelConfig
#%%
# dataset = "/nrs/cellmap/data/jrc_mus-cerebellum-1/jrc_mus-cerebellum-1.zarr/recon-1/em/fibsem-uint8/s0"
# script_path = "/groups/cellmap/cellmap/zouinkhim/cellmap-flow/example/model_setup04.py"
# script_path = "/groups/cellmap/cellmap/zouinkhim/cellmap-flow/example/dacapo_run_retrieve.py"

script_path = "/groups/cellmap/cellmap/zouinkhim/cellmap-flow/example/model_spec.py"
dataset = "/groups/cellmap/cellmap/ackermand/for_hideo/jrc_pri_neuron_0710Dish4/jrc_pri_neuron_0710Dish4.n5/em/fibsem-uint8/s0"

model_config = ScriptModelConfig(script_path=script_path)
server = CellMapFlowServer(dataset, model_config)
# %%
chunk_x = 2
chunk_y = 2
chunk_z = 2

server.chunk(None, None, chunk_x, chunk_y, chunk_z, None)
# %%
