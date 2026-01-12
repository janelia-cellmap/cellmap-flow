# %%
# path = "gs://fish2-derived/em_sofima_240112_raw/"
path = "/groups/cellmap/cellmap/zouinkhim/cellmap-flow/example_out.zarr/s0"
# %%
from cellmap_flow.models.models_config import ScriptModelConfig

# %%
p = "/groups/cellmap/cellmap/zouinkhim/cellmap-flow/model_config.py"
# %%
sc = ScriptModelConfig(script_path=p)
vars(sc)
# %%
from cellmap_flow.server import CellMapFlowServer

server = CellMapFlowServer(path, sc)
vars(server)
# %%
# from cellmap_flow.globals import g
# g
# from cellmap_flow.norm.input_normalize import MinMaxNormalizer
# from cellmap_flow.post.postprocessors import DefaultPostprocessor
# input_norms = [MinMaxNormalizer()]
# postprocess = [DefaultPostprocessor(0,200,0,1)]
# %%
# c,z,y,x

# x = server.chunk(None, None, 0, 6, 35, 34)

x = server._chunk_impl(None, None, 0, 6, 35, 34)
# s0/0/6/35/34


# x = server._chunk_impl(None, None, 2, 2, 2, None)
# x.shape
# %%


from cellmap_flow.image_data_interface import ImageDataInterface

p = "/groups/cellmap/cellmap/zouinkhim/cellmap-flow/example.zarr/s0"
inf_custom = ImageDataInterface(p)
# %%
inf_custom.info
# %%
