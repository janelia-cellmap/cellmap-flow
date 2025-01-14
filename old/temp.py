# %%
from funlib.persistence import open_ds

ds = open_ds(
    "/nrs/cellmap/data/jrc_c-elegans-op50-1/jrc_celegans-op50-1_normalized.zarr",
    "recon-1/em/fibsem-uint8/s2",
)
ds.roi.shape

# %%
from dacapo.store.create_store import create_config_store

store = create_config_store()
store.retrieve_run_config_names()
# %%
import neuroglancer
import pandas as pd
import neuroglancer
from funlib.persistence import open_ds
import numpy as np


neuroglancer.set_server_bind_address("0.0.0.0")
viewer = neuroglancer.Viewer()

with viewer.txn() as s:
    layer = neuroglancer.ImageLayer(
        source="zarr://https://cellmap-vm1.int.janelia.org/nrs/data/jrc_mus-liver-zon-1/jrc_mus-liver-zon-1.zarr/recon-1/em/fibsem-uint8",
        shader="""#uicontrol invlerp normalized
#uicontrol float prediction_threshold slider(min=-1, max=1, step=0.1)

void main() {
emitGrayscale(normalized());
}""",
        shader_controls={"prediction_threshold": 0},
    )
    s.layers["temp"] = layer
    print(s.layers["temp"].shaderControls.get("prediction_threshold"))
    # print(layer.equivalences)
    # s.layers.append(name=f"temp", layer=layer)
print(viewer)

# with viewer.txn() as s:
#     s.layers.__delitem__("temp")
# %%
with viewer.txn() as s:
    print(s.layers["temp"].shaderControls.get("prediction_threshold", 0))
# %%

all_eqs = neuroglancer.equivalence_map.EquivalenceMap()
print("Creating equivalence map for agglomeration")
all_eqs.union(1, 2)
all_eqs.union(2, 3)
all_eqs.to_json()
# %%
# create random int 3d cube
import numpy as np

cube = np.random.randint(0, 5, (36, 36, 36))
mask = np.zeros_like(cube, dtype=bool)
mask[1:-1, 1:-1, 1:-1] = True
cube_ma = np.ma.masked_array(cube, mask)
x, y, z = np.ma.where(cube_ma > 0)
vals = cube_ma[x, y, z]
# %%
from scipy.spatial.distance import pdist

pdist(list(zip(x, y, z)))
# %%
from scipy import spatial
import networkx as nx

G = nx.Graph()
tree = spatial.cKDTree(list(zip(x, y, z)))

# try within 1 um radius
neighbors = tree.query_ball_tree(tree, 1)
for i in range(len(neighbors)):
    for j in neighbors[i]:
        print(j)
        G.add_edge(vals[i], vals[j])
print(G)

# %%
import neuroglancer


def in_function():
    neuroglancer.set_server_bind_address("0.0.0.0")
    viewer = neuroglancer.Viewer()
    with viewer.txn() as s:
        s.layers["raw"] = neuroglancer.ImageLayer(
            source=f"zarr://https://cellmap-vm1.int.janelia.org/nrs/data/jrc_mus-liver-zon-1/jrc_mus-liver-zon-1.zarr/recon-1/em/fibsem-uint8",
        )
        s.layers[f"inference and postprocessing"] = neuroglancer.SegmentationLayer(
            source="n5://https://ackermand-ws2:8000", equivalences=[]
        )
        s.cross_section_scale = 10e-9
        s.projection_scale = 1000e-9
    state = viewer.state.to_json()
    for layer in state["layers"]:
        if layer["type"] == "segmentation":
            layer["equivalences"] = [[1, 2, 3]]
    print(viewer)
    viewer.set_state(state)
    print(viewer)


in_function()

# %%
from funlib.persistence import open_ds
from funlib.geometry import Coordinate, Roi

ZARR_PATH = "/nrs/cellmap/data/jrc_c-elegans-bw-1/jrc_c-elegans-bw-1_normalized.zarr"
DATASET = "recon-1/em/fibsem-uint8"
# load raw data
DS = open_ds(
    ZARR_PATH,
    f"{DATASET}/s0",
)
DS.roi.shape
data = DS.to_ndarray(Roi(Coordinate(5148, 3361, 5292) * 4, (16, 16, 16)))
print(data)
# %%
data[0, :, 0]
# %%
import zarr
import numpy as np

zarr.zeros([10000, 10000, 10000], chunks=(64, 64, 64), dtype=np.uint8)
import fsspec

store = fsspec.get_mapper("http://ackermand-ws2:8000/test.n5/test")
zarr.open(store, "r")
# %%
from funlib.persistence import open_ds

ds = open_ds("https://ackermand-ws2", "recon-1/em/fibsem-uint8/s0")
ds.data

# %%
import tensorstore as ts


def open_tensorstore():
    dataset_future = ts.open(
        {
            "driver": "n5",
            "kvstore": {
                "driver": "http",
                "base_url": "http://ackermand-ws2:8000/test.n5/test/s0",
            },
            "context": {"cache_pool": {"total_bytes_limit": 100_000_000}},
            "recheck_cached_data": "open",
        }
    )

    # Now you can use the dataset
    return dataset_future.result()


import neuroglancer

# %%
neuroglancer.set_server_bind_address("0.0.0.0")
viewer = neuroglancer.Viewer()
dataset = open_tensorstore()
with viewer.txn() as s:
    s.layers[f"inference and postprocessing"] = neuroglancer.SegmentationLayer(
        source=neuroglancer.LocalVolume(data=dataset)
    )
    s.cross_section_scale = 1e-9
    s.projection_scale = 500e-9
    s.layers.pop(f"inference and postprocessing")
print(viewer)
# %%
import neuroglancer

neuroglancer.url_state(
    "http://ackermand-ws2.hhmi.org:39173/v/89e6960b86582d0194f1dba1a22c09f86caf355b/"
)
# %%
import neuroglancer
import tensorstore as ts


def open_tensorstore():
    dataset_future = ts.open(
        {
            "driver": "n5",
            "kvstore": {
                "driver": "http",
                "base_url": "http://ackermand-ws2:8000/test.n5/inference_and_postprocessing_0/s0",
            },
            "context": {"cache_pool": {"total_bytes_limit": 100_000_000}},
            "recheck_cached_data": "open",
        }
    )

    # Now you can use the dataset
    return dataset_future.result()


TENSORSTORE = open_tensorstore()
LOCAL_VOLUME = neuroglancer.LocalVolume(
    data=TENSORSTORE,
    dimensions=neuroglancer.CoordinateSpace(
        names=["x", "y", "z", "c"],
        units=["nm", "nm", "nm", "nm"],
        scales=[16, 16, 16, 16],
        coordinate_arrays=[
            None,
            None,
            None,
            None,
        ],
    ),
)
LOCAL_VOLUME.data_type
# %%
import numpy as np

max_d = -1
for dx in range(-3, 4):
    for dy in range(-3, 4):
        for dz in range(-3, 4):
            d = np.linalg.norm([dx, dy, dz]) * 2
            if d < 6:
                if d>max_d:
                    max_d = d 
                    print(f"{max_d=}")
                print(np.linalg.norm([dx, dy, dz]) * 2)
# %%
