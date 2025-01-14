# %%
import argparse
from http import HTTPStatus
from flask import Flask, jsonify
from flask_cors import CORS

import numpy as np
import numcodecs
from scipy import spatial
from zarr.n5 import N5ChunkWrapper
from funlib.persistence import open_ds
from funlib.geometry import Roi
import numpy as np
from funlib.geometry import Coordinate
from skimage.morphology import erosion
from scipy.ndimage import binary_dilation


# NOTE: Normally we would just load in run but here we have to recreate it to save time since our run has so many points
import neuroglancer
import socket
import skimage
from image_data_interface import ImageDataInterface

app = Flask(__name__)
CORS(app)

# This demo produces an RGB volume for aesthetic purposes.
# Note that this is 3 (virtual) teravoxels per channel.
SEGMENTATION = True
NUM_OUT_CHANNELS = 9
if SEGMENTATION:
    NUM_OUT_CHANNELS = 1
BLOCK_SHAPE = np.array([16, 16, 16])
MAX_SCALE = 0

CHUNK_ENCODER = N5ChunkWrapper(np.uint8, BLOCK_SHAPE, compressor=numcodecs.GZip())
EQUIVALENCES = neuroglancer.equivalence_map.EquivalenceMap()

MODEL = None
DS = None
EROSION_ITERATIONS = 1
DILATION_ITERATIONS = 1
neuroglancer.set_server_bind_address("0.0.0.0")
VIEWER = neuroglancer.Viewer(token="canoliculi")
ip_address = socket.getfqdn()
ZARR_PATH = "/nrs/cellmap/data/jrc_mus-liver-zon-2/jrc_mus-liver-zon-2.zarr"
DATASET = "recon-1/em/fibsem-uint8"

with VIEWER.txn() as s:
    s.layers["raw"] = neuroglancer.ImageLayer(
        source=f'zarr://https://cellmap-vm1.int.janelia.org/{ZARR_PATH.replace("/nrs/cellmap", "/nrs")}/{DATASET}',
        shader="""#uicontrol invlerp normalized
#uicontrol float dilation_iterations slider(min=0, max=5, step=1, default=1)
#uicontrol float erosion_iterations slider(min=0, max=5, step=1, default=1)

void main() {
emitGrayscale(normalized());
}""",
        shaderControls={
            "dilation_iterations": DILATION_ITERATIONS,
            "erosion_iterations": EROSION_ITERATIONS,
        },
    )
    s.layers[
        f"inference_and_postprocessing_{DILATION_ITERATIONS}_{EROSION_ITERATIONS}"
    ] = neuroglancer.SegmentationLayer(
        source=f"n5://http://{ip_address}:8000/test.n5/inference_and_postprocessing_{DILATION_ITERATIONS}_{EROSION_ITERATIONS}",
        equivalences=EQUIVALENCES.to_json(),
    )
    s.cross_section_scale = 1e-9
    s.projection_scale = 500e-9

PREVIOUS_UPDATE_TIME = 0
import time


def update_state():
    global PREVIOUS_UPDATE_TIME, DILATION_ITERATIONS, EROSION_ITERATIONS, EQUIVALENCES, EDGE_VOXEL_POSITION_TO_VAL_DICT
    current_DILATION_ITERATIONS = VIEWER.state.layers["raw"].shaderControls.get(
        "dilation_iterations", 1
    )
    current_erosion_iterations = VIEWER.state.layers["raw"].shaderControls.get(
        "erosion_iterations", 1
    )
    # print(f"{current_DILATION_ITERATIONS=}")
    if (
        current_DILATION_ITERATIONS != DILATION_ITERATIONS
        or current_erosion_iterations != EROSION_ITERATIONS
    ):
        with VIEWER.txn() as s:
            s.layers.__delitem__(
                f"inference_and_postprocessing_{DILATION_ITERATIONS}_{EROSION_ITERATIONS}"
            )
            DILATION_ITERATIONS = current_DILATION_ITERATIONS
            EROSION_ITERATIONS = current_erosion_iterations
            EDGE_VOXEL_POSITION_TO_VAL_DICT = {}
            EQUIVALENCES = neuroglancer.equivalence_map.EquivalenceMap()
            s.layers[
                f"inference_and_postprocessing_{DILATION_ITERATIONS}_{EROSION_ITERATIONS}"
            ] = neuroglancer.SegmentationLayer(
                source=f"n5://http://{ip_address}:8000/test.n5/inference_and_postprocessing_{DILATION_ITERATIONS}_{EROSION_ITERATIONS}",
            )

    # with VIEWER.txn() as s:
    #     # print(f"{EQUIVALENCES.to_json()=}")
    #     s.layers[
    #         f"inference_and_postprocessing_{DILATION_ITERATIONS}_{EROSION_ITERATIONS}"
    #     ].equivalences = EQUIVALENCES.to_json()
    # LOCAL_VOLUME.invalidate()
    PREVIOUS_UPDATE_TIME = time.time()


# if __name__ == "__main__":
print(VIEWER)
while True:
    update_state()
    time.sleep(1)

# %%
