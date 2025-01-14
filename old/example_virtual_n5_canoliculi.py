"""
# Example Virtual N5

Example service showing how to host a virtual N5,
suitable for browsing in neuroglancer.

Neuroglancer is capable of browsing N5 files, as long as you store them on
disk and then host those files over http (with a CORS-friendly http server).
But what if your data doesn't exist on disk yet?

This server hosts a "virtual" N5.  Nothing is stored on disk,
but neuroglancer doesn't need to know that.  This server provides the
necessary attributes.json files and chunk files on-demand, in the
"locations" (url patterns) that neuroglancer expects.

For simplicity, this file uses Flask. In a production system,
you'd probably want to use something snazzier, like FastAPI.

To run the example, install a few dependencies:

    conda create -n example-virtual-n5 -c conda-forge zarr flask flask-cors
    conda activate example-virtual-n5

Then just execute the file:

    python example_virtual_n5.py

Or, for better performance, use a proper http server:

    conda install -c conda-forge gunicorn
    gunicorn --bind 0.0.0.0:8000 --workers 8 --threads 1 example_virtual_n5:app

You can browse the data in neuroglancer after configuring the viewer with the appropriate layer [settings][1].

[1]: http://neuroglancer-demo.appspot.com/#!%7B%22dimensions%22:%7B%22x%22:%5B1e-9%2C%22m%22%5D%2C%22y%22:%5B1e-9%2C%22m%22%5D%2C%22z%22:%5B1e-9%2C%22m%22%5D%7D%2C%22position%22:%5B5000.5%2C7500.5%2C10000.5%5D%2C%22crossSectionScale%22:25%2C%22projectionScale%22:32767.999999999996%2C%22layers%22:%5B%7B%22type%22:%22image%22%2C%22source%22:%7B%22url%22:%22n5://http://127.0.0.1:8000%22%2C%22transform%22:%7B%22outputDimensions%22:%7B%22x%22:%5B1e-9%2C%22m%22%5D%2C%22y%22:%5B1e-9%2C%22m%22%5D%2C%22z%22:%5B1e-9%2C%22m%22%5D%2C%22c%5E%22:%5B1%2C%22%22%5D%7D%7D%7D%2C%22tab%22:%22rendering%22%2C%22opacity%22:0.42%2C%22shader%22:%22void%20main%28%29%20%7B%5Cn%20%20emitRGB%28%5Cn%20%20%20%20vec3%28%5Cn%20%20%20%20%20%20getDataValue%280%29%2C%5Cn%20%20%20%20%20%20getDataValue%281%29%2C%5Cn%20%20%20%20%20%20getDataValue%282%29%5Cn%20%20%20%20%29%5Cn%20%20%29%3B%5Cn%7D%5Cn%22%2C%22channelDimensions%22:%7B%22c%5E%22:%5B1%2C%22%22%5D%7D%2C%22name%22:%22colorful-data%22%7D%5D%2C%22layout%22:%224panel%22%7D
"""

# %%
# NOTE: To generate host key and host cert do the following: https://serverfault.com/questions/224122/what-is-crt-and-key-files-and-how-to-generate-them
# openssl genrsa 2048 > host.key
# chmod 400 host.key
# openssl req -new -x509 -nodes -sha256 -days 365 -key host.key -out host.cert
# Then can run like this:
# gunicorn --certfile=host.cert --keyfile=host.key --bind 0.0.0.0:8000 --workers 1 --threads 1 example_virtual_n5:app
# NOTE: You will probably have to access the host:8000 separately and say it is safe to go there

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-p", "--port", default=8000)
    args = parser.parse_args()
    app.run(
        host="0.0.0.0",
        port=args.port,
        debug=args.debug,
        threaded=not args.debug,
        use_reloader=args.debug,
    )


# global DS, CONFIG_STORE, WEIGHTS_STORE, MODEL
ZARR_PATH = "/nrs/cellmap/data/jrc_mus-liver-zon-2/jrc_mus-liver-zon-2.zarr"
DATASET = "recon-1/em/fibsem-uint8"

# load raw data
SCALE_LEVEL = 2
IDI_RAW = ImageDataInterface(f"{ZARR_PATH}/{DATASET}/s{SCALE_LEVEL}")
IDI_CELL = ImageDataInterface(
    f"/nrs/cellmap/data/jrc_mus-liver-zon-2/jrc_mus-liver-zon-2.zarr/recon-1/labels/inference/segmentations/cell/s0",
)
IDI_ECS = ImageDataInterface(
    f"/nrs/cellmap/ackermand/nearline/zouinkhim/jrc_mus-liver-zon-2_postprocessed.zarr/20240503_ecs/ecs_step_1_full",
    output_voxel_size=Coordinate(3 * [8 * 2**SCALE_LEVEL]),
)
# %%
VOL_SHAPE_ZYX = np.array(IDI_RAW.ds.shape)
VOL_SHAPE = np.array([*VOL_SHAPE_ZYX[::-1]])
VOL_SHAPE_ZYX_IN_BLOCKS = np.ceil(VOL_SHAPE_ZYX / BLOCK_SHAPE[:3]).astype(int)
THRESHOLD = 170
EROSION_ITERATIONS = 1

VOXEL_SIZE = IDI_RAW.ds.voxel_size
OUTPUT_VOXEL_SIZE = [8 * 2**SCALE_LEVEL, 8 * 2**SCALE_LEVEL, 8 * 2**SCALE_LEVEL]

# %%
neuroglancer.set_server_bind_address("0.0.0.0")
VIEWER = neuroglancer.Viewer()
ip_address = socket.getfqdn()

with VIEWER.txn() as s:
    s.layers["raw"] = neuroglancer.ImageLayer(
        source=f'zarr://https://cellmap-vm1.int.janelia.org/{ZARR_PATH.replace("/nrs/cellmap", "/nrs")}/{DATASET}',
        shader="""#uicontrol invlerp normalized
#uicontrol float threshold slider(min=0, max=255, step=1, default=170)
#uicontrol float erosion_iterations slider(min=0, max=5, step=1, default=1)

void main() {
emitGrayscale(normalized());
}""",
        shaderControls={
            "threshold": THRESHOLD,
            "erosion_iterations": EROSION_ITERATIONS,
        },
    )
    s.layers[f"inference_and_postprocessing_{THRESHOLD}_{EROSION_ITERATIONS}"] = (
        neuroglancer.SegmentationLayer(
            source=f"n5://http://{ip_address}:8000/test.n5/inference_and_postprocessing_{THRESHOLD}_{EROSION_ITERATIONS}",
            equivalences=EQUIVALENCES.to_json(),
        )
    )
    s.cross_section_scale = 1e-9
    s.projection_scale = 500e-9
print(VIEWER)

PREVIOUS_UPDATE_TIME = 0
import time

EDGE_VOXEL_POSITION_TO_VAL_DICT = {}
EQUIVALENCES = neuroglancer.equivalence_map.EquivalenceMap()


def update_state():
    global PREVIOUS_UPDATE_TIME, THRESHOLD, EROSION_ITERATIONS, EQUIVALENCES, EDGE_VOXEL_POSITION_TO_VAL_DICT
    current_threshold = VIEWER.state.layers["raw"].shaderControls.get("threshold", 170)
    current_erosion_iterations = VIEWER.state.layers["raw"].shaderControls.get(
        "erosion_iterations", 1
    )
    print(f"{current_threshold=}")
    if (
        current_threshold != THRESHOLD
        or current_erosion_iterations != EROSION_ITERATIONS
    ):
        with VIEWER.txn() as s:
            s.layers.__delitem__(
                f"inference_and_postprocessing_{THRESHOLD}_{EROSION_ITERATIONS}"
            )
            THRESHOLD = current_threshold
            EROSION_ITERATIONS = current_erosion_iterations
            EDGE_VOXEL_POSITION_TO_VAL_DICT = {}
            EQUIVALENCES = neuroglancer.equivalence_map.EquivalenceMap()
            s.layers[
                f"inference_and_postprocessing_{THRESHOLD}_{EROSION_ITERATIONS}"
            ] = neuroglancer.SegmentationLayer(
                source=f"n5://http://{ip_address}:8000/test.n5/inference_and_postprocessing_{THRESHOLD}_{EROSION_ITERATIONS}",
            )

    with VIEWER.txn() as s:
        # print(f"{EQUIVALENCES.to_json()=}")
        s.layers[
            f"inference_and_postprocessing_{THRESHOLD}_{EROSION_ITERATIONS}"
        ].equivalences = EQUIVALENCES.to_json()
        # LOCAL_VOLUME.invalidate()
    PREVIOUS_UPDATE_TIME = time.time()


# %%
@app.route("/test.n5/<string:dataset>/attributes.json")
def top_level_attributes(dataset):
    scales = [[2**s, 2**s, 2**s] for s in range(MAX_SCALE + 1)]
    print(scales)
    attr = {
        "pixelResolution": {"dimensions": [*OUTPUT_VOXEL_SIZE], "unit": "nm"},
        "ordering": "C",
        "scales": scales,
        "axes": ["x", "y", "z"],
        "units": ["nm", "nm", "nm"],
        "translate": [0, 0, 0],
    }
    return jsonify(attr), HTTPStatus.OK


@app.route("/test.n5/<string:dataset>/s<int:scale>/attributes.json")
def attributes(dataset, scale):
    attr = {
        "transform": {
            "ordering": "C",
            "axes": ["x", "y", "z"],
            "scale": [
                *OUTPUT_VOXEL_SIZE,
            ],
            "units": ["nm", "nm", "nm"],
            "translate": [0.0, 0.0, 0.0],
        },
        "compression": {"type": "gzip", "useZlib": False, "level": -1},
        "blockSize": BLOCK_SHAPE.tolist(),
        "dataType": "uint8",
        "dimensions": (VOL_SHAPE[:3] // 2**scale).tolist(),
    }
    return jsonify(attr), HTTPStatus.OK


@app.route(
    "/test.n5/<string:dataset>/s<int:scale>/<int:chunk_x>/<int:chunk_y>/<int:chunk_z>"
)
def chunk(dataset, scale, chunk_x, chunk_y, chunk_z):
    """
    Serve up a single chunk at the requested scale and location.

    This 'virtual N5' will just display a color gradient,
    fading from black at (0,0,0) to white at (max,max,max).
    """

    # assert chunk_c == 0, "neuroglancer requires that all blocks include all channels"
    if dataset == f"inference_and_postprocessing_{THRESHOLD}_{EROSION_ITERATIONS}":
        corner = BLOCK_SHAPE[:3] * np.array([chunk_z, chunk_y, chunk_x])
        box = np.array([corner, BLOCK_SHAPE[:3]]) * OUTPUT_VOXEL_SIZE
        global_id_offset = np.prod(BLOCK_SHAPE[:3]) * (
            VOL_SHAPE_ZYX_IN_BLOCKS[0] * VOL_SHAPE_ZYX_IN_BLOCKS[1] * chunk_x
            + VOL_SHAPE_ZYX_IN_BLOCKS[0] * chunk_y
            + chunk_z
        )
        block_vol = postprocess_for_chunk(
            inference_for_chunk(box, global_id_offset), corner
        )
        return (
            # Encode to N5 chunk format (header + compressed data)
            CHUNK_ENCODER.encode(block_vol),
            HTTPStatus.OK,
            {"Content-Type": "application/octet-stream"},
        )
    else:
        return jsonify(dataset), HTTPStatus.OK


# %%


def erode_image(image):
    # Structuring element (3x3 square)
    selem = np.ones((3, 3, 3), dtype=bool)

    # Erode each region
    eroded_image = np.zeros_like(image)
    for id in np.unique(image):
        if id == 0:  # Skip background
            continue
        mask = image == id
        eroded_mask = erosion(mask, selem)
        eroded_image[eroded_mask] = id

    return eroded_image


def inference_for_chunk(box, global_id_offset):
    global THRESHOLD, EROSION_ITERATIONS
    print(THRESHOLD)
    # Compute the portion of the box that is actually populated.
    # It will differ from [(0,0,0), BLOCK_SHAPE] at higher scales,
    # where the chunk may extend beyond the bounding box of the entire volume.
    box = box.copy()
    # box[1] = np.minimum(box[0] + box[1], VOL_SHAPE[:3] // 2**scale)
    roi = Roi(box[0], box[1])

    cell = IDI_CELL.to_ndarray_ts(
        roi.grow(256 * EROSION_ITERATIONS, 256 * EROSION_ITERATIONS)
    )
    for _ in range(EROSION_ITERATIONS):
        cell = erode_image(cell)
    not_cell = (
        cell[
            EROSION_ITERATIONS:-EROSION_ITERATIONS,
            EROSION_ITERATIONS:-EROSION_ITERATIONS,
            EROSION_ITERATIONS:-EROSION_ITERATIONS,
        ]
        == 0
    )
    not_cell = not_cell.repeat(8, 0).repeat(8, 1).repeat(8, 2)

    not_ecs = IDI_ECS.to_ndarray_ts(roi) == 0

    raw = IDI_RAW.to_ndarray_ts(roi) > THRESHOLD

    raw = raw * not_cell * not_ecs
    data = raw.astype(np.uint64)  # skimage.measure.label(raw).astype(np.uint64)
    data[data > 0] += global_id_offset
    return data


# %%
import numpy_indexed as npi
import mwatershed as mws
from scipy.ndimage import measurements


def postprocess_for_chunk(chunk, corner):
    segmentation = chunk
    # get touching voxels
    mask = np.zeros_like(segmentation, dtype=bool)
    mask[1:-1, 1:-1, 1:-1] = True
    segmentation_ma = np.ma.masked_array(segmentation, mask)
    z, y, x = np.ma.where(segmentation_ma > 0)
    values = segmentation_ma[z, y, x]
    # EDGE_VOXEL_POSITION_TO_VAL_DICT.update(
    #     dict(
    #         zip(
    #             zip(
    #                 z + corner[0],
    #                 y + corner[1],
    #                 x + corner[2],
    #             ),
    #             values,
    #         )
    #     )
    # )
    segmentation = segmentation.astype(np.uint8) + 1
    update_equivalences()
    return segmentation


# %%
def update_equivalences():
    global PREVIOUS_UPDATE_TIME, EDGE_VOXEL_POSITION_TO_VAL_DICT, EQUIVALENCES
    if time.time() - PREVIOUS_UPDATE_TIME > 5:
        # print("Updating equivalences")
        positions = list(EDGE_VOXEL_POSITION_TO_VAL_DICT.keys())
        ids = list(EDGE_VOXEL_POSITION_TO_VAL_DICT.values())
        if ids:
            tree = spatial.cKDTree(positions)
            neighbors = tree.query_ball_tree(tree, 1)  # distance of 1 voxel
            for i in range(len(neighbors)):
                for j in neighbors[i]:
                    EQUIVALENCES.union(ids[i], ids[j])
        update_state()
        PREVIOUS_UPDATE_TIME = time.time()
        # print("Updated equivalences")


# %%
if __name__ == "__main__":
    main()

# %%
