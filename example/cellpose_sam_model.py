"""Cellpose-SAM (Cellpose 4's ``cpsam``) as a script model, run slice by slice.

Cellpose 4 cannot share cellmap-flow's default environment, whose cellpose 3
pins an older numpy, so this script runs in the ``cellpose4`` pixi
environment: give its model entry ``env: cellpose4`` (see
``cellpose_sam.yaml``, and "Running a model in its own environment" in the
docs). ``example/cellpose_model.py`` is the cellpose 3 model, which runs in
the default environment.

Each z slice of a chunk is segmented in 2D, with ``CONTEXT`` voxels of
margin around it in y and x, which are cut off again. What the layer shows
is ``OUTPUT``:

- ``"probability"``: Cellpose's cell probability, from 0 to 1 (float32).
  It is computed per voxel, so it joins up across chunks, and the mask
  dynamics are skipped, which makes it the faster of the two.
- ``"masks"``: Cellpose's instance masks (uint64), ids unique within a
  chunk. Masks are made per chunk: an object that crosses a chunk's edge is
  cut there, with another id on each side, and objects are not joined from
  slice to slice. Serve them with the ``MortonSegmentationRelabeling``
  postprocessor so that ids are unique across chunks too; for whole-volume
  masks, run the probability through ``cellmap_flow blockwise`` and segment
  that.

Licence: Cellpose-SAM's weights were trained on data that includes
datasets licensed CC-BY-NC, so they are for non-commercial use.

The weights (about 1 GB) are downloaded to ``~/.cellpose/models`` the first
time a server starts.
"""

import numpy as np
from cellpose import models
from funlib.geometry import Coordinate

from cellmap_flow.image_data_interface import ImageDataInterface

# "probability" or "masks"; see above.
OUTPUT = "probability"

# The voxel size the data is read at; Cellpose-SAM is trained on objects
# about 30 pixels across, so pick the scale at which yours are roughly that.
input_voxel_size = Coordinate((64, 64, 64))
output_voxel_size = input_voxel_size

# Each chunk: SLICES slices of SIZE x SIZE voxels, read with CONTEXT more on
# each side in y and x so that objects at the chunk's edge are seen whole
# by Cellpose. None in z: each slice is segmented on its own.
SLICES, SIZE, CONTEXT = 8, 512, 32
write_shape = Coordinate((SLICES, SIZE, SIZE)) * output_voxel_size
read_shape = Coordinate((SLICES, SIZE + 2 * CONTEXT, SIZE + 2 * CONTEXT)) * input_voxel_size
context = (read_shape - write_shape) / 2

output_channels = 1
block_shape = np.array((SLICES, SIZE, SIZE, output_channels))
output_dtype = np.uint64 if OUTPUT == "masks" else np.float32
channels = ["cell"]

# Cellpose's own evaluation settings; diameter None keeps Cellpose-SAM's
# own scale (objects about 30 pixels across).
eval_kwargs = {
    "diameter": None,
    "flow_threshold": 0.4,
    "cellprob_threshold": 0.0,
    "batch_size": 8,
    "compute_masks": OUTPUT == "masks",
}

model = models.CellposeModel(gpu=True, pretrained_model="cpsam")


def process_chunk(idi: ImageDataInterface, output_roi):
    data = idi.to_ndarray_ts(output_roi.grow(context, context))
    # A list of 2D images: Cellpose segments each on its own. (A 3D array
    # without do_3D is taken to be one 2D image with channels.)
    masks, flows, _ = model.eval(list(data), **eval_kwargs)
    inner = (slice(CONTEXT, CONTEXT + SIZE),) * 2

    if OUTPUT == "probability":
        # flows[i][2] is the cell probability as a logit.
        logits = np.stack([flow[2][inner] for flow in flows]).astype(np.float32)
        return (1.0 / (1.0 + np.exp(-logits)))[np.newaxis]

    output = np.zeros((len(masks), SIZE, SIZE), dtype=np.uint64)
    next_id = 0
    for z, mask in enumerate(masks):
        mask = mask[inner].astype(np.uint64)
        # Each slice numbers its objects from 1; shift them past the slices
        # before, so that an id means one object in the whole chunk.
        output[z] = np.where(mask > 0, mask + next_id, 0)
        next_id = max(next_id, int(output[z].max()))
    return output[np.newaxis]
