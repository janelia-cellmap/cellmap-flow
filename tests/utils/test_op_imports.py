"""Reading a chain loads neither the dashboard nor the segmentation stack.

pipeline_spec and post.postprocessors are imported wherever a chain is read or
listed, so globals (which configures logging), Flask, neuroglancer, torch and
the libraries only some steps use stay unloaded until a step needs them.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

CHECK = """
import pickle, sys
import numpy as np

def loaded(names):
    return [m for m in names if m in sys.modules]

import cellmap_flow.pipeline_spec
light = ["cellmap_flow.globals", "flask", "neuroglancer", "torch", "huggingface_hub",
         "peft", "cellmap_flow.norm.input_normalize", "cellmap_flow.post.postprocessors"]
assert not loaded(light), loaded(light)

from cellmap_flow.post.postprocessors import (
    LabelPostprocessor, MortonSegmentationRelabeling, SimpleBlockwiseMerger,
    get_postprocessors, get_postprocessors_list,
)
get_postprocessors_list()
get_postprocessors([{"name": "SigmoidPostprocessor"}, {"name": "AffinityPostprocessor"}])
heavy = ["neuroglancer", "pymorton", "mwatershed", "fastremap", "fastmorph", "scipy.ndimage"]
assert not loaded(heavy), loaded(heavy)

# The steps that import them still run, and the merger still pickles.
data = np.zeros((1, 4, 4, 4), dtype=np.uint8)
data[0, 1:3, 1:3, 1:3] = 1
assert LabelPostprocessor()(data, chunk_corner=(0, 0, 0), chunk_num_voxels=64).max() == 1
relabeled = MortonSegmentationRelabeling()(
    data, chunk_corner=(1, 0, 0), chunk_num_voxels=np.int64(64)
)
assert relabeled.max() == 1 + 64
merger = SimpleBlockwiseMerger(face_erosion_iterations=1)
merger(data.astype(np.uint64), chunk_corner=(0, 0, 0))
merger.equivalences.union(1, 2)
assert pickle.loads(pickle.dumps(merger)).equivalences_json() == merger.equivalences_json()
"""


def test_reading_a_chain_stays_light():
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [str(REPO_ROOT), os.environ.get("PYTHONPATH", "")]
    ))
    result = subprocess.run(
        [sys.executable, "-c", CHECK], cwd=REPO_ROOT, env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
