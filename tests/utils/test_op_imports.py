"""Reading a chain must not load the segmentation stack.

post.postprocessors is imported wherever a chain is built or described
(the dashboard pages, every serializer, the finetune manifest), so the
libraries only some steps use are imported by those steps.
"""

import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

HEAVY = [
    "neuroglancer",
    "pymorton",
    "mwatershed",
    "fastremap",
    "fastmorph",
    "scipy.ndimage",
]


def _loaded_after(statement):
    code = (
        f"import sys\n{statement}\n"
        f"print('LOADED=' + ','.join(m for m in {HEAVY!r} if m in sys.modules))\n"
    )
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [str(REPO_ROOT), os.environ.get("PYTHONPATH", "")]
    ))
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, env=env,
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    (line,) = [line for line in result.stdout.splitlines() if line.startswith("LOADED=")]
    return [m for m in line[len("LOADED="):].split(",") if m]


def test_importing_the_postprocessors_loads_none_of_them():
    assert _loaded_after("import cellmap_flow.post.postprocessors") == []


def test_listing_and_building_light_steps_loads_none_of_them():
    statement = (
        "from cellmap_flow.post.postprocessors import get_postprocessors, "
        "get_postprocessors_list\n"
        "get_postprocessors_list()\n"
        "get_postprocessors([{'name': 'SigmoidPostprocessor'}, "
        "{'name': 'AffinityPostprocessor', 'bias': 0.5}])"
    )
    assert _loaded_after(statement) == []


def test_the_steps_that_need_them_still_work():
    """The imports moved; the results must not."""
    from cellmap_flow.post.postprocessors import (
        LabelPostprocessor,
        MortonSegmentationRelabeling,
        SimpleBlockwiseMerger,
    )

    data = np.zeros((1, 4, 4, 4), dtype=np.uint8)
    data[0, 1:3, 1:3, 1:3] = 1
    labels = LabelPostprocessor()(data, chunk_corner=(0, 0, 0), chunk_num_voxels=64)
    assert labels.max() == 1

    relabeled = MortonSegmentationRelabeling()(
        data, chunk_corner=(1, 0, 0), chunk_num_voxels=np.int64(64)
    )
    assert relabeled.max() == 1 + 64  # interleave(1, 0, 0) == 1

    merger = SimpleBlockwiseMerger(face_erosion_iterations=1)
    merger(data.astype(np.uint64), chunk_corner=(0, 0, 0))


def test_the_merger_still_pickles():
    from cellmap_flow.post.postprocessors import SimpleBlockwiseMerger

    merger = SimpleBlockwiseMerger()
    merger.equivalences.union(1, 2)
    restored = pickle.loads(pickle.dumps(merger))
    assert restored.equivalences_json() == merger.equivalences_json()
    assert restored.to_dict() == merger.to_dict()
