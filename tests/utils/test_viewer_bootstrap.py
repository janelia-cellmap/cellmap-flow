"""A new viewer's coordinate space (viewer.bootstrap).

The raw data decides it, from the finest level of its pyramid, before any
layer is added; otherwise neuroglancer takes it from whichever layer it
likes, such as an extra layer at another voxel size. What each caller's
viewer holds is pinned in test_layer_sources_snapshot.
"""

from types import SimpleNamespace

import neuroglancer
import numpy as np
import pytest
from neuroglancer.viewer_base import ViewerBase

from cellmap_flow.viewer import bootstrap


@pytest.mark.parametrize("dataset", [
    pytest.param(lambda f: f.ome_pyramid(((8, 0), (16, 4))), id="pyramid"),
    pytest.param(lambda f: f.ome_pyramid(((8, 0), (16, 4))) + "/s1", id="its-coarser-level"),
    pytest.param(lambda f: f.raw_zarr(np.zeros((16, 16, 16), np.uint8)), id="plain-array"),  # flat: no contrast range
])
def test_a_viewer_takes_its_dimensions_from_the_finest_raw_level(monkeypatch, ome_pyramid, raw_zarr, dataset):
    monkeypatch.setattr(neuroglancer, "Viewer", ViewerBase)
    viewer = bootstrap.new_viewer(dataset(SimpleNamespace(ome_pyramid=ome_pyramid, raw_zarr=raw_zarr)))
    assert viewer.state.dimensions.to_json() == {axis: [8e-9, "m"] for axis in "zyx"}


def test_an_unreadable_dataset_leaves_the_viewers_dimensions_to_neuroglancer(tmp_path):
    assert bootstrap.raw_dimensions(str(tmp_path / "missing.zarr")) is None


def test_a_viewer_keeps_2_gb_of_chunks_on_the_gpu(monkeypatch, raw_zarr):
    """Neuroglancer's default 1 GB held a few Cellpose chunks, so panning back fetched them again."""
    monkeypatch.setattr(neuroglancer, "Viewer", ViewerBase)
    viewer = bootstrap.new_viewer(raw_zarr(np.zeros((16, 16, 16), np.uint8)))
    assert viewer.state.gpu_memory_limit == 2_000_000_000
