"""A new viewer's coordinate space (viewer.bootstrap).

The raw data decides it, from the finest level of its pyramid, before any
layer is added; otherwise neuroglancer takes it from whichever layer it
likes, such as an extra layer at another voxel size. What the dashboard's
viewers hold is checked in test_layer_sources_snapshot.
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


def test_a_viewer_is_x_y_z_so_its_main_panel_is_the_xy_plane(monkeypatch, raw_zarr):
    # Neuroglancer's "xy" panel shows the first two dimensions: in z, y, x
    # order, z across and y down. The display dimensions are left to follow
    # them, as the painting fork's brush needs.
    monkeypatch.setattr(neuroglancer, "Viewer", ViewerBase)
    viewer = bootstrap.new_viewer(raw_zarr(np.zeros((16, 16, 16), np.uint8)))
    assert list(viewer.state.dimensions.names) == ["x", "y", "z"]
    assert not viewer.state.display_dimensions
    # Each axis keeps its own voxel size; without all three, the order given.
    space = bootstrap.viewer_space(["z", "y", "x"], [40, 8, 4])
    assert space.to_json() == {"x": [4e-9, "m"], "y": [8e-9, "m"], "z": [40e-9, "m"]}
    assert list(space.names) == ["x", "y", "z"]
    assert list(bootstrap.viewer_space(["y", "x"], [8, 4]).names) == ["y", "x"]


@pytest.mark.parametrize("given, expected", [
    # Neuroglancer's name for a server on every interface: the machine's DNS
    # name, which off campus still pointed at the laptop's campus address.
    ("http://ackermand-lm3.hhmi.org:43210/v/abc/", "http://192.168.1.170:43210/v/abc/"),
    # A server bound to one address is reachable there alone.
    ("http://127.0.0.1:43210/v/abc/", "http://127.0.0.1:43210/v/abc/"),
    ("http://localhost:43210/v/abc/", "http://localhost:43210/v/abc/"),
    ("http://[::1]:43210/v/abc/", "http://[::1]:43210/v/abc/"),
    ("http://ackermand-lm3/v/abc/", "http://192.168.1.170/v/abc/"),
    # Not a URL at all (a viewer whose server never started): left alone.
    ("<neuroglancer.viewer_base.ViewerBase object>", "<neuroglancer.viewer_base.ViewerBase object>"),
])
def test_a_browser_is_given_the_viewer_by_this_machines_address(monkeypatch, given, expected):
    class Viewer:  # a neuroglancer.Viewer's str() is its URL
        def __str__(self):
            return given

    monkeypatch.setattr(bootstrap, "this_machine_ip", lambda: "192.168.1.170")
    assert bootstrap.viewer_url(Viewer()) == expected
