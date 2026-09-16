"""Bounding boxes must be read from the layer they were drawn in.

neuroglancer's Layers.__getitem__ resolves a name through index(), which
returns -1 when the name is absent -- so s.layers["missing"] silently
returns _layers[-1], the last layer, rather than raising KeyError. The box
extraction asked for a layer name that never existed ("annotations") and
worked only because the real layer ("bboxes") happened to be created last.
Anything appended after it would have taken its place, silently.
"""

import neuroglancer
from neuroglancer import AxisAlignedBoundingBoxAnnotation as BBox

from cellmap_flow.dashboard.routes.bbx_generator import (
    BBOX_LAYER_NAME,
    _extract_bounding_boxes,
)


class _FakeViewer:
    """Just enough of a neuroglancer viewer to exercise a txn()."""

    def __init__(self, state):
        self._state = state

    def txn(self):
        state = self._state

        class _Txn:
            def __enter__(self):
                return state

            def __exit__(self, *exc):
                return False

        return _Txn()


def _state_with_boxes(*, trailing_layer: bool):
    s = neuroglancer.viewer_state.ViewerState()
    s.layers["fibsem"] = neuroglancer.ImageLayer(source="zarr://http://x/y")
    s.layers[BBOX_LAYER_NAME] = neuroglancer.LocalAnnotationLayer(
        dimensions=neuroglancer.CoordinateSpace(
            names=["z", "y", "x"], units="nm", scales=[1, 1, 1]
        )
    )
    s.layers[BBOX_LAYER_NAME].annotations.append(
        BBox(id="1", point_a=[10, 20, 30], point_b=[40, 60, 80])
    )
    if trailing_layer:
        # The case that broke it: any layer added after the box layer.
        s.layers["good_regions"] = neuroglancer.LocalAnnotationLayer(
            dimensions=neuroglancer.CoordinateSpace(
                names=["z", "y", "x"], units="nm", scales=[1, 1, 1]
            )
        )
    return _FakeViewer(s)


def test_missing_name_really_does_return_the_last_layer():
    """Pin the neuroglancer behaviour this guards against."""
    s = neuroglancer.viewer_state.ViewerState()
    s.layers["first"] = neuroglancer.ImageLayer(source="zarr://http://x/y")
    s.layers["last"] = neuroglancer.ImageLayer(source="zarr://http://x/z")
    assert s.layers["no_such_layer"] is s.layers["last"]
    assert "no_such_layer" not in s.layers, "membership is the safe check"


def test_boxes_are_read_from_the_box_layer():
    boxes = _extract_bounding_boxes(_state_with_boxes(trailing_layer=False))
    assert boxes == [{"offset": [10, 20, 30], "shape": [30, 40, 50]}]


def test_a_layer_added_after_the_box_layer_does_not_steal_the_lookup():
    boxes = _extract_bounding_boxes(_state_with_boxes(trailing_layer=True))
    assert boxes == [{"offset": [10, 20, 30], "shape": [30, 40, 50]}]


def test_no_box_layer_yields_no_boxes_rather_than_another_layers_contents():
    """The decisive case.

    There is no box layer, but there IS another annotation layer holding
    boxes of its own. An unguarded lookup lands on _layers[-1] and reports
    that layer's boxes as though the user had drawn them.
    """
    s = neuroglancer.viewer_state.ViewerState()
    s.layers["fibsem"] = neuroglancer.ImageLayer(source="zarr://http://x/y")
    s.layers["good_regions"] = neuroglancer.LocalAnnotationLayer(
        dimensions=neuroglancer.CoordinateSpace(
            names=["z", "y", "x"], units="nm", scales=[1, 1, 1]
        )
    )
    s.layers["good_regions"].annotations.append(
        BBox(id="decoy", point_a=[1, 1, 1], point_b=[2, 2, 2])
    )
    assert BBOX_LAYER_NAME not in s.layers
    assert _extract_bounding_boxes(_FakeViewer(s)) == []


def test_no_viewer_is_not_an_error():
    assert _extract_bounding_boxes(None) == []
