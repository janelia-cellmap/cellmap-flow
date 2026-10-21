"""What the user set on a layer survives its redraw, in both viewers.

Every path that draws a layer again in place of one of the same name (a
Submit, a model's server coming up, a finetune's next iteration, a mark
drawn) builds the layer anew, and neuroglancer showed it as built: the
user's additive blending, opacity or segment colouring were back at their
defaults. cellmap_flow.viewer.layers.replace_layer keeps them in the Python
dashboard and web/src/viewer/state.ts's replaceLayers in the browser's, by
one rule (carried_layer_json, carriedLayer), over one list of the
properties that belong to a source's data: the browser reads
SOURCE_BOUND_LAYER_KEYS from the Python's, exported (web/build/export_data.py,
kept fresh by tests/web/test_python_data_fresh.py). The rule's cases are
the vitest suite's (web/test/unit/viewer-state.test.ts), and the Python
dashboard's redraws keep the user's settings at the route
(tests/utils/test_pipeline_route.py).
"""

import neuroglancer

from cellmap_flow.viewer.layers import replace_layer


def test_replace_layer_keeps_the_users_settings_and_hands_a_layer_over():
    state = neuroglancer.ViewerState()
    state.layers["data"] = neuroglancer.ImageLayer(source="zarr://http://x/raw")
    state.layers["m-ft1"] = neuroglancer.ImageLayer(source="zarr://http://h/ft1", shader="s", blend="additive", opacity=0.4)
    state.layers["m-ft1"].visible = False
    state.layers["z"] = neuroglancer.ImageLayer(source="zarr://http://h/z")

    replace_layer(state.layers, "m-ft2", neuroglancer.ImageLayer(source="zarr://http://h/ft2", shader="new"), replacing="m-ft1")
    assert [layer.name for layer in state.layers] == ["data", "m-ft2", "z"], "in the old one's place, the old one gone"
    ft2 = state.layers["m-ft2"]
    assert (ft2.blend, ft2.opacity, ft2.visible, ft2.shader) == ("additive", 0.4, False, "new")

    replace_layer(state.layers, "data", neuroglancer.ImageLayer(source="zarr://http://x/raw", shader="raw2"))
    assert [layer.name for layer in state.layers] == ["data", "m-ft2", "z"] and state.layers["data"].visible
