"""What the user set on a layer survives its redraw, in both viewers.

Every path that draws a layer again in place of one of the same name (a
Submit, a model's server coming up, a finetune's next iteration, a mark
drawn) builds the layer anew, and neuroglancer showed it as built: the
user's additive blending, opacity or segment colouring were back at their
defaults. cellmap_flow.viewer.layers.replace_layer keeps them in the Python
dashboard and web/src/viewer/state.ts's replaceLayers in the browser's, by
one rule (carried_layer_json, carriedLayer), whose list of the properties
that belong to a source's data must be the same in both: checked here,
with the rule's cases (the vitest suite has the same ones for the
TypeScript, web/test/unit/viewer-state.test.ts).
"""

import re
from pathlib import Path

import neuroglancer

from cellmap_flow.viewer.layers import SOURCE_BOUND_LAYER_KEYS, carried_layer_json, replace_layer

STATE_TS = Path(__file__).resolve().parents[2] / "web" / "src" / "viewer" / "state.ts"

# A model layer as the user left it: additive, half opaque, finer
# resolution, channel 2 shown, its contrast moved. Without its name, which
# replace_layer gives the managed layer.
SHOWN = {
    "type": "image", "source": "zarr3://http://h/mito__CFLOW_ARGS__a__CFLOW_ARGS__/",
    "shader": "user's", "shaderControls": {"normalized": {"range": [0, 9]}},
    "blend": "additive", "opacity": 0.5, "crossSectionRenderScale": 0.5,
    "localDimensions": {"c^": [1, ""]}, "localPosition": [2.5], "tab": "rendering",
}


def test_the_browser_keeps_the_same_properties_with_their_source():
    block = re.search(r"SOURCE_BOUND_LAYER_KEYS: readonly string\[\] = \[(.*?)\];", STATE_TS.read_text(), re.S)
    assert block, "web/src/viewer/state.ts no longer declares SOURCE_BOUND_LAYER_KEYS as a list"
    assert set(re.findall(r'"([^"]+)"', block.group(1))) == SOURCE_BOUND_LAYER_KEYS


def test_every_setting_the_new_layer_does_not_make_is_kept():
    """Another model added: this one is drawn again as it was."""
    assert carried_layer_json(SHOWN, {"type": "image", "source": SHOWN["source"], "shader": "user's"}) == SHOWN


def test_the_new_layers_own_properties_win():
    merged = carried_layer_json(SHOWN, {"type": "image", "source": SHOWN["source"], "shader": "the app's", "opacity": 1})
    assert (merged["shader"], merged["opacity"], merged["blend"]) == ("the app's", 1, "additive")


def test_what_belonged_to_the_old_sources_data_goes_with_a_new_source():
    new = {"type": "image", "source": "zarr3://http://h/mito__CFLOW_ARGS__b__CFLOW_ARGS__/", "shader": "new range"}
    assert carried_layer_json(SHOWN, new) == {
        **new, "blend": "additive", "opacity": 0.5, "crossSectionRenderScale": 0.5, "tab": "rendering"}


def test_a_segmentation_keeps_its_colouring_and_its_segments_only_with_its_source():
    seg = {"type": "segmentation", "source": "zarr3://http://h/a/", "selectedAlpha": 0.8, "colorSeed": 7,
           "segments": ["5"], "equivalences": [["5", "6"]]}
    assert carried_layer_json(seg, {"type": "segmentation", "source": "zarr3://http://h/a/"}) == seg
    assert carried_layer_json(seg, {"type": "segmentation", "source": "zarr3://http://h/b/"}) == {
        "type": "segmentation", "source": "zarr3://http://h/b/", "selectedAlpha": 0.8, "colorSeed": 7}


def test_sources_are_compared_by_their_urls():
    written = {**SHOWN, "source": [{"url": SHOWN["source"], "transform": {"outputDimensions": {}}}]}
    assert carried_layer_json(written, {"type": "image", "source": SHOWN["source"]})["localPosition"] == [2.5]


def test_only_being_hidden_crosses_a_change_of_type():
    new = {"type": "segmentation", "source": "zarr3://http://h/labels/"}
    assert carried_layer_json({**SHOWN, "visible": False}, new) == {**new, "visible": False}


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
