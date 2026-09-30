"""What each path that builds viewer layers puts in the viewer.

The paths:
- the CLI's startup viewer (``neuroglancer_utils.generate_neuroglancer_url``);
- Submit (``/api/process``);
- a model started from the Models tab, from the catalog or Hugging Face;
- a finetuned model's layer (``dashboard.finetune_layers``);
- the viewers ``/api/set-data`` and the bounding-box tool open.

Each prediction layer is pinned as neuroglancer receives it: its type, its
source's URL and the chain blob in it, the scales its source transform
declares, and its shader. What a viewer is given is what users see, so a
change here belongs in a commit marked as a behaviour change. The raw
layer's own content is get_raw_layer's, pinned in test_viewer; here only
where it goes.

The raw data is one level at 24, 12, 12 nm. Four model servers answer, each
writing 16 nm voxels:
- "old" predates ``effective_output_voxel_size`` and ``has_channel``;
- "new" reads the raw at its own scale, 24, 12, 12 nm, so nothing was
  relabelled;
- "flat" writes no channel axis;
- "silent" predates ``model_info`` itself, so only its config knows its
  voxel size.
"""

import json
from types import SimpleNamespace

import neuroglancer
import pytest
from neuroglancer.viewer_base import ViewerBase

from cellmap_flow.globals import g
from cellmap_flow.norm.input_normalize import MinMaxNormalizer
from cellmap_flow.pipeline_spec import PipelineSpec
from cellmap_flow.post.postprocessors import ThresholdPostprocessor
from cellmap_flow.utils import server_info
from cellmap_flow.utils.web_utils import ARGS_KEY, decode_to_json, get_norms_post_args

RAW_VOXEL_SIZE = (24, 12, 12)
MODEL_INFO = {
    "http://old:8000": {"output_voxel_size": [16, 16, 16], "output_class": "unit"},
    "http://new:8000": {"output_voxel_size": [16, 16, 16], "output_class": "unit",
                        "effective_output_voxel_size": [16, 16, 16], "has_channel": True,
                        "output_axes": ["z", "y", "x", "c"]},
    "http://flat:8000": {"output_voxel_size": [16, 16, 16], "output_class": "unit",
                         "effective_output_voxel_size": [24, 12, 12], "has_channel": False,
                         "output_axes": ["z", "y", "x"]},
    "http://silent:8000": None,  # a 404
}
MODELS = ("old", "new", "flat", "silent")
INPUT_NORM = [MinMaxNormalizer(0, 255)]

# The scales a layer overlaid on the raw declares: the raw's, and the channel axis
# if the served array has one.
OVERLAID = {"z": 24.0, "y": 12.0, "x": 12.0, "c^": 1}
OVERLAID_3D = {"z": 24.0, "y": 12.0, "x": 12.0}
EIGHT_NM = {"z": 8.0, "y": 8.0, "x": 8.0}
# The live chain as a layer URL carries it.
CHAIN = {"input_norm": [{"name": "MinMaxNormalizer", "min_value": 0.0, "max_value": 255.0, "invert": False}],
         "postprocess": []}
THRESHOLD = [{"name": "ThresholdPostprocessor", "threshold": 0.5}]


def _unit(color):
    """The shader over an output in [0, 1]."""
    return ('#uicontrol invlerp normalized(range=[0, 1], window=[-0.5, 1.5]);\n'
            f'#uicontrol vec3 color color(default="{color}");\n'
            'void main(){emitRGB(color * normalized());}')


class _Answer:
    def __init__(self, info):
        self.status_code, self._info = (404 if info is None else 200), info

    def json(self):
        return self._info


@pytest.fixture
def servers(monkeypatch, ome_pyramid):
    """The four model servers, the raw data as g.dataset_path, and viewers without a web server."""
    monkeypatch.setattr(server_info.requests, "get",
                        lambda url, timeout=None: _Answer(MODEL_INFO[url.split("/__control__")[0]]))
    monkeypatch.setattr(neuroglancer, "Viewer", ViewerBase)
    g.dataset_path = ome_pyramid(((RAW_VOXEL_SIZE, None),))
    g.jobs = [SimpleNamespace(model_name=name, host=f"http://{name}:8000") for name in MODELS]
    g.models_config = [SimpleNamespace(name="silent", config=SimpleNamespace(output_voxel_size=(16, 16, 16)))]
    g.shaders, g.shader_controls, g.extra_layers = {}, {}, {}
    g.set_pipeline(PipelineSpec.from_steps(INPUT_NORM, []))
    return g.dataset_path


def _scales(dimensions):
    """{axis: nm, or the unit-less size of a channel axis}."""
    return {axis: round(size * 1e9, 6) if unit == "m" else size for axis, (size, unit) in dimensions.items()}


def _layer(layer):
    """(the chain blob in its URL, (type, URL without the blob, the scales its
    source transform declares or None, shader[, shaderControls]))."""
    data = layer.to_json()
    (source,) = data["source"] if isinstance(data["source"], list) else [data["source"]]
    source = {"url": source} if isinstance(source, str) else source
    url, blob, suffix = source["url"].split(ARGS_KEY)
    assert suffix == "", source["url"]
    transform = source.get("transform")
    if transform is not None:
        assert set(transform) == {"inputDimensions", "outputDimensions"}, transform
        assert transform["inputDimensions"] == transform["outputDimensions"], transform
    pinned = (data["type"], url, _scales(transform["outputDimensions"]) if transform else None, data.get("shader"))
    if "shaderControls" in data:
        pinned += (data["shaderControls"],)
    return decode_to_json(blob), pinned


def _viewer(viewer, predictions=()):
    """The viewer's layers in order, its dimensions, the chain every prediction
    layer carries, and each prediction layer."""
    state = viewer.state
    pinned = {
        "layers": [(layer.name, layer.type) for layer in state.layers],
        "dimensions": _scales(state.dimensions.to_json()),
    }
    layers = {name: _layer(state.layers[name]) for name in predictions}
    chains = [chain for chain, _ in layers.values()]
    if chains:
        assert all(chain == chains[0] for chain in chains), chains
        pinned["chain"] = chains[0]
    return {**pinned, **{name: layer for name, (_, layer) in layers.items()}}


def test_the_startup_viewer(servers, monkeypatch):
    from cellmap_flow.utils import neuroglancer_utils

    served = []
    monkeypatch.setattr(neuroglancer_utils, "create_and_run_app", lambda neuroglancer_url: served.append(neuroglancer_url))
    # Not up yet: a zarr://None/... layer would never load, and nothing replaces it later.
    g.jobs = g.jobs + [SimpleNamespace(model_name="queued", host=None)]
    g.extra_layers = {"extra": neuroglancer.ImageLayer(source="zarr://http://files/extra.zarr")}

    neuroglancer_utils.generate_neuroglancer_url(servers)

    assert served == [str(g.viewer)]
    assert _viewer(g.viewer, MODELS) == {
        "layers": [("data", "image"), ("old", "image"), ("new", "image"), ("flat", "image"), ("silent", "image"),
                   ("extra", "image")],
        "dimensions": {"z": 24.0, "y": 12.0, "x": 12.0},
        "chain": CHAIN,
        "old": ("image", "zarr://http://old:8000/old", OVERLAID, _unit("red")),
        "new": ("image", "zarr://http://new:8000/new", None, _unit("green")),
        "flat": ("image", "zarr://http://flat:8000/flat", OVERLAID_3D, _unit("blue")),
        "silent": ("image", "zarr://http://silent:8000/silent", OVERLAID, _unit("yellow")),
    }


def test_the_startup_viewer_shows_a_labelling_chain_as_segmentations(servers, monkeypatch):
    from cellmap_flow.utils import neuroglancer_utils

    monkeypatch.setattr(neuroglancer_utils, "create_and_run_app", lambda neuroglancer_url: None)
    g.set_pipeline(PipelineSpec.from_steps(INPUT_NORM, [ThresholdPostprocessor(0.5)]))

    neuroglancer_utils.generate_neuroglancer_url(servers)

    assert _viewer(g.viewer, ["old"]) == {
        "layers": [("data", "image"), ("old", "segmentation"), ("new", "segmentation"), ("flat", "segmentation"),
                   ("silent", "segmentation")],
        "dimensions": {"z": 24.0, "y": 12.0, "x": 12.0},
        "chain": dict(CHAIN, postprocess=THRESHOLD),
        "old": ("segmentation", "zarr://http://old:8000/old", OVERLAID, None),
    }


SUBMITTED = {"input_norm": [{"name": "MinMaxNormalizer", "min_value": 0, "max_value": 255}]}


@pytest.fixture
def submit(servers, dashboard, viewer):
    """Submit ``postprocess`` with a job still queued, over a viewer where the
    user had given "old" a shader and a control of their own."""
    def post(postprocess):
        g.jobs = g.jobs + [SimpleNamespace(model_name="queued", host=None)]
        with viewer.txn() as s:
            s.layers["old"] = neuroglancer.ImageLayer(source="zarr://http://old:8000/old", shader="void main() {}",
                                                      shader_controls={"brightness": 0.5})
        payload = dict(SUBMITTED, postprocess=postprocess)
        assert dashboard.post("/api/process", data=json.dumps(payload), content_type="application/json").status_code == 200
        return _viewer(viewer, MODELS)

    return post


def test_submit(submit):
    assert submit([]) == {
        "layers": [("old", "image"), ("data", "image"), ("new", "image"), ("flat", "image"), ("silent", "image")],
        "dimensions": EIGHT_NM,
        "chain": dict(SUBMITTED, postprocess=[], dashboard_url="http://localhost/", digest="5ba6d02f85d55e92"),
        "old": ("image", "zarr://http://old:8000/old", OVERLAID, "void main() {}", {"brightness": 0.5}),
        "new": ("image", "zarr://http://new:8000/new", None, _unit("green")),
        "flat": ("image", "zarr://http://flat:8000/flat", OVERLAID_3D, _unit("blue")),
        "silent": ("image", "zarr://http://silent:8000/silent", None, _unit("yellow")),
    }


def test_submit_shows_a_labelling_chain_as_segmentations(submit):
    assert submit(THRESHOLD) == {
        "layers": [("old", "segmentation"), ("data", "image"), ("new", "segmentation"), ("flat", "segmentation"),
                   ("silent", "segmentation")],
        "dimensions": EIGHT_NM,
        "chain": dict(SUBMITTED, postprocess=THRESHOLD, dashboard_url="http://localhost/", digest="724f969450d08854"),
        "old": ("segmentation", "zarr://http://old:8000/old", OVERLAID, None),
        "new": ("segmentation", "zarr://http://new:8000/new", None, None),
        "flat": ("segmentation", "zarr://http://flat:8000/flat", OVERLAID_3D, None),
        "silent": ("segmentation", "zarr://http://silent:8000/silent", None, None),
    }


@pytest.mark.parametrize("launch", ["catalog", "huggingface"])
def test_a_model_started_from_the_models_tab(servers, viewer, monkeypatch, launch):
    from cellmap_flow.dashboard.services import launch

    started = []

    def start_hosts(command, job_name, queue=None, charge_group=None):
        started.append(job_name)
        job = SimpleNamespace(model_name=job_name, host=f"http://{job_name}:8000")
        g.jobs = g.jobs + [job]
        return job

    monkeypatch.setattr(launch, "start_hosts", start_hosts)
    g.jobs = []
    blob = get_norms_post_args(INPUT_NORM, [])
    for name in MODELS:
        if launch == "catalog":
            launch.run_model(f"/models/{name}", name, blob)
        else:
            launch.run_hf_model(f"cellmap/{name}", name, blob)

    assert started == list(MODELS)
    assert _viewer(viewer, MODELS) == {
        "layers": [(name, "image") for name in MODELS],
        "dimensions": EIGHT_NM,
        "chain": CHAIN,
        "old": ("image", "zarr://http://old:8000/old", OVERLAID, _unit("red")),
        "new": ("image", "zarr://http://new:8000/new", None, _unit("green")),
        "flat": ("image", "zarr://http://flat:8000/flat", OVERLAID_3D, _unit("blue")),
        "silent": ("image", "zarr://http://silent:8000/silent", None, _unit("yellow")),
    }


@pytest.mark.parametrize("server, postprocess, layer", [
    pytest.param("old", [], ("image", OVERLAID, _unit("red")), id="old"),
    pytest.param("new", [], ("image", None, _unit("red")), id="new"),
    pytest.param("flat", [], ("image", OVERLAID_3D, _unit("red")), id="flat"),
    pytest.param("old", [ThresholdPostprocessor(0.5)], ("segmentation", OVERLAID, None), id="a labelling chain"),
])
def test_a_finetuned_models_layer(servers, viewer, server, postprocess, layer):
    from cellmap_flow.dashboard.finetune_layers import add_finetuned_layer

    g.set_pipeline(PipelineSpec.from_steps(INPUT_NORM, postprocess))
    job = SimpleNamespace(model_name=server, lsf_job=SimpleNamespace(job_id="7"), finetuned_model_name=None,
                          inference_server_url=f"http://{server}:8000", params={"output_voxel_size": [16, 16, 16]})
    add_finetuned_layer(job, f"{server}_finetuned_1")

    kind, scales, shader = layer
    name = f"{server}_finetuned_1"
    assert _viewer(viewer, [name]) == {
        "layers": [(name, kind)],
        "dimensions": EIGHT_NM,
        "chain": dict(CHAIN, postprocess=[step.to_dict() for step in postprocess]),
        name: (kind, f"zarr://http://{server}:8000/{name}", scales, shader),
    }


def test_the_viewers_set_data_and_the_box_tool_open(servers, dashboard):
    assert dashboard.post("/api/set-data", json={"dataset_path": servers}).status_code == 200
    assert _viewer(g.viewer) == {"layers": [("data", "image")], "dimensions": EIGHT_NM}
    drawn = [{"offset": [8, 16, 24], "shape": [80, 40, 40]}]
    assert dashboard.post("/api/bbx-generator", json={"dataset_path": servers, "existing_bounding_boxes": drawn}
                          ).status_code == 200
    viewer = g.bbx_generator_state["viewer"]
    assert _viewer(viewer) == {"layers": [("fibsem", "image"), ("bboxes", "annotation")], "dimensions": EIGHT_NM}
    boxes = viewer.state.layers["bboxes"].to_json()
    assert (boxes["source"], boxes["annotations"]) == (
        [{"url": "local://annotations", "transform": {"outputDimensions": {axis: [1e-9, "m"] for axis in "zyx"}}}],
        [{"type": "axis_aligned_bounding_box", "pointA": [8.0, 16.0, 24.0], "pointB": [88.0, 56.0, 64.0],
          "id": "bbox-1", "description": "Bounding box 1"}],
    )
