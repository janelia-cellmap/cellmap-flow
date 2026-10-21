"""What the viewers hold once a dataset is opened, and once a model is
started from the Models tab.

The raw data is one level at 24, 12, 12 nm, and each model's server answers
that it writes 16 nm voxels. What a viewer is given is what users see. The
raw layer's own content is get_raw_layer's, checked in test_viewer; the chain
a redrawn layer carries, and what the user set on it, test_pipeline_route's.
"""

from types import SimpleNamespace

import neuroglancer
import pytest
from neuroglancer.viewer_base import ViewerBase

from cellmap_flow.dashboard.state import get_session
from cellmap_flow.norm.input_normalize import MinMaxNormalizer
from cellmap_flow.pipeline_spec import PipelineSpec
from cellmap_flow.serving import client
from cellmap_flow.serving.protocol import split_dataset_url

MODELS = ("mito", "nuc")


class _Answer:
    """A model server's model_info."""

    status_code = 200

    def json(self):
        return {"output_voxel_size": [16, 16, 16], "output_class": "unit", "effective_output_voxel_size": [16, 16, 16],
                "has_channel": True, "output_axes": ["c", "z", "y", "x"], "engine": "chunkmirage"}


@pytest.fixture
def raw(monkeypatch, ome_pyramid):
    """The raw data as the session's dataset_path, model servers that answer,
    and viewers without a web server."""
    monkeypatch.setattr(client.requests, "get", lambda url, timeout=None: _Answer())
    monkeypatch.setattr(neuroglancer, "Viewer", ViewerBase)
    get_session().dataset_path = ome_pyramid((((24, 12, 12), None),))
    get_session().set_pipeline(PipelineSpec.from_steps([MinMaxNormalizer(0, 255)], []))
    return get_session().dataset_path


def _source_url(layer):
    source = layer.to_json()["source"]
    source = source[0] if isinstance(source, list) else source
    return source["url"] if isinstance(source, dict) else source


def test_the_viewers_set_data_and_the_box_tool_open(raw, dashboard):
    """The dashboard's viewer shows the dataset; the bounding-box tool's shows
    it too, with the boxes drawn so far where they were drawn."""
    assert dashboard.post("/api/set-data", json={"dataset_path": raw}).status_code == 200
    assert [layer.name for layer in get_session().viewer.state.layers] == ["data"]
    drawn = [{"offset": [8, 16, 24], "shape": [80, 40, 40]}]
    assert dashboard.post("/api/bbx-generator", json={"dataset_path": raw, "existing_bounding_boxes": drawn}
                          ).status_code == 200
    layers = get_session().bbx_generator_state["viewer"].state.layers
    assert [(layer.name, layer.type) for layer in layers] == [("fibsem", "image"), ("bboxes", "annotation")]
    (box,) = layers["bboxes"].to_json()["annotations"]
    assert (box["pointA"], box["pointB"]) == ([8.0, 16.0, 24.0], [88.0, 56.0, 64.0])


@pytest.mark.parametrize("source", ["catalog", "huggingface"])
def test_a_model_started_from_the_models_tab_is_drawn_from_its_server_through_the_chain(raw, viewer, monkeypatch,
                                                                                       source):
    """Each model gets a layer of its name, reading its own server, through
    the chain configured when it was started."""
    from cellmap_flow.dashboard.services import launch

    started = []

    def start_hosts(command, job_name, queue=None, charge_group=None):
        started.append(job_name)
        job = SimpleNamespace(model_name=job_name, host=f"http://{job_name}:8000")
        get_session().jobs = get_session().jobs + [job]
        return job

    monkeypatch.setattr(launch, "start_hosts", start_hosts)
    get_session().jobs = []
    blob = get_session().pipeline_spec.to_url_blob()
    for name in MODELS:
        if source == "catalog":
            launch.run_model(f"/models/{name}", name, blob)
        else:
            launch.run_hf_model(f"cellmap/{name}", name, blob)

    assert started == list(MODELS)
    layers = viewer.state.layers
    assert [layer.name for layer in layers] == list(MODELS)
    for name in MODELS:
        url = _source_url(layers[name])
        assert url.startswith(f"zarr://http://{name}:8000/{name}"), url
        assert PipelineSpec.from_url_blob(split_dataset_url(url))[0] == get_session().pipeline_spec
