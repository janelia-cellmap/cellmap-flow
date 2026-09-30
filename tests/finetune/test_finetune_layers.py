"""The dashboard's finetune listener (dashboard.finetune_layers): a finetuned
model's viewer layer and its model in the pipeline builder, as the job manager
tells it of the job. When the manager tells it is test_finetune_job_manager's;
the layer's source and shader are pinned in test_layer_sources_snapshot."""

from types import SimpleNamespace

import pytest

from cellmap_flow.dashboard.finetune_layers import FinetuneLayerListener
from cellmap_flow.globals import g

URL = "http://node7:8123"


class _Answer:
    status_code = 200

    def json(self):
        return {"output_class": None}


@pytest.fixture
def followed(make_job, viewer, monkeypatch):
    """The listener told of a local run (a process, no LSF job id) as the
    manager tells it: iteration 1 before its server, the server, then
    iteration 2; the manager's name for the job's model is set after each
    event. The viewer's layer names after each."""
    from cellmap_flow.post.postprocessors import SigmoidPostprocessor
    from cellmap_flow.utils import server_info

    monkeypatch.setattr(server_info.requests, "get", lambda url, timeout=None: _Answer())
    for key, value in dict(jobs=[], models_config=[], input_norms=[], postprocess=[SigmoidPostprocessor()]).items():
        monkeypatch.setattr(g, key, value)
    job = make_job(lsf_job=SimpleNamespace(process=SimpleNamespace(pid=99)))
    listener, layers = FinetuneLayerListener(), []

    def event(name, *args):
        getattr(listener, name)(job, *args)
        job.finetuned_model_name = args[-1]
        layers.append([layer.name for layer in viewer.state.layers])

    event("on_iteration_complete", "m_finetuned_1")
    job.inference_server_url = URL
    event("on_server_ready", URL, "m_finetuned_1")
    event("on_iteration_complete", "m_finetuned_2")
    return SimpleNamespace(layers=layers, state=viewer.state)


def test_the_viewer_gets_the_layer_once_the_server_is_up_and_each_iteration_replaces_it(followed):
    """A layer was added before the server existed, with the source zarr://None/...;
    and a local run's LocalJob has no job_id, so adding its layer raised."""
    assert followed.layers == [[], ["m_finetuned_1"], ["m_finetuned_2"]]
    assert [job.job_id for job in g.jobs] == ["local"]


def test_the_finetuned_layer_shows_the_outputs_own_range(followed):
    """It was always [0, 255], so a sigmoid's [0, 1] rendered black, and the
    finetuned model looked worse than the same one added the normal way."""
    assert "range=[0, 1]" in followed.state.layers["m_finetuned_2"].shader


def test_the_pipeline_builder_gets_each_iterations_model(followed):
    assert [config.name for config in g.models_config] == ["m_finetuned_2"]
