"""The dashboard's session (dashboard.state) keeps its state on g, and only there.

conftest restores ``vars(g)`` after each test, and the CLIs and the job
manager still read g, so anything stored beside it would outlive a test and
disagree with them.
"""

from types import SimpleNamespace

import pytest

from cellmap_flow.dashboard.state import get_session
from cellmap_flow.globals import SERVER_CONFIG_KEYS, g
from cellmap_flow.jobs.site import current_site
from cellmap_flow.pipeline_spec import PipelineSpec


@pytest.mark.parametrize("name, on_g", [
    pytest.param("viewer", "viewer", id="viewer"),
    pytest.param("neuroglancer_url", "NEUROGLANCER_URL", id="neuroglancer-url"),
    pytest.param("minio_state", "minio_state", id="minio-state"),
    *[pytest.param(key, key, id=f"setting-{key}") for key in SERVER_CONFIG_KEYS],
])
def test_the_session_reads_and_writes_g(name, on_g):
    value = object()
    setattr(get_session(), name, value)
    assert getattr(g, on_g) is value and getattr(get_session(), name) is value


def test_the_session_stores_nothing_of_its_own():
    with pytest.raises(AttributeError):
        get_session().dataset_pth = "/a/misspelt/attribute"
    with pytest.raises(AttributeError):
        get_session().postprocess = []  # set_pipeline() is the one way to change the chain


@pytest.mark.parametrize("step", ["change", "check"])
def test_every_test_starts_from_the_state_a_new_process_has(step):
    """A canary for conftest's isolation. "change" leaves the viewer, the
    queue, the started jobs and the chain changed, the jobs in place as
    start_hosts appends them; "check" runs after it and must find each as a
    process starts with it."""
    session = get_session()
    if step == "change":
        session.viewer = object()
        session.queue = "gpu_elsewhere"
        session.jobs.append(SimpleNamespace(model_name="leaked", host="http://leaked:8000"))
        session.set_pipeline(PipelineSpec([{"name": "MinMaxNormalizer"}], [{"name": "SigmoidPostprocessor"}]))
        return
    assert (session.viewer, session.queue, session.jobs) == (None, current_site().default_queue, [])
    assert (session.input_norms, session.postprocess, session.pipeline_spec) == ([], [], PipelineSpec())
