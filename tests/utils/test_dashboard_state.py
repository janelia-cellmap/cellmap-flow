"""The dashboard's session (dashboard.state): what it forwards to the owners it
shares with processes that have no dashboard, and what it refuses.

The launcher settings, the started jobs and the chain are read by
start_hosts, cleanup_handler and the servers from their owners; a session
that kept its own copy would disagree with them.
"""

from types import SimpleNamespace

import pytest

from cellmap_flow.dashboard.state import get_session
from cellmap_flow.jobs.launch import started_jobs
from cellmap_flow.jobs.settings import SERVER_CONFIG_KEYS, launcher_settings
from cellmap_flow.jobs.site import current_site
from cellmap_flow.pipeline_spec import PipelineSpec


@pytest.mark.parametrize("name, owner", [
    *[pytest.param(key, lambda key=key: getattr(launcher_settings(), key), id=f"setting-{key}")
      for key in SERVER_CONFIG_KEYS],
    pytest.param("jobs", started_jobs, id="jobs"),
])
def test_the_session_reads_and_writes_the_owners_state(name, owner):
    value = [object()]
    setattr(get_session(), name, value)
    assert owner() == value and getattr(get_session(), name) == value


def test_assigning_the_jobs_keeps_the_list_start_hosts_appends_to():
    started = started_jobs()
    get_session().jobs = [SimpleNamespace(model_name="mito")]
    assert started_jobs() is started and [job.model_name for job in started] == ["mito"]


def test_a_misspelt_or_read_only_attribute_raises():
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
