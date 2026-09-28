"""What the dashboard's two server-rendered pages hand to the browser."""

from types import SimpleNamespace

import pytest

from cellmap_flow.dashboard.app import app
from cellmap_flow.globals import g


@pytest.fixture
def client():
    return app.test_client()


def _job(name):
    return SimpleNamespace(model_name=name, host="http://node:8000")


def test_rendering_the_index_leaves_the_model_catalog_alone(client):
    catalog_before = {k: dict(v) for k, v in g.model_catalog.items()}
    g.jobs = [_job("mito_server")]

    response = client.get("/")

    assert response.status_code == 200
    # The running model is still listed on the Models tab...
    assert 'value="mito_server"' in response.get_data(as_text=True)
    # ...but not added to the catalog everything else reads.
    assert g.model_catalog == catalog_before
    assert "User" not in g.model_catalog or "mito_server" not in g.model_catalog["User"]
