"""The dashboard listens on every interface, so it must not hand out arbitrary
files, and other sites must not be able to script it through the browser."""

import pytest

from cellmap_flow.dashboard.app import app

READ_YAML = "/api/finetune/read-yaml"


@pytest.fixture
def client():
    return app.test_client()


def test_a_yaml_file_is_served(client, tmp_path):
    crops = tmp_path / "crops.yaml"
    crops.write_text("crops: []\n")
    response = client.get(READ_YAML, query_string={"path": str(crops)})
    assert response.status_code == 200
    assert response.get_json()["text"] == "crops: []\n"


@pytest.mark.parametrize("name", ["id_rsa", "notes.txt", "settings.yaml.bak"])
def test_other_files_are_refused_whether_or_not_they_exist(client, tmp_path, name):
    existing = tmp_path / name
    existing.write_text("secret")
    for path in (existing, tmp_path / f"missing_{name}"):
        response = client.get(READ_YAML, query_string={"path": str(path)})
        assert response.status_code == 400
        assert "secret" not in response.get_data(as_text=True)


def test_a_yaml_named_link_to_another_file_is_refused(client, tmp_path):
    secret = tmp_path / "id_rsa"
    secret.write_text("secret")
    link = tmp_path / "crops.yaml"
    link.symlink_to(secret)
    response = client.get(READ_YAML, query_string={"path": str(link)})
    assert response.status_code == 400
    assert "secret" not in response.get_data(as_text=True)


def test_other_origins_get_no_cors_grant(client, tmp_path):
    crops = tmp_path / "crops.yaml"
    crops.write_text("crops: []\n")
    origin = {"Origin": "https://elsewhere.example"}
    response = client.get(READ_YAML, query_string={"path": str(crops)}, headers=origin)
    assert "Access-Control-Allow-Origin" not in response.headers

    preflight = client.options(
        "/api/models",
        headers={**origin, "Access-Control-Request-Method": "POST"},
    )
    assert "Access-Control-Allow-Origin" not in preflight.headers
