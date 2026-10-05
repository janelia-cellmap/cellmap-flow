"""The AI-annotate config: off without a file, strict about what it holds, and
showing the browser only ids, models and destinations."""

import json

import pytest
import yaml

from cellmap_flow.ai_annotate import config
from cellmap_flow.ai_annotate.errors import AIAnnotateError

VERTEX = {
    "type": "vertex_gemini",
    "project": "secret-project-1234",
    "location": "global",
    "models": ["gemini-3-pro-image"],
    "timeout_s": 77,
}


@pytest.fixture
def write_config(tmp_path, monkeypatch):
    """``write_config(dict_or_text)``: write the config file and point the env var at it."""
    path = tmp_path / "ai.yaml"
    monkeypatch.setenv(config.CONFIG_ENV, str(path))

    def write(content):
        path.write_text(content if isinstance(content, str) else yaml.safe_dump(content))
        return path

    return write


def _enabled(**providers):
    return {"enabled": True, "providers": providers or {"vertex": dict(VERTEX)}}


def test_config_path_defaults_to_home_and_follows_the_env_var(tmp_path, monkeypatch):
    monkeypatch.delenv(config.CONFIG_ENV, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert config.config_path() == tmp_path / ".cellmap_flow" / "ai_annotate.yaml"
    monkeypatch.setenv(config.CONFIG_ENV, str(tmp_path / "x.yaml"))
    assert config.config_path() == tmp_path / "x.yaml"


def test_no_file_means_off_with_a_reason_saying_how_to_enable(tmp_path, monkeypatch):
    monkeypatch.setenv(config.CONFIG_ENV, str(tmp_path / "missing.yaml"))
    assert config.load_config() is None
    reason = config.disabled_reason()
    assert "missing.yaml" in reason and config.CONFIG_ENV in reason and "docs/ai_annotate.md" in reason


def test_enabled_false_or_absent_means_off(write_config):
    write_config({"enabled": False, "providers": {"vertex": VERTEX}})
    assert config.load_config() is None
    assert "enabled: false" in config.disabled_reason()
    write_config({"providers": {"vertex": VERTEX}})
    assert config.load_config() is None


def test_a_valid_file_loads_with_defaults(write_config):
    write_config({"enabled": True, "providers": {"fake": {"type": "fake", "models": ["fake-threshold"]}, "vertex": {
        "type": "vertex_gemini", "models": ["gemini-3-pro-image"]}}})
    loaded = config.load_config()
    assert loaded.enabled
    assert loaded.default_provider == "fake"
    assert loaded.daily_call_limit == 200 and loaded.crop_size_px == 512
    assert loaded.allowed_dataset_prefixes == ()
    vertex = loaded.provider("vertex")
    assert vertex.models == ("gemini-3-pro-image",)
    assert vertex.options == {"location": "global", "timeout_s": 120}


def test_an_inline_api_key_is_rejected_without_echoing_it(write_config):
    write_config(_enabled(vertex={**VERTEX, "api_key": "AIzaSyVERYSECRETVALUE123"}))
    with pytest.raises(AIAnnotateError) as caught:
        config.load_config()
    assert caught.value.category == "config"
    assert "api_key" in caught.value.user_message
    assert "AIzaSyVERYSECRETVALUE123" not in caught.value.user_message
    assert "api_key_env" in caught.value.user_message


@pytest.mark.parametrize("name", ["token", "Secret", "password"])
def test_other_secret_looking_settings_are_rejected(write_config, name):
    write_config(_enabled(vertex={**VERTEX, name: "hunter2hunter2"}))
    with pytest.raises(AIAnnotateError, match="secrets may not"):
        config.load_config()


@pytest.mark.parametrize(
    "change, problem",
    [
        ({"locaton": "us-east1"}, "unknown setting"),
        ({"type": "openai"}, "type must be one of"),
        ({"models": []}, "models must be"),
        ({"models": ["ok", "bad model; rm -rf"]}, "models must be"),
        ({"location": "us central"}, "location must be"),
        ({"timeout_s": 0}, "timeout_s must be"),
        ({"timeout_s": True}, "timeout_s must be"),
        ({"project": "x" * 200}, "project must be"),
    ],
)
def test_malformed_provider_settings_name_the_problem_not_the_value(write_config, change, problem):
    write_config(_enabled(vertex={**VERTEX, **change}))
    with pytest.raises(AIAnnotateError) as caught:
        config.load_config()
    assert problem in caught.value.user_message
    for value in change.values():
        if isinstance(value, str) and len(value) > 3:
            assert value not in caught.value.user_message


@pytest.mark.parametrize(
    "settings, problem",
    [
        ({"daily_call_limit": "lots"}, "daily_call_limit"),
        ({"daily_call_limit": -1}, "daily_call_limit"),
        ({"crop_size_px": 1_000_000}, "crop_size_px"),
        ({"default_provider": "nope"}, "default_provider"),
        ({"allowed_dataset_prefixes": "/nrs"}, "allowed_dataset_prefixes"),
        ({"providers": {}}, "providers"),
        ({"providers": {"bad id!": VERTEX}}, "provider ids"),
        ({"extra": 1}, "unknown setting"),
        ({"enabled": "yes"}, "enabled"),
    ],
)
def test_malformed_top_level_settings_are_errors(write_config, settings, problem):
    write_config({**_enabled(), **settings})
    with pytest.raises(AIAnnotateError, match=problem):
        config.load_config()


def test_invalid_yaml_reports_the_line_but_not_its_text(write_config):
    write_config("enabled: true\nproviders: {vertex: [\n  api_key: SECRETSECRETSECRET\n")
    with pytest.raises(AIAnnotateError) as caught:
        config.load_config()
    assert "not valid YAML" in caught.value.user_message
    assert "SECRETSECRETSECRET" not in caught.value.user_message


def test_public_holds_only_ids_types_models_and_destination(write_config):
    write_config(
        {
            **_enabled(vertex=dict(VERTEX), fake={"type": "fake", "models": ["fake-threshold"]}),
            "allowed_dataset_prefixes": ["/nrs/secret-lab/"],
            "default_provider": "vertex",
        }
    )
    public = config.load_config().public()
    assert set(public) == {"providers", "default_provider", "daily_call_limit", "crop_size_px"}
    for provider in public["providers"]:
        assert set(provider) == {"id", "type", "models", "destination"}
    text = json.dumps(public)
    for hidden in ("secret-project-1234", "77", "/nrs/secret-lab/", "timeout"):
        assert hidden not in text
    destinations = {p["id"]: p["destination"] for p in public["providers"]}
    assert "Vertex AI" in destinations["vertex"] and "any region" in destinations["vertex"]
    assert "fake" in destinations["fake"]


def test_a_regional_vertex_destination_names_its_location():
    provider = config.ProviderConfig("v", "vertex_gemini", ("m",), {"location": "europe-west4"})
    assert provider.destination() == "Google Cloud Vertex AI, location europe-west4"


def test_unknown_provider_or_model_from_the_browser_is_a_400(write_config):
    write_config(_enabled())
    loaded = config.load_config()
    with pytest.raises(AIAnnotateError) as caught:
        loaded.provider("nope")
    assert (caught.value.category, caught.value.http_status) == ("config", 400)
    loaded.check_model("vertex", "gemini-3-pro-image")
    with pytest.raises(AIAnnotateError) as caught:
        loaded.check_model("vertex", "gemini-ultra")
    assert caught.value.http_status == 400


def test_check_dataset_applies_the_prefix_list(write_config):
    write_config({**_enabled(), "allowed_dataset_prefixes": ["/nrs/cellmap/", "s3://bucket/data"]})
    loaded = config.load_config()
    loaded.check_dataset("/nrs/cellmap/jrc_x.zarr/recon-1/em")
    loaded.check_dataset("s3://bucket/data/x.zarr")
    for refused in ("/groups/other/x.zarr", "/nrs/cellmap/../private/x.zarr", "/nrs/cellmapx/y"):
        with pytest.raises(AIAnnotateError) as caught:
            loaded.check_dataset(refused)
        assert (caught.value.category, caught.value.http_status) == ("refused", 403)


def test_a_url_with_dot_segments_is_refused_even_under_an_allowed_prefix(write_config):
    write_config({**_enabled(), "allowed_dataset_prefixes": ["https://host/allowed/"]})
    loaded = config.load_config()
    loaded.check_dataset("https://host/allowed/x.zarr/s0")
    loaded.check_dataset("https://host/allowed/x.v2.zarr?ver=1.0")  # dots inside names are fine
    for refused in (
        "https://host/allowed/../secret/raw.zarr",
        "https://host/allowed/./x.zarr",
        "https://host/allowed/%2e%2e/secret",
        "https://host/allowed/.%2E/secret",
        "https://host/allowed%2f..%2fsecret",
    ):
        with pytest.raises(AIAnnotateError) as caught:
            loaded.check_dataset(refused)
        assert caught.value.category == "refused", refused


def test_an_empty_prefix_list_allows_any_dataset(write_config):
    write_config(_enabled())
    config.load_config().check_dataset("/anything/at/all")
