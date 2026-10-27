"""cellmap's models on the Hugging Face Hub as the Models tab lists them
(models.hf_catalog, GET /api/huggingface-models): each with what its row
shows of it without downloading it, its formats, finetune modes, size and
browser copy, read from the repo's files and cellmap/onnx-web-models'
manifests when the cache is built. The Hub is faked: no test touches the
network."""

import json
from types import SimpleNamespace

import pytest

from cellmap_flow.models import hf_catalog

MB = 1_000_000

# The org's repos as the Hub lists them, with their files' sizes: a ResUNet
# export with every format, and cellmap/onnx-web-models' copy of it; an
# export with no model.pt2 (TorchScript and ONNX, as jrc_mus-livers'); and a
# finetuned export with the pruned browser copy of it in a folder of its own.
FILES = {
    "cellmap/mito-aff-resunet-setup-54": {
        "metadata.json": 1352, "model.pt2": 56 * MB, "model.ts": 43 * MB, "model.onnx": 137_637, "model.onnx.data": 43_106_304,
        "model_state_dict.pt": 42_832_493, "README.md": 2963,
    },
    "cellmap/jrc_mus-livers_16nm_to_8nm_mito": {"metadata.json": 1352, "model.ts": 900 * MB, "model.onnx": 947_270_000},
    "cellmap/mito_aff_unet_setup_16_ft": {
        "metadata.json": 1192, "model.ts": 3_169_733_315, "ft-small/manifest.json": 6976, "ft-small/ft-small_in178_fp32.onnx": 43_368_144,
    },
    hf_catalog.WEB_MODELS_REPO: {
        "README.md": 27583, "setup_54/manifest.json": 7339, "setup_54/setup_54_in132_fp32.onnx": 42_914_187,
        # A level down: the browser lists top-level folders only.
        "conv3d/setup_54/manifest.json": 7339, "conv3d/setup_54/setup_54_in116_fp32.onnx": 42_914_184,
    },
}

MANIFESTS = {
    (hf_catalog.WEB_MODELS_REPO, "setup_54"): {"source_repo": "cellmap/mito-aff-resunet-setup-54", "variants": [
        {"variant": "fp32", "input_size": 148, "file_mb": 42.91}, {"variant": "fp16", "input_size": 148, "file_mb": 21.55},
        {"variant": "fp32", "input_size": 116, "file_mb": 42.9}]},
    (hf_catalog.WEB_MODELS_REPO, "conv3d/setup_54"): {"source_repo": "cellmap/jrc_mus-livers_16nm_to_8nm_mito", "variants": []},
    # Its source is the base it was finetuned from, not this repo.
    ("cellmap/mito_aff_unet_setup_16_ft", "ft-small"): {"source_repo": "cellmap/mito-aff-unet-setup-16", "variants": [
        {"variant": "fp32", "input_size": 178, "file_mb": 43.37}, {"variant": "int8w", "input_size": 178, "file_mb": 11.0}]},
}


@pytest.fixture
def hub(tmp_path, monkeypatch):
    """The fake Hub, and the cache under tmp_path; ``hub.asked`` counts the
    listings of the org."""
    state = SimpleNamespace(asked=0)

    def listing(author, expand=None):
        state.asked += 1
        assert expand == ["siblings"]
        return [SimpleNamespace(id=repo, siblings=[SimpleNamespace(rfilename=name) for name in files])
                for repo, files in FILES.items()]

    def download(repo, filename):
        folder, _, name = filename.rpartition("/")
        content = {"description": repo} if name == "metadata.json" else MANIFESTS[(repo, folder)]
        if name == "metadata.json" and name not in FILES[repo]:
            raise OSError("404 Entry Not Found")
        path = tmp_path / "downloads" / repo / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content))
        return str(path)

    monkeypatch.setattr(hf_catalog, "list_models", listing)
    monkeypatch.setattr(hf_catalog, "list_repo_files", lambda repo, revision=None: list(FILES[repo]))
    monkeypatch.setattr(hf_catalog, "get_paths_info", lambda repo, paths: [
        SimpleNamespace(path=path, size=FILES[repo][path]) for path in paths])
    monkeypatch.setattr(hf_catalog, "hf_hub_download", download)
    monkeypatch.setattr(hf_catalog, "HF_CACHE_DIR", str(tmp_path / "hugging_face"))
    monkeypatch.setattr(hf_catalog, "HF_CACHE_FILE", str(tmp_path / "hugging_face" / "models_cache.json"))
    return state


def test_each_model_is_listed_with_its_formats_size_and_browser_copy(dashboard, hub):
    models = dashboard.get("/api/huggingface-models").get_json()

    # onnx-web-models has no metadata.json: it is not a model, but its
    # manifests say which models it has a copy of.
    assert sorted(models) == sorted(list(FILES)[:3])
    facts = {repo: hf_catalog.facts_of(meta) for repo, meta in models.items()}
    assert facts["cellmap/mito-aff-resunet-setup-54"] == {
        "formats": ["TorchScript", "torch.export", "ONNX"], "finetune_modes": ["lora", "full"], "size_mb": 56.0,
        "browser_copy": "cellmap/onnx-web-models/setup_54",
        "variants": [{"variant": "fp32", "file_mb": 42.91}, {"variant": "fp16", "file_mb": 21.55}]}
    # No model.pt2: full finetuning only, as the training form offers; and no
    # browser copy, the conv3d/ manifest that names it not being one.
    assert facts["cellmap/jrc_mus-livers_16nm_to_8nm_mito"] == {
        "formats": ["TorchScript", "ONNX"], "finetune_modes": ["full"], "size_mb": 947.27, "browser_copy": None}
    # Its own pruned copy, whatever its manifest says it came from.
    assert facts["cellmap/mito_aff_unet_setup_16_ft"] == {
        "formats": ["TorchScript"], "finetune_modes": ["full"], "size_mb": 3169.73,
        "browser_copy": "cellmap/mito_aff_unet_setup_16_ft/ft-small",
        "variants": [{"variant": "fp32", "file_mb": 43.37}, {"variant": "int8w", "file_mb": 11.0}]}
    assert models["cellmap/mito-aff-resunet-setup-54"]["description"] == "cellmap/mito-aff-resunet-setup-54"

    # Cached: listed again without asking the Hub.
    dashboard.get("/api/huggingface-models")
    assert hub.asked == 1

    # A cache written before the listing said what each model is is fetched
    # again, once.
    with open(hf_catalog.HF_CACHE_FILE, "w") as f:
        json.dump({"cellmap/mito-aff-resunet-setup-54": {"description": "old"}}, f)
    assert dashboard.get("/api/huggingface-models").get_json() == models
    dashboard.get("/api/huggingface-models")
    assert hub.asked == 2
