"""cellmap's models on the Hugging Face Hub as the Models tab lists them
(models.hf_catalog, GET /api/huggingface-models): each with what its row
shows of it without downloading it, its formats, finetune modes, size and
browser copy, read from the repo's files and cellmap/onnx-web-models'
manifests when the cache is built. The Hub is faked: no test touches the
network."""

import json
import os
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from cellmap_flow.models import hf_catalog
from tests.utils.mia_helpers import PRETRAINING, SEGMENTATION, install_fake_adapter, model_metadata

MB = 1_000_000
# The organizations listed, each one listing of its repos.
ORGS = len(hf_catalog.HUGGING_FACE_ORGS)

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
    # Models trained with mia-train: a segmentation model, served as type
    # mia, and a pretraining checkpoint, which is not listed.
    "ai-hhmi/lmd-vitl-p8-cellmap-lvd": {"README.md": 4000, "config.json": 9000, "model.safetensors": 1214 * MB},
    "ai-hhmi/simmim-vitl": {"config.json": 900, "model.safetensors": 1214 * MB},
}
CONFIGS = {"ai-hhmi/lmd-vitl-p8-cellmap-lvd": SEGMENTATION, "ai-hhmi/simmim-vitl": PRETRAINING}
# When the Hub says each repo last changed; a repo it gives no date for has none.
MODIFIED = {"cellmap/mito-aff-resunet-setup-54": datetime(2026, 10, 17, 19, 24, 6, tzinfo=timezone.utc)}
# The mia-train model's card, as the MIA team's open: its front matter, its
# title, then what it is.
CARD = """---
license: other
---

# lmd-vitl-p8-cellmap-lvd

Organelle segmentation model for volumetric EM data. The encoder is
[ai-hhmi/lmd-vitl-p8-simmim-lvd](https://huggingface.co/ai-hhmi/lmd-vitl-p8-simmim-lvd), **fine-tuned**.

| Model | Starts from |
"""

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
    listings of an org, the orgs in ``hub.failing`` cannot be listed, and
    ``hub.planned`` is what mia_metadata was asked of."""
    from cellmap_flow.models.configs import mia

    state = SimpleNamespace(asked=0, failing=set(), planned=[])
    state.adapter = install_fake_adapter(monkeypatch)
    monkeypatch.setenv(mia.MIA_CACHE_ENV, str(tmp_path / "mia_exports"))  # none of its exports made

    def planned(source, revision=None, classes=None):
        state.planned.append(source)
        return model_metadata(**state.adapter.declared)

    state.adapter.mia_metadata = planned

    def listing(author, expand=None):
        state.asked += 1
        assert expand == ["siblings", "lastModified"]
        if author in state.failing:
            raise OSError("429 Too Many Requests")
        return [SimpleNamespace(id=repo, siblings=[SimpleNamespace(rfilename=name) for name in files], last_modified=MODIFIED.get(repo))
                for repo, files in FILES.items() if repo.startswith(f"{author}/")]

    def download(repo, filename):
        folder, _, name = filename.rpartition("/")
        if name in ("metadata.json", "config.json") and name not in FILES[repo]:
            raise OSError("404 Entry Not Found")
        if name == "README.md":
            path = tmp_path / "downloads" / repo / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(CARD)
            return str(path)
        if name == "metadata.json":
            content = {"description": repo}
        elif name == "config.json":
            content = CONFIGS[repo]
        else:
            content = MANIFESTS[(repo, folder)]
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
    monkeypatch.setattr(hf_catalog, "HF_CACHE_FILE", os.path.join(hf_catalog.HF_CACHE_DIR, os.path.basename(hf_catalog.HF_CACHE_FILE)))
    return state


def test_each_model_is_listed_with_its_formats_size_and_browser_copy(dashboard, hub, tmp_path):
    models = dashboard.get("/api/huggingface-models").get_json()

    # onnx-web-models has no metadata.json: it is not a model, but its
    # manifests say which models it has a copy of. The pretraining
    # checkpoint is no model cellmap-flow serves.
    assert sorted(models) == sorted([*list(FILES)[:3], "ai-hhmi/lmd-vitl-p8-cellmap-lvd"])
    facts = {repo: hf_catalog.facts_of(meta) for repo, meta in models.items()}
    assert facts["cellmap/mito-aff-resunet-setup-54"] == {
        "formats": ["TorchScript", "torch.export", "ONNX"], "finetune_modes": ["lora", "full"], "size_mb": 56.0,
        "browser_copy": "cellmap/onnx-web-models/setup_54",
        "variants": [{"variant": "fp32", "file_mb": 42.91}, {"variant": "fp16", "file_mb": 21.55}],
        "last_modified": "2026-10-17T19:24:06Z"}
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
    # The mia-train model is listed to be served as type mia, which converts
    # it; its uint8 output, which its config says its export will have,
    # cannot be finetuned.
    mia = models["ai-hhmi/lmd-vitl-p8-cellmap-lvd"]
    assert hf_catalog.listed_entry("ai-hhmi/lmd-vitl-p8-cellmap-lvd", mia, "lmd")["type"] == "mia"
    assert hf_catalog.facts_of(mia) == {"formats": [], "finetune_modes": [], "size_mb": 1214.0, "browser_copy": None}
    # Its row says what its card says it is, as plain text, and what it is served as.
    assert mia["description"].startswith(
        "Organelle segmentation model for volumetric EM data. The encoder is ai-hhmi/lmd-vitl-p8-simmim-lvd, fine-tuned. "
        "Served here as 2 channels: mito, er.")
    # What its export will declare is read from the config.json downloaded
    # to tell what it is, in its folder, not asked of the repo again (which
    # lists its snapshot and downloads its model card).
    assert hub.planned == [str(tmp_path / "downloads" / "ai-hhmi/lmd-vitl-p8-cellmap-lvd")]

    # Cached: listed again without asking the Hub.
    dashboard.get("/api/huggingface-models")
    assert hub.asked == ORGS

    # When what its export will declare cannot be read, how it can be
    # finetuned is left out, not cached as what its config answers without it.
    def unreadable(source, revision=None, classes=None):
        raise ValueError("config.json gives neither input.training_crop nor img_size")

    hub.adapter.mia_metadata = unreadable
    mia = dashboard.post("/api/huggingface-models/refresh").get_json()["ai-hhmi/lmd-vitl-p8-cellmap-lvd"]
    assert "finetune_modes" not in mia


def test_an_organization_that_cannot_be_listed_leaves_the_others_listed_and_is_listed_again_next_time(dashboard, hub):
    # An older cellmap-flow's cache in the same home, of cellmap's models
    # alone, lists every entry as a huggingface model: neither read nor
    # written here.
    older = os.path.join(hf_catalog.HF_CACHE_DIR, "models_cache.json")
    os.makedirs(hf_catalog.HF_CACHE_DIR)
    with open(older, "w") as f:
        json.dump({"cellmap/mito-aff-resunet-setup-54": {"formats": []}}, f)

    hub.failing = {"ai-hhmi"}
    models = dashboard.get("/api/huggingface-models").get_json()
    assert sorted(models) == sorted(list(FILES)[:3])

    # Its listing was not cached as whole: the next listing asks for it
    # again, and for it alone.
    hub.failing = set()
    models = dashboard.get("/api/huggingface-models").get_json()
    assert "ai-hhmi/lmd-vitl-p8-cellmap-lvd" in models
    assert hub.asked == ORGS + 1

    # A Refresh that cannot list it keeps what was listed of it.
    hub.failing = {"ai-hhmi"}
    assert dashboard.post("/api/huggingface-models/refresh").get_json() == models
    with open(older) as f:
        assert json.load(f) == {"cellmap/mito-aff-resunet-setup-54": {"formats": []}}
