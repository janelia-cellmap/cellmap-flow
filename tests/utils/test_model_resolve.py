"""``models.resolve``: what each kind of model reference a user pastes resolves
to, and the dashboard's route. Hugging Face and the zoo are fakes: a test
that reaches for the network fails."""

import json
import shutil
import zipfile

import pytest

from cellmap_flow.models import resolve as resolve_module
from cellmap_flow.models.configs.onnx import MANIFEST_KEY
from cellmap_flow.models.resolve import resolve
from tests.utils.onnx_helpers import FINETUNED_MANIFEST, HEAD, with_metadata

ZOO = [
    {"type": "model", "id": "10.5281/zenodo.5764892", "nickname": "affable-shark",
     "concept_doi": "10.5281/zenodo.11092561"},
    {"type": "model", "id": "conscientious-dromedary", "concept_doi": "10.5281/zenodo.15535727"},
]
HUB = {
    "cellmap/mito": ["README.md", "metadata.json", "model.pt", "model.ts"],
    "cellmap/onnx-web-models": ["README.md", "setup_54/manifest.json", "setup_54/setup_54_in148_fp32.onnx",
                                "setup_53_2d/manifest.json"],
}


class Online:
    """What the fake Hub and zoo answer, and what was asked of them."""

    def __init__(self):
        self.hf = {}  # repo -> its files
        self.asked = []

    def files(self, repo, revision):
        self.asked.append(("hf", repo, revision))
        if repo not in self.hf:
            raise LookupError(f"There is no Hugging Face model repo {repo}")
        return self.hf[repo]

    def entries(self):
        self.asked.append(("zoo",))
        return ZOO


@pytest.fixture(autouse=True)
def online(monkeypatch):
    fake = Online()
    monkeypatch.setattr(resolve_module, "_hf_files", fake.files)
    monkeypatch.setattr(resolve_module, "_zoo_entries", fake.entries)
    return fake


def _models_on_disk(tmp):
    """One model of each kind a path can name, under ``tmp``."""
    (tmp / "unet.py").write_text("x = 1\n")
    (tmp / "mito_export").mkdir()
    (tmp / "mito_export" / "metadata.json").write_text("{}")
    (tmp / "mito_export" / "model.ts").write_text("")
    (tmp / "nuclei").mkdir()
    (tmp / "nuclei" / "rdf.yaml").write_text("")
    with zipfile.ZipFile(tmp / "package.zip", "w") as package:
        package.writestr("sub/bioimageio.yaml", "type: model\n")
    (tmp / "run07").mkdir()
    (tmp / "run07" / "model_checkpoint_432000").write_bytes(b"")
    (tmp / "run07" / "train.py").write_text("labels = ['mito', 'er']\nvoxel_size = (8, 8, 8)\n")
    (tmp / "ft_mito" / "lora_adapter").mkdir(parents=True)
    (tmp / "ft_mito" / "lora_adapter" / "adapter_config.json").write_text("{}")
    # torch.save's zip of Cellpose-SAM weights, told by their keys.
    with zipfile.ZipFile(tmp / "my_cpsam", "w") as weights:
        weights.writestr("cpsam8/data.pkl", b"\x80\x02ccollections\nOrderedDict\n"
                                            b"encoder.patch_embed.proj.weight\nencoder.neck.0.weight")
        weights.writestr("cpsam8/data/0", b"\0" * 16)
    # As the browser's Download finetuned model writes it: the manifest in its metadata_props.
    (tmp / "ft1.onnx").write_bytes(with_metadata(HEAD.read_bytes(), {MANIFEST_KEY: json.dumps(FINETUNED_MANIFEST)}))
    shutil.copy(HEAD, tmp / "plain.onnx")


@pytest.mark.parametrize("ref, expected", [
    # Local files and folders.
    pytest.param("{tmp}/unet.py", ("script", {"script_path": "{tmp}/unet.py"}, "unet", []), id="script"),
    pytest.param("{tmp}/mito_export/", ("cellmap", {"folder_path": "{tmp}/mito_export"}, "mito_export", []),
                 id="cellmap-export-folder"),
    pytest.param("{tmp}/nuclei", ("bioimage", {"model": "{tmp}/nuclei/rdf.yaml"}, "nuclei", []),
                 id="bioimage-folder"),
    pytest.param("{tmp}/package.zip", ("bioimage", {"model": "{tmp}/package.zip"}, "package", []),
                 id="bioimage-package"),
    # The run's train.py names its channels and voxel size.
    pytest.param("{tmp}/run07/model_checkpoint_432000",
                 ("fly", {"checkpoint_path": "{tmp}/run07/model_checkpoint_432000"}, "run07_432000", []),
                 id="training-checkpoint"),
    pytest.param("{tmp}/ft_mito/lora_adapter",
                 ("finetune", {"lora_adapter_path": "{tmp}/ft_mito/lora_adapter"}, "ft_mito", ["base_model"]),
                 id="lora-finetune"),
    pytest.param("{tmp}/my_cpsam", ("cellpose", {"pretrained_model": "{tmp}/my_cpsam"}, "my_cpsam", ["voxel_size"]),
                 id="cellpose-sam-weights"),
    pytest.param("{tmp}/ft1.onnx", ("onnx", {"path": "{tmp}/ft1.onnx"}, "ft1", []), id="onnx-with-its-manifest"),
    pytest.param("{tmp}/plain.onnx", ("onnx", {"path": "{tmp}/plain.onnx"}, "plain", ["input_voxel_size"]),
                 id="onnx-without-one"),
    # Names, prefixes and URLs.
    pytest.param("cpsam", ("cellpose", {"pretrained_model": "cpsam"}, "cpsam", ["voxel_size"]), id="cellpose-name"),
    pytest.param("hf:org/repo@v2", ("huggingface", {"repo": "org/repo", "revision": "v2"}, "repo", []),
                 id="hf-prefix"),
    pytest.param("dacapo:my@run@100000", ("dacapo", {"run_name": "my@run", "iteration": 100000}, "my@run_100000", []),
                 id="dacapo-prefix"),
    pytest.param("cellmap/mito", ("huggingface", {"repo": "cellmap/mito"}, "mito", []), id="hub-repo"),
    pytest.param("https://huggingface.co/cellmap/mito/tree/v1",
                 ("huggingface", {"repo": "cellmap/mito", "revision": "v1"}, "mito", []), id="hub-url"),
    pytest.param("affable-shark", ("bioimage", {"model": "affable-shark"}, "affable-shark", []), id="zoo-nickname"),
    pytest.param("10.5281/zenodo.11092561", ("bioimage", {"model": "10.5281/zenodo.11092561"}, "affable-shark", []),
                 id="zoo-concept-doi"),
    pytest.param("https://bioimage.io/#/artifacts/affable-shark",
                 ("bioimage", {"model": "affable-shark"}, "affable-shark", []), id="zoo-url"),
    pytest.param("setup_54", ("onnx", {"repo": "cellmap/onnx-web-models", "folder": "setup_54"}, "setup_54", []),
                 id="onnx-web-export-name"),
    pytest.param("https://huggingface.co/cellmap/onnx-web-models/tree/main/setup_54",
                 ("onnx", {"repo": "cellmap/onnx-web-models", "folder": "setup_54", "revision": "main"},
                  "setup_54", []), id="onnx-web-export-url"),
])
def test_each_kind_of_reference_resolves_to_its_model_type(tmp_path, online, ref, expected):
    """(type, params, name, what it still needs) for what is pasted; ``{tmp}``
    is the folder the models on disk are in."""
    _models_on_disk(tmp_path)
    online.hf.update(HUB)
    resolved = resolve(ref.format(tmp=tmp_path))
    expected = json.loads(json.dumps(expected).replace("{tmp}", str(tmp_path)))
    assert [resolved.type, resolved.params, resolved.name, resolved.needs] == expected


def test_the_route_answers_with_the_resolved_model(dashboard):
    answer = dashboard.post("/api/models/resolve", json={"ref": " cpsam ", "name": "cells", "voxel_size": "8"})
    assert answer.status_code == 200
    body = answer.get_json()
    assert (body["success"], body["type"], body["params"], body["name"], body["env"], body["needs"]) == (
        True, "cellpose", {"pretrained_model": "cpsam", "voxel_size": [8, 8, 8]}, "cells", "cellpose4", [])
    assert dashboard.post("/api/models/resolve", json={"ref": "cpsam"}).get_json()["needs"] == ["voxel_size"]


@pytest.mark.parametrize("body, message", [
    ({}, "ref is required"),
    ({"ref": "google-bert/bert-base-uncased"}, "not a cellmap-models export"),
])
def test_the_route_refuses_what_it_cannot_resolve(dashboard, online, body, message):
    online.hf["google-bert/bert-base-uncased"] = ["config.json"]
    answer = dashboard.post("/api/models/resolve", data=json.dumps(body), content_type="application/json")
    assert answer.status_code == 400
    assert answer.get_json()["success"] is False and message in answer.get_json()["error"]
