"""``models.resolve``: each check of its order, the prefixes, what a model still
needs, the bioimage arguments from either constructor, and the dashboard's
route. Hugging Face and the zoo are fakes: a test that reaches for the
network fails."""

import json
import zipfile

import pytest

from cellmap_flow.models import registry, resolve as resolve_module
from cellmap_flow.models.resolve import bioimage_params, resolve

ZOO = [
    {"type": "model", "id": "10.5281/zenodo.5764892", "nickname": "affable-shark",
     "concept_doi": "10.5281/zenodo.11092561"},
    {"type": "model", "id": "conscientious-dromedary", "concept_doi": "10.5281/zenodo.15535727"},
]
EXPORT = ["README.md", "metadata.json", "model.pt", "model.ts"]


class Online:
    """What the fake Hub and zoo answer, and what was asked of them."""

    def __init__(self):
        self.hf = {}  # repo -> its files, or an exception to raise
        self.zoo = ZOO
        self.asked = []

    def files(self, repo, revision):
        self.asked.append(("hf", repo, revision))
        answer = self.hf.get(repo, LookupError(f"There is no Hugging Face model repo {repo}"))
        if isinstance(answer, Exception):
            raise answer
        return answer

    def entries(self):
        self.asked.append(("zoo",))
        if isinstance(self.zoo, Exception):
            raise self.zoo
        return self.zoo


@pytest.fixture(autouse=True)
def online(monkeypatch):
    fake = Online()
    monkeypatch.setattr(resolve_module, "_hf_files", fake.files)
    monkeypatch.setattr(resolve_module, "_zoo_entries", fake.entries)
    return fake


def _summary(resolved):
    return resolved.type, resolved.params, resolved.name, resolved.needs


# --- 1. local files and folders -------------------------------------------------------

def test_a_script_is_a_script_model(tmp_path):
    (tmp_path / "unet.py").write_text("x = 1\n")
    assert _summary(resolve(str(tmp_path / "unet.py"))) == (
        "script", {"script_path": str(tmp_path / "unet.py")}, "unet", [])


def test_an_export_folder_is_a_cellmap_model_and_needs_its_model_ts(tmp_path):
    export = tmp_path / "mito_export"
    export.mkdir()
    (export / "metadata.json").write_text("{}")
    with pytest.raises(ValueError, match="no model.ts.*fly:"):
        resolve(str(export))
    (export / "model.ts").write_text("")
    resolved = resolve(str(export) + "/")
    assert _summary(resolved) == ("cellmap", {"folder_path": str(export)}, "mito_export", [])
    assert (resolved.env, resolved.how) == (None, "a cellmap-models export folder")


def _rdf_zip(path, inner="rdf.yaml"):
    with zipfile.ZipFile(path, "w") as package:
        package.writestr(inner, "type: model\n")
    return str(path)


def _bioimage_on_disk(tmp_path, kind):
    """(what is pasted, what bioimageio.core is to load, the name) for each kind on disk."""
    if kind == "folder":
        (tmp_path / "nuclei").mkdir()
        (tmp_path / "nuclei" / "rdf.yaml").write_text("")
        return tmp_path / "nuclei", tmp_path / "nuclei" / "rdf.yaml", "nuclei"
    if kind == "rdf-file":
        (tmp_path / "boundaries.rdf.yaml").write_text("")
        return tmp_path / "boundaries.rdf.yaml", tmp_path / "boundaries.rdf.yaml", "boundaries"
    _rdf_zip(tmp_path / "package.zip", "sub/bioimageio.yaml")
    return tmp_path / "package.zip", tmp_path / "package.zip", "package"


@pytest.mark.parametrize("kind", ["folder", "rdf-file", "zip"])
def test_a_bioimage_description_or_package_is_a_bioimage_model(tmp_path, kind):
    pasted, model, name = _bioimage_on_disk(tmp_path, kind)
    resolved = resolve(str(pasted), voxel_size="8")
    assert _summary(resolved) == ("bioimage", {"model": str(model), "voxel_size": [8, 8, 8]}, name, [])
    assert resolved.env == "bioimageio"


def test_a_zip_without_a_description_is_refused(tmp_path):
    _rdf_zip(tmp_path / "x.zip", "readme.txt")
    with pytest.raises(ValueError, match="without an rdf.yaml"):
        resolve(str(tmp_path / "x.zip"))


def _run(tmp_path, train_py=None):
    run = tmp_path / "run07"
    run.mkdir()
    (run / "model_checkpoint_432000").write_bytes(b"")
    if train_py:
        (run / "train.py").write_text(train_py)
    return run


def test_a_training_checkpoint_is_a_fly_model_that_reads_its_run(tmp_path):
    run = _run(tmp_path, "labels = ['mito', 'er']\nvoxel_size = (8, 8, 8)\n")
    resolved = resolve(str(run / "model_checkpoint_432000"))
    assert _summary(resolved) == (
        "fly", {"checkpoint_path": str(run / "model_checkpoint_432000")}, "run07_432000", [])
    assert resolved.env == "fly"
    assert resolved.notes == ["reads channels mito, er and voxel size [8, 8, 8] from the files beside the checkpoint"]
    # The entry builds as it is: the type reads the run again.
    model = registry.build_model(resolved.entry(), resolved.name)
    assert (model.channels, model.name) == (["mito", "er"], "run07_432000")


@pytest.mark.parametrize("voxel_size, needs", [(None, ["channels", "input_voxel_size"]), (4, ["channels"])])
def test_a_fly_model_whose_run_says_nothing_needs_what_it_lacks(tmp_path, voxel_size, needs):
    run = _run(tmp_path)
    resolved = resolve(str(run / "model_checkpoint_432000"), voxel_size=voxel_size)
    assert resolved.needs == needs
    assert resolved.params.get("input_voxel_size") == ([4, 4, 4] if voxel_size else None)


@pytest.mark.parametrize("file, name, env", [("mito.ts", "mito", None), ("model.pt", "exported", "fly")])
def test_a_torchscript_or_pickled_model_is_a_fly_model(tmp_path, file, name, env):
    folder = tmp_path / "exported"
    folder.mkdir()
    (folder / file).write_bytes(b"")
    resolved = resolve(str(folder / file), voxel_size=8)
    assert (resolved.type, resolved.name, resolved.env, resolved.needs) == ("fly", name, env, ["channels"])


def test_a_folder_of_checkpoints_points_at_the_newest(tmp_path):
    run = _run(tmp_path)
    (run / "model_checkpoint_9000").write_bytes(b"")
    with pytest.raises(ValueError, match="model_checkpoint_432000"):
        resolve(str(run))


@pytest.mark.parametrize("parts, key, name", [
    (("ft_mito", "lora_adapter", "adapter_config.json"), "lora_adapter_path", "ft_mito"),
    (("ft_mito", "full_finetune", "model_state_dict.pt"), "weights_path", "ft_mito"),
], ids=["lora", "full"])
def test_a_finetune_needs_its_base_model(tmp_path, parts, key, name):
    path = tmp_path.joinpath(*parts)
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    target = path.parent if key == "lora_adapter_path" else path
    resolved = resolve(str(target))
    assert _summary(resolved) == ("finetune", {key: str(target)}, name, ["base_model"])
    # Its env is its base model's, which is not known yet.
    assert resolved.env is None


def _torch_zip(path, keys):
    with zipfile.ZipFile(path, "w") as weights:
        weights.writestr("cpsam8/data.pkl", b"\x80\x02ccollections\nOrderedDict\n" + b"\n".join(keys))
        weights.writestr("cpsam8/data/0", b"\0" * 16)
    return str(path)


def test_cellpose_sam_weights_are_told_by_their_keys(tmp_path):
    weights = _torch_zip(tmp_path / "my_cpsam", [b"encoder.patch_embed.proj.weight", b"encoder.neck.0.weight"])
    resolved = resolve(weights)
    assert _summary(resolved) == ("cellpose", {"pretrained_model": weights}, "my_cpsam", ["voxel_size"])
    assert resolved.env == "cellpose4"
    other = _torch_zip(tmp_path / "other", [b"downsample.down.res_down_0.weight"])
    with pytest.raises(ValueError, match="fly:.*cellpose:"):
        resolve(other)


# --- 2. prefixes ------------------------------------------------------------------------

def test_the_prefixes_say_what_a_reference_is_without_looking_it_up(tmp_path, online):
    (tmp_path / "unet.py").write_text("")
    run = _run(tmp_path, "labels = ['mito']\nvoxel_size = 8\n")
    cases = {
        "hf:org/repo@v2": ("huggingface", {"repo": "org/repo", "revision": "v2"}, "repo", []),
        "huggingface:org/repo": ("huggingface", {"repo": "org/repo"}, "repo", []),
        "bioimageio:affable-shark": ("bioimage", {"model": "affable-shark"}, "affable-shark", []),
        "bioimage:https://bioimage.io/#/artifacts/affable-shark":
            ("bioimage", {"model": "affable-shark"}, "affable-shark", []),
        "cellpose:my_user_model": ("cellpose", {"pretrained_model": "my_user_model"}, "my_user_model", ["voxel_size"]),
        f"fly:{run}/model_checkpoint_432000":
            ("fly", {"checkpoint_path": f"{run}/model_checkpoint_432000"}, "run07_432000", []),
        "dacapo:my@run@100000": ("dacapo", {"run_name": "my@run", "iteration": 100000}, "my@run_100000", []),
        "dacapo:my_run": ("dacapo", {"run_name": "my_run"}, "my_run", ["iteration"]),
        f"SCRIPT:{tmp_path}/unet.py": ("script", {"script_path": f"{tmp_path}/unet.py"}, "unet", []),
    }
    for ref, expected in cases.items():
        assert _summary(resolve(ref)) == expected, ref
    assert online.asked == []
    for ref, message in [("fly:/no/such/file", "there is no /no/such/file"), ("dacapo:run@last", "whole number"),
                         ("hf:", "needs something"), (f"bioimageio:{tmp_path}/unet.py", "not a bioimage.io model")]:
        with pytest.raises(ValueError, match=message):
            resolve(ref)


# --- 3. Cellpose names -------------------------------------------------------------------

@pytest.mark.parametrize("name", ["cpsam_v2", "cpsam", "cpdino", "cpdino-vitb"])
def test_a_cellpose_model_name_is_a_cellpose_model(name):
    resolved = resolve(name, voxel_size="16,8,8")
    assert _summary(resolved) == ("cellpose", {"pretrained_model": name, "voxel_size": [16, 8, 8]}, name, [])
    assert resolve(name).needs == ["voxel_size"]


# --- 4. URLs ----------------------------------------------------------------------------

@pytest.mark.parametrize("url, model", [
    ("https://bioimage.io/#/?id=affable-shark", "affable-shark"),
    ("https://bioimage.io/#/artifacts/affable-shark", "affable-shark"),
    ("https://zenodo.org/records/5764892", "10.5281/zenodo.5764892"),
    ("https://doi.org/10.5281/zenodo.5764892", "10.5281/zenodo.5764892"),
    ("https://hypha.aicell.io/bioimage-io/artifacts/affable-shark", "affable-shark"),
    ("https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/affable-shark/1.2/files/rdf.yaml",
     "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/affable-shark/1.2/files/rdf.yaml"),
])
def test_a_bioimage_url_is_a_bioimage_model(url, model):
    resolved = resolve(url)
    assert (resolved.type, resolved.params, resolved.needs) == ("bioimage", {"model": model}, [])
    if url.endswith("rdf.yaml"):
        assert resolved.name == "affable-shark"


def test_a_hugging_face_url_is_its_repo(online):
    online.hf["cellmap/mito"] = EXPORT
    resolved = resolve("https://huggingface.co/cellmap/mito/tree/v1")
    assert _summary(resolved) == ("huggingface", {"repo": "cellmap/mito", "revision": "v1"}, "mito", [])
    assert online.asked == [("hf", "cellmap/mito", "v1")]
    for url in ("https://huggingface.co/datasets/org/data", "https://example.com/model"):
        with pytest.raises(ValueError):
            resolve(url)


# --- 5. org/repo --------------------------------------------------------------------------

def test_a_repo_is_checked_on_hugging_face(online):
    online.hf.update({
        "cellmap/mito": EXPORT,
        "google-bert/bert-base-uncased": ["config.json", "model.safetensors"],
        "cellmap/half": ["metadata.json", "model.onnx"],
        "cellmap/away": ConnectionError("Hub unreachable"),
    })
    resolved = resolve("cellmap/mito", voxel_size=8)
    assert _summary(resolved) == ("huggingface", {"repo": "cellmap/mito"}, "mito", [])
    assert (resolved.how, resolved.notes) == (
        "a cellmap-models export on Hugging Face", ["voxel_size is not used: a huggingface model says its own"])
    with pytest.raises(ValueError, match="not a cellmap-models export.*script"):
        resolve("google-bert/bert-base-uncased")
    with pytest.raises(ValueError, match="no model.ts"):
        resolve("cellmap/half")
    with pytest.raises(ValueError, match="There is no Hugging Face model repo cellmap/gone"):
        resolve("cellmap/gone")
    assert "Hub unreachable" in resolve("cellmap/away").notes[0]


def test_offline_a_repo_is_taken_on_trust(online):
    resolved = resolve("cellmap/mito", online=False)
    assert (resolved.type, resolved.notes) == ("huggingface", ["not checked (offline): taken to be a cellmap-models export"])
    assert online.asked == []


def test_a_relative_path_that_is_not_there_is_not_a_repo(online):
    with pytest.raises(ValueError, match="there is none at .*runs/model.ts"):
        resolve("runs/model.ts")
    assert online.asked == []


# --- 6. zoo ids and nicknames ----------------------------------------------------------------

@pytest.mark.parametrize("ref", ["affable-shark", "10.5281/zenodo.5764892", "10.5281/zenodo.11092561"])
def test_a_zoo_id_nickname_or_doi_is_looked_up(ref):
    resolved = resolve(ref, voxel_size=[8])
    assert _summary(resolved) == ("bioimage", {"model": ref, "voxel_size": [8, 8, 8]}, "affable-shark", [])


def test_what_is_not_in_the_zoo_is_refused_and_an_unreachable_zoo_assumed(online):
    with pytest.raises(ValueError, match="not a model in the BioImage Model Zoo"):
        resolve("no-such-model")
    with pytest.raises(ValueError, match="Could not tell what model 'unet'"):
        resolve("unet")
    online.zoo = OSError("zoo unreachable")
    assert "zoo unreachable" in resolve("no-such-model").notes[0]
    with pytest.raises(ValueError, match="Could not tell"):
        resolve("unet")


def test_offline_only_a_zoo_shaped_name_is_taken_on_trust(online):
    assert resolve("conscientious-dromedary", online=False).type == "bioimage"
    with pytest.raises(ValueError, match="offline: not looked up"):
        resolve("unet", online=False)
    assert online.asked == []


# --- what is given -----------------------------------------------------------------------------

def test_a_name_and_voxel_size_are_given_as_asked(tmp_path):
    resolved = resolve("cpsam", name="cells", voxel_size="(5.24, 4, 4)")
    assert resolved.entry() == {"type": "cellpose", "pretrained_model": "cpsam", "voxel_size": [5.24, 4, 4],
                                "name": "cells"}
    for bad in ("a,b", [1, 2], 0):
        with pytest.raises(ValueError, match="voxel_size"):
            resolve("cpsam", voxel_size=bad)
    with pytest.raises(ValueError, match="Give a model reference"):
        resolve("  ")


class OldBio:
    def __init__(self, model_name: str, voxel_size, edge_length_to_process=None, name=None, scale=None):
        pass


class NewBio:
    def __init__(self, model: str, voxel_size=None, name=None, scale=None):
        pass


class NoVoxelBio:
    def __init__(self, model: str, name=None):
        pass


@pytest.mark.parametrize("cls, voxel_size, expected", [
    (OldBio, None, ({"model_name": "m"}, ["voxel_size"])),
    (OldBio, [8, 8, 8], ({"model_name": "m", "voxel_size": [8, 8, 8]}, [])),
    (NewBio, None, ({"model": "m"}, [])),
    (NewBio, [8, 8, 8], ({"model": "m", "voxel_size": [8, 8, 8]}, [])),
    (NoVoxelBio, [8, 8, 8], ({"model": "m"}, [])),
])
def test_the_bioimage_arguments_follow_its_constructor(cls, voxel_size, expected):
    assert bioimage_params("m", voxel_size, cls=cls) == expected


def test_the_bioimage_type_is_asked_for_its_signature(monkeypatch):
    real = registry.model_type
    monkeypatch.setattr(registry, "model_type", lambda name: NewBio if name == "bioimage" else real(name))
    assert _summary(resolve("affable-shark")) == ("bioimage", {"model": "affable-shark"}, "affable-shark", [])


# --- the dashboard's route -----------------------------------------------------------------------

def test_the_route_answers_with_the_resolved_model(dashboard):
    answer = dashboard.post("/api/models/resolve", json={"ref": " cpsam ", "name": "cells", "voxel_size": "8"})
    assert answer.status_code == 200
    assert answer.get_json() == {
        "success": True, "type": "cellpose", "params": {"pretrained_model": "cpsam", "voxel_size": [8, 8, 8]},
        "name": "cells", "env": "cellpose4", "how": "a Cellpose 4 pretrained model", "needs": [], "notes": [],
    }
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
