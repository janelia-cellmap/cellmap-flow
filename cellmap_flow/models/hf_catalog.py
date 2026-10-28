"""The cellmap models published on the Hugging Face Hub, for the dashboard's picker.

``list_huggingface_models`` answers from a cache file under
``~/.cellmap_flow`` once there is one, so opening the dashboard does not
query the Hub each time; ``refresh_huggingface_models`` rebuilds it;
``cached_huggingface_models`` reads the cache alone, for a list that must
not wait on the Hub. Each model is keyed by its repo id, with the
``metadata.json`` it publishes and, beside its fields, what every model list
shows of a model without downloading it (``FACTS``): its weight formats, how
it can be finetuned, the size of its largest weights file, and the copy of it
that runs in the browser, if it has one. Those are read from the repo's file
list and the sizes of its weights (one listing of the organization's repos
with their file names, then one request of sizes per repo), and from the
manifests of cellmap/onnx-web-models, when the cache is built, and kept in it:
a page's poll never asks the Hub for them.
"""

import json
import os
from typing import Any, Dict, List, Optional

from huggingface_hub import get_paths_info, hf_hub_download, list_models, list_repo_files

from cellmap_flow.models.configs.cellmap import export_finetune_modes
from cellmap_flow.models.resolve import _onnx_folders

HUGGING_FACE_ORGS_NAME = "cellmap"
HF_CACHE_DIR = os.path.expanduser("~/.cellmap_flow/hugging_face")
HF_CACHE_FILE = os.path.join(HF_CACHE_DIR, "models_cache.json")

# The repo of the models exported for the browser (the browser's catalog,
# web/src/models), one folder each with its manifest.json and .onnx files.
WEB_MODELS_REPO = "cellmap/onnx-web-models"

# A format a repo's top-level files hold, by its file's ending, as the model
# lists' tags name it: a cellmap_models export's model.ts, model.pt2 and
# model.onnx.
FORMATS = ((".ts", "TorchScript"), (".pt2", "torch.export"), (".onnx", "ONNX"))

# The endings of weight files, the largest of which is a model's size: the
# formats', an ONNX export's external data (model.onnx.data, which holds its
# weights where model.onnx is a few hundred kB), and saved PyTorch weights
# (model.pt, model_state_dict.pt).
WEIGHT_ENDINGS = (".ts", ".pt2", ".onnx", ".onnx.data", ".pt", ".pth", ".safetensors", ".ckpt")

# What the listing adds to each model's metadata, which the model lists read
# (static/js/shared/model-list.js factTags) and the ranking's candidates carry
# (routes/finetune/rank_routes.py), under the names the browser's listing
# gives them too (web/src/models/metadata.ts):
# - ``formats``: the labels of FORMATS the model has;
# - ``finetune_modes``: what the training form offers for it;
# - ``size_mb``: its largest weights file, in MB (1e6 bytes, as the browser
#   manifests' file_mb), or None when no size is known;
# - ``browser_copy``: the id the browser's catalog knows its copy by
#   ("cellmap/onnx-web-models/setup_54", or "<repo>/<folder>" for a copy in
#   the repo's own folder), or None when it has none. Left out when that
#   could not be found out (cellmap/onnx-web-models not read), so a list
#   filtering on it does not hide a model it knows nothing of;
# - ``variants``: the copy's precisions, ``{"variant", "file_mb"}`` each.
FACTS = ("formats", "finetune_modes", "size_mb", "browser_copy", "variants")


def weight_facts(sizes: Dict[str, Optional[int]]) -> Dict[str, Any]:
    """``formats``, ``finetune_modes`` and ``size_mb`` of a model whose
    top-level files are ``sizes`` (each name's size in bytes, None when not
    known).

    ``finetune_modes`` is the training form's for the model
    (``export_finetune_modes``)."""
    weights = [size for name, size in sizes.items() if name.endswith(WEIGHT_ENDINGS) and size]
    return {
        "formats": [label for ending, label in FORMATS if any(name.endswith(ending) for name in sizes)],
        "finetune_modes": list(export_finetune_modes(sizes)),
        "size_mb": round(max(weights) / 1e6, 2) if weights else None,
    }


def folder_facts(folder: str) -> Dict[str, Any]:
    """``FACTS`` of a cellmap_models export's folder (the Local Catalog's
    models, models.yaml), from its files' names and sizes as a repo's are
    read, loading nothing; a browser cannot read a folder of the cluster's,
    so none has a browser copy. Empty when the folder cannot be listed."""
    try:
        with os.scandir(folder) as entries:
            sizes = {entry.name: entry.stat().st_size for entry in entries if entry.is_file()}
    except OSError:
        return {}
    return {**weight_facts(sizes), "browser_copy": None}


def facts_of(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """The ``FACTS`` a listed model's entry has (a cache written before they
    were listed has none)."""
    return {key: metadata[key] for key in FACTS if key in metadata}


def _variants(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    """A browser copy's precisions, one ``{"variant", "file_mb"}`` each in the
    order its manifest lists them, the largest of the blocks' files (each
    block's file of a precision is about the same size)."""
    sizes: Dict[str, float] = {}
    for variant in manifest.get("variants") or []:
        name, mb = variant.get("variant"), variant.get("file_mb")
        if isinstance(name, str) and isinstance(mb, (int, float)):
            sizes[name] = max(sizes.get(name, mb), mb)
    return [{"variant": name, "file_mb": mb} for name, mb in sizes.items()]


def _manifest(repo: str, folder: str) -> Dict[str, Any]:
    with open(hf_hub_download(repo, f"{folder}/manifest.json")) as f:
        return json.load(f)


def _linked_copies() -> Optional[Dict[str, Dict[str, Any]]]:
    """cellmap/onnx-web-models' browser copies by the repo each was exported
    from: ``{repo: {"browser_copy", "variants"}}``. Every manifest there
    names it (``source_repo``: setup_54's is cellmap/mito-aff-resunet-setup-54,
    setup_54_2d's the 2D student's cellmap/mito-aff-resunet-setup-54-2d), so
    nothing is matched by name or setup number. Its top-level folders, as
    the browser lists them (web/src/models/manifest.ts modelFolders, and Add
    a model, models/resolve.py _onnx_folders): conv3d/ repeats its models a
    level down, for runtimes other than the browser's. None when the repo
    cannot be read; a manifest that cannot be is left out."""
    try:
        folders = _onnx_folders(list_repo_files(WEB_MODELS_REPO))
    except Exception as e:
        print(f"{WEB_MODELS_REPO}: could not list its browser copies ({e})")
        return None
    copies = {}
    for folder in folders:
        try:
            manifest = _manifest(WEB_MODELS_REPO, folder)
        except Exception as e:
            print(f"{WEB_MODELS_REPO}/{folder}: could not read its manifest.json ({e})")
            continue
        if isinstance(manifest.get("source_repo"), str):
            copies[manifest["source_repo"]] = {"browser_copy": f"{WEB_MODELS_REPO}/{folder}", "variants": _variants(manifest)}
    return copies


def _repo_facts(repo: str, names: List[str], linked: Optional[Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
    """``FACTS`` of ``repo``, whose files are ``names``: its formats and
    finetune modes from the names of its top-level files, its size from
    theirs (asked for those of its weights alone: one request), and its
    browser copy, cellmap/onnx-web-models' for it (``linked``, _linked_copies;
    None when that could not be read) or else one in a folder of its own (as
    the pruned "-small" copies an export makes beside a finetuned model's
    TorchScript, ackermand/mito_aff_unet_setup_16_finetuned_20261027_003514
    has). A size that cannot be had is None, the rest still read."""
    top = [name for name in names if "/" not in name]
    weights = [name for name in top if name.endswith(WEIGHT_ENDINGS)]
    sizes: Dict[str, Optional[int]] = dict.fromkeys(top)
    try:
        sizes.update({info.path: info.size for info in get_paths_info(repo, weights)} if weights else {})
    except Exception as e:
        print(f"{repo}: could not read its files' sizes ({e})")
    facts = weight_facts(sizes)
    if linked and repo in linked:
        return {**facts, **linked[repo]}
    for folder in _onnx_folders(names):
        try:
            return {**facts, "browser_copy": f"{repo}/{folder}", "variants": _variants(_manifest(repo, folder))}
        except Exception as e:
            print(f"{repo}/{folder}: could not read its manifest.json ({e})")
    return {**facts, "browser_copy": None} if linked is not None else facts


def repo_facts(repo: str, revision: Optional[str] = None) -> Dict[str, Any]:
    """``FACTS`` of one repo, for a model added by reference that the listing
    does not have (another owner's, as David's finetuned exports): from its
    files, and its browser copy only when it holds one in a folder of its
    own, cellmap/onnx-web-models having copies of cellmap's models alone,
    which the listing has. Raises what listing its files raises."""
    return _repo_facts(repo, list_repo_files(repo, revision=revision), {})


def _fetch_huggingface_models(org_name: str = HUGGING_FACE_ORGS_NAME) -> Dict[str, Any]:
    """Fetch models from Hugging Face Hub and save to cache."""
    result = {}
    try:
        # With their file names, which the facts read (and their sizes, which
        # no listing gives, asked for per repo).
        models = list(list_models(author=org_name, expand=["siblings"]))
    except Exception as e:
        print(f"Error fetching Hugging Face models: {str(e)}")
        return {}
    linked = _linked_copies()
    for m in models:
        try:
            path = hf_hub_download(m.id, "metadata.json")
            with open(path) as f:
                metadata = json.load(f)
        except Exception as e:
            print(f"{m.id}: Could not load metadata.json ({e})")
            continue
        result[m.id] = {**metadata, **_repo_facts(m.id, [s.rfilename for s in m.siblings or []], linked)}

    # Save to cache
    os.makedirs(HF_CACHE_DIR, exist_ok=True)
    with open(HF_CACHE_FILE, "w") as f:
        json.dump(result, f)

    return result


def list_huggingface_models(org_name: str = HUGGING_FACE_ORGS_NAME) -> Dict[str, Any]:
    """
    List available Hugging Face models, using cache if available.

    A cache written before the listing gave each model's ``FACTS`` is fetched
    again, once: its rows would show nothing of what each model is.

    Args:
        org_name: Hugging Face organization name to filter models (default: "cellmap")

    Returns:
        A dict mapping model IDs to their metadata
    """
    cached = cached_huggingface_models()
    if cached is not None and all("formats" in metadata for metadata in cached.values()):
        return cached
    return _fetch_huggingface_models(org_name)


def cached_huggingface_models() -> Optional[Dict[str, Any]]:
    """The models as the last fetch cached them, or None when nothing is
    cached (or the cache cannot be read), without asking the Hub: the
    Finetune tab's ranking lists them so (routes/finetune/rank_routes.py),
    and its Refresh is the Models tab's, ``refresh_huggingface_models``."""
    try:
        with open(HF_CACHE_FILE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def refresh_huggingface_models(org_name: str = HUGGING_FACE_ORGS_NAME) -> Dict[str, Any]:
    """Force refresh the Hugging Face models cache."""
    return _fetch_huggingface_models(org_name)
