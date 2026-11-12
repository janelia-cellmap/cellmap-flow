"""The models on the Hugging Face Hub that cellmap-flow serves, for the dashboard's picker.

Those of the organizations in ``HUGGING_FACE_ORGS`` that cellmap-flow can
serve: cellmap-models exports (a ``metadata.json`` at the root), and models
trained with mia-train whose ``config.json`` cellmap_models can serve
(``resolve.servable_mia_config``), which are served as type ``mia``
(``LISTED_TYPE``); any other repo is left out.

``list_huggingface_models`` answers from a cache file under
``~/.cellmap_flow`` once it holds every organization's listing, so opening
the dashboard does not query the Hub each time; ``refresh_huggingface_models``
rebuilds it; ``cached_huggingface_models`` reads the cache alone, for a list
that must not wait on the Hub. Each organization is listed on its own, and
the cache says which it holds the listing of (``LISTED_ORGS``): one not
listed yet, or whose listing failed, is asked for again by the next listing,
the others answered from the cache. Each model is keyed by its repo id, with the
``metadata.json`` it publishes and, beside its fields, what every model list
shows of a model without downloading it (``FACTS``): its weight formats, how
it can be finetuned, the size of its largest weights file, and the copy of it
that runs in the browser, if it has one. Those are read from the repo's file
list and the sizes of its weights (one listing of the organization's repos
with their file names, then one request of sizes per repo), and from the
manifests of cellmap/onnx-web-models, when the cache is built, and kept in it:
a page's poll never asks the Hub for them. A mia-train model has no
metadata.json; its ``config.json`` is read instead, once per such repo when
the cache is built, with its model card for its description, and how it
can be finetuned is read from those same downloads (``_mia_plan``).
"""

import json
import os
import re
from datetime import timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from huggingface_hub import get_paths_info, hf_hub_download, list_models, list_repo_files

from cellmap_flow.models.configs.cellmap import export_finetune_modes, integer_output_refusal
from cellmap_flow.models.resolve import EXPORT_METADATA, _mia_files, _onnx_folders, servable_mia_config

# The Hugging Face organizations the list shows the models of: cellmap's
# exports, and the MIA team's (HHMI AI) models trained with mia-train.
HUGGING_FACE_ORGS = ("cellmap", "ai-hhmi")
HF_CACHE_DIR = os.path.expanduser("~/.cellmap_flow/hugging_face")
# The cache: every model listed, by its repo id, and under LISTED_ORGS the
# organizations whose listing it holds. Not the models_cache.json beside it,
# which a cellmap-flow from before ai-hhmi was listed reads and writes in the
# same home: that one caches cellmap's models alone, which here would leave
# the ai-hhmi group empty until a Refresh, and lists whatever its cache holds
# as type huggingface, so ai-hhmi's repos there would be listed where they
# cannot run. Each version keeps its own, and neither reads the other's.
HF_CACHE_FILE = os.path.join(HF_CACHE_DIR, "listed_models.json")
# Where the cache says which organizations it holds the listing of, beside
# the repo ids, none of which it can be: each is "<org>/<name>".
LISTED_ORGS = "listed_orgs"

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
# - ``variants``: the copy's precisions, ``{"variant", "file_mb"}`` each;
# - ``last_modified``: when the repo last changed on the Hub (its
#   lastModified, as "2026-11-11T17:12:49Z"), which Sort by newest orders by;
#   left out when not known (a cache written before it was listed).
FACTS = ("formats", "finetune_modes", "size_mb", "browser_copy", "variants", "last_modified")

# The model type a listed repo is served as, when it is not ``huggingface``:
# ``mia`` for a model trained with mia-train. In the listing beside the
# metadata's fields, under a name no metadata.json has; ``listed_entry``
# reads it.
LISTED_TYPE = "cellmap_flow_type"
MIA_DESCRIPTION = ("Trained with mia-train; converted into a cellmap-models export on its first start, "
                   "on its server's GPU, and served on a server only")
# After a mia-train model's own description, the first paragraph of its
# model card (_card_summary), which says what it predicts and how it was
# trained: what tells the MIA team's repos apart in the list, as their
# names alone do not.
MIA_NOTE = "(trained with mia-train; runs on a GPU server only, converted on its first start)"


def listed_entry(repo: str, metadata: Optional[Dict[str, Any]], name: str) -> Dict[str, Any]:
    """The model entry of ``repo`` as the list has it (``metadata``, its
    entry; None when the list has none): ``huggingface``, or the type it is
    listed as (``LISTED_TYPE``). The Models tab's Submit and the ranking's
    candidates build each listed model from it."""
    return {"type": (metadata or {}).get(LISTED_TYPE) or "huggingface", "repo": repo, "name": name}


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


def _listed_metadata(repo: str, names: List[str]) -> Optional[Dict[str, Any]]:
    """What the list keeps of ``repo``, whose files are ``names``, before its
    facts: a cellmap-models export's metadata.json; for a model trained with
    mia-train that cellmap-flow can serve, its type, a description (the
    first paragraph of its model card, then the channels it is served as) and,
    when that is known, how it can be finetuned (``_mia_plan``), which its
    files cannot say. None for any other repo. Raises what downloading a
    file raises."""
    if EXPORT_METADATA in names or not names:
        with open(hf_hub_download(repo, EXPORT_METADATA)) as f:
            return json.load(f)
    if _mia_files(names):
        path = hf_hub_download(repo, "config.json")
        with open(path) as f:
            config = json.load(f)
        if not servable_mia_config(config):
            print(f"{repo}: a mia-train model cellmap-flow cannot serve (not a segmentation network); not listed")
            return None
        summary = _card_summary(repo) if "README.md" in names else None
        modes, served = _mia_plan(repo, os.path.dirname(path))
        description = " ".join(part for part in (summary, served, MIA_NOTE) if part) if summary else MIA_DESCRIPTION
        metadata = {LISTED_TYPE: "mia", "model_name": repo.split("/")[-1], "description": description}
        return metadata if modes is None else {**metadata, "finetune_modes": modes}
    print(f"{repo}: neither a cellmap-models export nor a mia-train model; not listed")
    return None


def _card_summary(repo: str) -> Optional[str]:
    """The first paragraph of ``repo``'s model card (README.md), past its
    front matter and title, as plain text (a link's text kept, its address
    and the emphasis dropped); None when it has none or cannot be read. The
    MIA team's cards open with what the model predicts, what its encoder
    started from and how long it was trained, which is what tells their
    repos apart."""
    try:
        with open(hf_hub_download(repo, "README.md"), encoding="utf-8") as f:
            lines = f.read().splitlines()
    except Exception as e:
        print(f"{repo}: could not read its model card ({e})")
        return None
    if lines and lines[0].strip() == "---":
        ends = [i for i, line in enumerate(lines[1:], 1) if line.strip() == "---"]
        lines = lines[ends[0] + 1:] if ends else []
    paragraph: List[str] = []
    for line in lines:
        text = line.strip()
        if not text:
            if paragraph:
                break
            continue
        if text.startswith(("#", "|", "```", "<", "!", "-", "*", ">")) and not paragraph:
            continue
        paragraph.append(text)
    if not paragraph:
        return None
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", " ".join(paragraph))
    return text.replace("**", "").replace("`", "")


def _mia_plan(repo: str, folder: str) -> Tuple[Optional[List[str]], Optional[str]]:
    """What the export of the mia-train model of ``repo`` will be, for its
    row: how it can be finetuned, as its config says
    (``MiaModelConfig.finetune_modes``: none for a uint8 output), so the list
    and the training form agree; and a sentence naming the channels it is
    served as, which are not its network's (the semantic model's card names
    its 75 classes, of which it is served as the 12 organelles summed). Both
    None when what the export will declare cannot be read
    (``planned_metadata``): what its config answers then is not the model's
    (it says none while it does not know), which the list leaves out rather
    than cache as what the model takes until a Refresh.

    The config is built of ``folder``, where Hugging Face's cache keeps the
    config.json (and model card) _listed_metadata has just downloaded: that
    is all cellmap_models' mia_metadata needs, as such a folder says its repo
    and commit. Built of the repo id, it would list the repo's snapshot and
    download its licence besides, for every mia-train repo at every build of
    the list, which the list does not keep. (Nor does it look for an export a
    server has made already, which declares what mia_metadata says it will.)"""
    from cellmap_flow.models.configs.mia import MiaModelConfig

    config = MiaModelConfig(path=folder, name=repo.split("/")[-1])
    planned = config.planned_metadata()
    if planned is None:
        print(f"{repo}: what its export will declare could not be read; listed without how it can be finetuned")
        return None, None
    names = list(planned.channels_names or [])
    # A field of the cellmap_models pinned, not of the PyPI release (CI's).
    offsets = getattr(planned, "affinity_offsets", None)
    if offsets:
        served = f"Served here as {len(names)} affinities, offsets {', '.join(str(list(o)) for o in offsets)}."
    else:
        served = f"Served here as {len(names)} channels: {', '.join(names)}." if names else None
    return list(config.finetune_modes()), served


def _of(orgs: Iterable[str], models: Dict[str, Any]) -> Dict[str, Any]:
    """The models of ``orgs`` among ``models`` (keyed by repo id)."""
    return {repo: metadata for repo, metadata in models.items() if repo.split("/")[0] in orgs}


def _as_orgs(orgs) -> Tuple[str, ...]:
    """``orgs``, one organization's name or several, as a tuple."""
    return (orgs,) if isinstance(orgs, str) else tuple(orgs)


def _read_cache() -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """The models cached, by repo id (None when nothing is cached, or the
    cache cannot be read), and the organizations whose listing the cache
    holds (``LISTED_ORGS``; none when it does not say)."""
    try:
        with open(HF_CACHE_FILE) as f:
            cache = json.load(f)
    except (OSError, ValueError):
        return None, []
    return cache, cache.pop(LISTED_ORGS, [])


def _fetch_huggingface_models(orgs=HUGGING_FACE_ORGS) -> Dict[str, Any]:
    """Fetch the models of ``orgs`` (one organization's name, or several)
    from the Hugging Face Hub, and cache them beside what the cache has of
    any other organization; answers every model cached.

    Each organization is listed on its own: one that cannot be (a rate
    limit, one renamed or made private, the Hub failing for a moment) is
    left out, saying why, and the others listed all the same. What the cache
    had of it is kept, so a Refresh at such a moment does not empty its
    group, but the cache no longer says it holds its listing, so the next
    listing asks for it again rather than serving a cache that lacks it as
    whole."""
    orgs = _as_orgs(orgs)
    listed, models = [], []
    for org in orgs:
        try:
            # With their file names, which the facts read (and their sizes, which
            # no listing gives, asked for per repo), and when each last changed.
            models += list(list_models(author=org, expand=["siblings", "lastModified"]))
        except Exception as e:
            print(f"{org}: could not list its Hugging Face models ({e}); listed again next time")
            continue
        listed.append(org)
    result = {}
    linked = _linked_copies() if models else None
    for m in models:
        names = [s.rfilename for s in m.siblings or []]
        try:
            metadata = _listed_metadata(m.id, names)
        except Exception as e:
            print(f"{m.id}: Could not load its metadata.json or config.json ({e})")
            continue
        if metadata is None:
            continue
        facts = _repo_facts(m.id, names, linked)
        if metadata.get(LISTED_TYPE) == "mia":
            # Not from its files: its config's, which _listed_metadata gave
            # when it is known (configs/mia.py MiaModelConfig.finetune_modes).
            del facts["finetune_modes"]
        elif integer_output_refusal(m.id, metadata.get("output_dtype")):
            # A whole-number output passes no gradient (HuggingFaceModelConfig.finetune_modes).
            facts["finetune_modes"] = []
        if m.last_modified is not None:
            facts["last_modified"] = m.last_modified.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        result[m.id] = {**metadata, **facts}

    # The cache's models of every organization not listed now are kept; it
    # holds the listing of those listed now, and of those not asked for that
    # it held before.
    cached, cached_orgs = _read_cache()
    kept = {repo: metadata for repo, metadata in (cached or {}).items() if repo.split("/")[0] not in listed}
    result = {**kept, **result}
    whole = [org for org in cached_orgs if org not in orgs] + listed
    os.makedirs(HF_CACHE_DIR, exist_ok=True)
    with open(HF_CACHE_FILE, "w") as f:
        json.dump({**result, LISTED_ORGS: whole}, f)
    return result


def list_huggingface_models(orgs=HUGGING_FACE_ORGS) -> Dict[str, Any]:
    """
    List available Hugging Face models, using cache if available.

    The organizations whose listing the cache does not hold (every one
    before a first listing; one whose listing failed,
    ``_fetch_huggingface_models``) are fetched, and the others answered from
    the cache.

    Args:
        orgs: the Hugging Face organizations to list (HUGGING_FACE_ORGS by default)

    Returns:
        A dict mapping model IDs to their metadata
    """
    orgs = _as_orgs(orgs)
    cached, listed = _read_cache()
    missing = [org for org in orgs if org not in listed]
    if missing:
        cached = _fetch_huggingface_models(missing)
    return _of(orgs, cached or {})


def cached_huggingface_models() -> Optional[Dict[str, Any]]:
    """The models as the last fetch cached them, or None when nothing is
    cached (or the cache cannot be read), without asking the Hub: the
    Finetune tab's ranking lists them so (routes/finetune/rank_routes.py),
    and its Refresh is the Models tab's, ``refresh_huggingface_models``."""
    return _read_cache()[0]


def refresh_huggingface_models(orgs=HUGGING_FACE_ORGS) -> Dict[str, Any]:
    """Force refresh the Hugging Face models cache."""
    orgs = _as_orgs(orgs)
    return _of(orgs, _fetch_huggingface_models(orgs))
