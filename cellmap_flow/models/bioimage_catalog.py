"""The BioImage Model Zoo's models, for the dashboard's picker.

They come from bioimage.io's artifact server (Hypha), whose listing holds
each model's whole description: its axes (so 2D or 3D, and any declared
voxel size, without a fetch per model) and weights. The legacy
``collection.json`` index, which lags it (125 models to its 157 in
October 2026), is the fallback when the server cannot be reached.

``list_bioimage_models`` answers from a cache file under
``~/.cellmap_flow/bioimage`` once there is one, so opening the Models tab
does not download the zoo's index each time; ``refresh_bioimage_models``
fetches it again. Both give ``{"fetched": <ISO time>, "models": [...]}``,
the index's models in its order, each as ``normalise`` gives it. A failed
fetch raises ``ZooIndexError``, whose message says why, and leaves the
cache as it was.

``bioimage_entry`` turns a ticked model into ``BioModelConfig``'s
constructor arguments: the one place that knows their names.
"""

import inspect
import json
import os
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Optional

ZOO_INDEX_URL = "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/collection.json"
HYPHA_ARTIFACTS = "https://hypha.aicell.io/bioimage-io/artifacts"
HYPHA_MODELS_URL = (
    HYPHA_ARTIFACTS + '/bioimage.io/children?limit=1000&filters={"type":"model"}'
).replace('{"type":"model"}', "%7B%22type%22%3A%22model%22%7D")
# The index is ~300 kB; a stalled connection must not hang the dashboard's
# request thread.
FETCH_TIMEOUT_S = 30
BIOIMAGE_CACHE_DIR = os.path.expanduser("~/.cellmap_flow/bioimage")
BIOIMAGE_CACHE_FILE = os.path.join(BIOIMAGE_CACHE_DIR, "models_cache.json")
ZOO_PAGE_URL = "https://bioimage.io/#/artifacts/{}"

# Tags (lower case) that mark an electron microscopy model. The zoo has no
# controlled vocabulary: "electron-microscopy" and "electron microscopy"
# both occur, and many EM models carry only a dataset or field tag.
_EM_TAGS = {
    "electron-microscopy", "electron microscopy", "em", "tem", "sem", "stem", "fib-sem", "fibsem",
    "sbem", "sbf-sem", "sstem", "ssem", "vem", "volume-em", "connectomics", "cremi", "isbi2012-challenge",
}
_ELECTRON_MICROSCOPY = re.compile(r"electron[\s-]+microscop", re.IGNORECASE)
# Upper case only, and not inside a word: "SEM" is a modality, "semantic" is
# not. An underscore counts as a separator, as in the name "SEM_N2V".
_EM_ACRONYM = re.compile(r"(?<![A-Za-z])(?:v?EM|TEM|STEM|FIB-?SEM|SBF?-?SEM|ss(?:T)?EM|SEM)(?![A-Za-z])")

# Weight formats as the RDF names them, and the tags that stand for them in
# the index, which lists no weights.
_WEIGHT_FORMATS = {
    "pytorch_state_dict": "pytorch",
    "torchscript": "torchscript",
    "onnx": "onnx",
    "tensorflow_saved_model_bundle": "tensorflow",
    "tensorflow_js": "tensorflow",
    "keras_hdf5": "keras",
}
_WEIGHT_TAGS = {"pytorch": "pytorch", "torchscript": "torchscript", "onnx": "onnx",
                "tensorflow": "tensorflow", "keras": "keras"}


class ZooIndexError(RuntimeError):
    """The zoo's index could not be fetched or read."""


def _words(text: str) -> set:
    return set(re.split(r"[^a-z0-9]+", text.lower())) - {""}


def _dims(tags: list, name: str) -> Optional[str]:
    """"2d" or "3d" from the tags ("2d", "3d-segmentation", ...), else from the
    name ("3D UNet ..."); None when neither says, or both do."""
    tag_words = set()
    for tag in tags:
        tag_words |= _words(tag)
    for words in (tag_words, _words(name)):
        found = {"2d", "3d"} & words
        if found:
            return found.pop() if len(found) == 1 else None
    return None


def _is_em(tags: list, text: str) -> bool:
    if any(t.lower() in _EM_TAGS for t in tags):
        return True
    return bool(_ELECTRON_MICROSCOPY.search(text) or _EM_ACRONYM.search(text))


def _weight_formats(raw: dict, tags: list) -> list:
    weights = raw.get("weights")
    if isinstance(weights, dict) and weights:
        found = {_WEIGHT_FORMATS.get(key, key) for key in weights}
    else:
        found = {_WEIGHT_TAGS[t.lower()] for t in tags if t.lower() in _WEIGHT_TAGS}
    return sorted(found)


def _is_doi(zoo_id: str) -> bool:
    return zoo_id.startswith("10.")


def normalise(raw: dict) -> dict:
    """One index entry as the picker shows it.

    ``id`` is the index's (a Zenodo DOI for older models, the nickname for
    newer ones), ``nickname`` the animal name ("affable-shark") or None, and
    ``key`` the one to load it by, the nickname when there is one. ``dims``
    is "2d", "3d" or None; ``em`` whether its tags, name or description say
    electron microscopy; ``weight_formats`` short names ("pytorch", "onnx").
    """
    zoo_id = str(raw["id"])
    nickname = raw.get("nickname") or None
    key = nickname or zoo_id
    tags = [str(t) for t in raw.get("tags") or []]
    name = str(raw.get("name") or key)
    description = " ".join(str(raw.get("description") or "").split())
    covers = raw.get("covers") or []
    return {
        "id": zoo_id,
        "nickname": nickname,
        "key": key,
        "name": name,
        "description": description,
        "tags": tags,
        "dims": _dims(tags, name),
        "em": _is_em(tags, f"{name} {description}"),
        "weight_formats": _weight_formats(raw, tags),
        "license": raw.get("license"),
        "cover": covers[0] if covers and isinstance(covers[0], str) else None,
        # The model's description, read for its declared voxel size
        # (declared_voxel_size) without bioimageio.core.
        "rdf_source": raw.get("rdf_source") if isinstance(raw.get("rdf_source"), str) else None,
        # A DOI-only entry has no page of its own on bioimage.io; its DOI
        # resolves to the Zenodo record.
        "url": f"https://doi.org/{key}" if _is_doi(key) else ZOO_PAGE_URL.format(key),
    }


def _fetch_index(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "cellmap-flow"})
    try:
        with urllib.request.urlopen(request, timeout=FETCH_TIMEOUT_S) as response:
            return json.load(response)
    except (urllib.error.URLError, OSError) as e:
        raise ZooIndexError(f"Could not fetch the BioImage Model Zoo index from {url}: {e}") from e
    except ValueError as e:
        raise ZooIndexError(f"The BioImage Model Zoo index at {url} is not JSON: {e}") from e


def _space_axes(axes):
    """The ids of an RDF input's space axes, and nm per voxel for those with a unit.

    ``axes`` is spec 0.5's list of axis objects, or 0.4's string ("bczyx"),
    which carries no units.
    """
    if isinstance(axes, str):
        return [a for a in axes if a in "zyx"], {}
    ids, sizes = [], {}
    for axis in axes if isinstance(axes, list) else []:
        if isinstance(axis, dict) and axis.get("type") == "space":
            ids.append(axis.get("id"))
            if axis.get("unit") in _NM_PER_UNIT:
                sizes[axis.get("id")] = float(axis.get("scale", 1.0)) * _NM_PER_UNIT[axis["unit"]]
    return ids, sizes


def _voxel_size_from(sizes: dict) -> Optional[list]:
    if "y" in sizes and "x" in sizes:
        # A 2D model's z is its slices' spacing, which BioModelConfig takes as its y.
        return [sizes.get("z", sizes["y"]), sizes["y"], sizes["x"]]
    return None


def normalise_artifact(artifact: dict) -> Optional[dict]:
    """One Hypha artifact as the picker shows it: ``normalise``'s fields from
    its description, with 2D/3D and the declared voxel size from its input's
    axes. None for an artifact with no description."""
    manifest = artifact.get("manifest")
    alias = artifact.get("alias")
    if not isinstance(manifest, dict) or not alias:
        return None
    files = f"{HYPHA_ARTIFACTS}/{alias}/files/"
    covers = [c if str(c).startswith("http") else files + str(c) for c in manifest.get("covers") or []]
    entry = normalise({
        **manifest, "id": alias, "nickname": alias, "covers": covers, "rdf_source": files + "rdf.yaml",
    })
    inputs = manifest.get("inputs") or []
    ids, sizes = _space_axes(inputs[0].get("axes") if inputs and isinstance(inputs[0], dict) else None)
    if ids:
        entry["dims"] = "3d" if "z" in ids else "2d"
    entry["declared_voxel_size"] = _voxel_size_from(sizes)
    entry["downloads"] = artifact.get("download_count")
    return entry


def _fetch_from_hypha() -> list:
    listing = _fetch_index(HYPHA_MODELS_URL)
    artifacts = listing.get("items") if isinstance(listing, dict) else listing
    if not isinstance(artifacts, list):
        raise ZooIndexError(f"bioimage.io's model listing at {HYPHA_MODELS_URL} is not a list")
    return [e for e in (normalise_artifact(a) for a in artifacts if isinstance(a, dict)) if e]


def _fetch_from_index() -> list:
    index = _fetch_index(ZOO_INDEX_URL)
    entries = index.get("collection") if isinstance(index, dict) else None
    if not isinstance(entries, list):
        raise ZooIndexError(f"The BioImage Model Zoo index at {ZOO_INDEX_URL} has no 'collection' list.")
    return [normalise(e) for e in entries if isinstance(e, dict) and e.get("type") == "model" and e.get("id")]


def _fetch_models() -> dict:
    """Fetch and normalise the models, and cache them: the cache document.

    From bioimage.io's artifact server, else (unreachable, or an answer
    not understood) from the legacy index; both failing raises the
    server's error with the index's.
    """
    try:
        models, source = _fetch_from_hypha(), "hypha"
    except ZooIndexError as hypha_error:
        try:
            models, source = _fetch_from_index(), "index"
        except ZooIndexError as index_error:
            raise ZooIndexError(f"{hypha_error}; and {index_error}") from index_error
    document = {"fetched": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "source": source, "models": models}

    # Written beside and moved into place: a reader never sees half a file.
    os.makedirs(os.path.dirname(BIOIMAGE_CACHE_FILE), exist_ok=True)
    partial = BIOIMAGE_CACHE_FILE + ".partial"
    with open(partial, "w") as f:
        json.dump(document, f)
    os.replace(partial, BIOIMAGE_CACHE_FILE)
    return document


def _read_cache() -> Optional[dict]:
    try:
        with open(BIOIMAGE_CACHE_FILE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def list_bioimage_models() -> dict:
    """The zoo's models, from the cache when there is one."""
    return _read_cache() or _fetch_models()


def refresh_bioimage_models() -> dict:
    """Fetch the zoo's index again, and cache it."""
    return _fetch_models()


def find_bioimage_model(key: str) -> Optional[dict]:
    """The cached model whose id, nickname or key is ``key``, or None.

    Only the cache is read: the Models tab's submit must not wait on the
    network, and a model the tab listed is in it.
    """
    for model in (_read_cache() or {}).get("models", []):
        if key in (model["id"], model["nickname"], model["key"]):
            return model
    return None


# nm per unit of the RDF's space axes (bioimage.io spec 0.5).
_NM_PER_UNIT = {"angstrom": 0.1, "nanometer": 1.0, "micrometer": 1e3, "millimeter": 1e6}
_declared = {}


def declared_voxel_size(entry: dict) -> Optional[list]:
    """The voxel size (nm, z y x) the model's description declares, None when it declares none.

    Read from its RDF (``rdf_source``) with a plain fetch, so the dashboard,
    which has no bioimageio.core, can refuse a blank voxel size at Submit:
    the server would otherwise refuse it after its job started, out of
    sight. Remembered per model. Raises ZooIndexError when the description
    cannot be read, which the caller treats as "don't know".
    """
    if "declared_voxel_size" in entry:
        return entry["declared_voxel_size"]  # read from the listing's description
    source = entry.get("rdf_source")
    if not source:
        raise ZooIndexError(f"{entry.get('key')} names no description to read")
    if source not in _declared:
        import yaml

        request = urllib.request.Request(source, headers={"User-Agent": "cellmap-flow"})
        try:
            with urllib.request.urlopen(request, timeout=FETCH_TIMEOUT_S) as response:
                rdf = yaml.safe_load(response.read())
        except (urllib.error.URLError, OSError, ValueError, yaml.YAMLError) as e:
            raise ZooIndexError(f"Could not read {source}: {e}") from e
        inputs = (rdf or {}).get("inputs") or []
        _, sizes = _space_axes(inputs[0].get("axes") if inputs and isinstance(inputs[0], dict) else None)
        _declared[source] = _voxel_size_from(sizes)
    return _declared[source]


def _model_parameter(cls) -> str:
    """The constructor argument naming the zoo model: ``model`` or ``model_name``."""
    params = inspect.signature(cls.__init__).parameters
    return "model" if "model" in params else "model_name"


def bioimage_entry(model_id: str, voxel_size=None, name: Optional[str] = None, cls=None) -> dict:
    """``BioModelConfig``'s constructor arguments for the zoo model ``model_id``
    (an id or nickname bioimageio.core loads), at ``voxel_size`` (nm, z y x)
    and called ``name``.

    The argument names come from the signature, so this holds whether the
    model is called ``model_name`` or ``model``. A ``voxel_size`` of None
    is left out when the class can read it from the model; when it cannot,
    a ValueError says to give it. ``cls`` is for tests.
    """
    if cls is None:
        from cellmap_flow.models.configs.bio import BioModelConfig as cls

    params = inspect.signature(cls.__init__).parameters
    entry = {_model_parameter(cls): model_id}
    if voxel_size is not None:
        entry["voxel_size"] = list(voxel_size)
    elif "voxel_size" in params and params["voxel_size"].default is inspect.Parameter.empty:
        raise ValueError(f"BioImage model '{model_id}' needs a voxel size: give it in nm, as z,y,x or one number.")
    if name is not None:
        entry["name"] = name
    return entry


def entry_model_id(model_config) -> Optional[str]:
    """The zoo model a ``BioModelConfig`` runs, read from its ``to_dict()``,
    which ``bioimage_entry``'s names round-trip through."""
    return model_config.to_dict().get(_model_parameter(type(model_config)))
