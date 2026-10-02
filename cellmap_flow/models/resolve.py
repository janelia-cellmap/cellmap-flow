"""A model from whatever reference a user has in hand, without knowing the model types.

``resolve(ref)`` turns a path, a URL, a Hugging Face repo, a zoo nickname or
a Cellpose model name into a ``Resolved``: the model type, its constructor
arguments, a name, and what the user must still give. ``Resolved.entry()``
is the model entry that ``registry.build_model`` (and so a YAML's
``models:``) takes. The dashboard's ``POST /api/models/resolve`` and
``cellmap_flow add`` answer with it.

The checks, in this order; the first that matches wins. Each is cheap: a
stat, a file name, a regex, or (only when ``online``) one request.

1. A local file or folder (``~`` expanded; the entry gets its absolute path):

   - a ``.py`` file: ``script``;
   - a folder with ``metadata.json``: ``cellmap`` (a cellmap-models export,
     which is served from its ``model.ts``);
   - a folder with ``rdf.yaml`` or ``bioimageio.yaml``, such a file itself
     (``*.rdf.yaml`` too), or a ``.zip`` with one inside: ``bioimage``;
   - ``model_checkpoint_<n>``, a ``.ts`` or a ``model.pt``: ``fly``, which
     reads the channels and voxel sizes from the files beside it; what it
     cannot find is reported in ``needs``;
   - a folder with ``adapter_config.json`` (a PEFT LoRA adapter), or a full
     finetune's ``model_state_dict.pt``: ``finetune``, which needs its
     ``base_model``;
   - Cellpose-SAM weights (Cellpose 4 saves a torch state dict without an
     extension; its keys say what it is): ``cellpose``.

2. A prefix, to say outright what a reference is: ``hf:org/repo[@revision]``,
   ``bioimageio:<id, nickname, URL or path>``, ``cellpose:<name or path>``,
   ``fly:<path>``, ``dacapo:<run>@<iteration>``, ``script:<path>``. The
   type's own name works as its prefix too (``huggingface:``,
   ``bioimage:``). Nothing after a prefix is looked up online.

3. A Cellpose model name (``cpsam_v2``, ``cpsam``, ``cpdino``,
   ``cpdino-vitb``): ``cellpose``.

4. A URL: a bioimage.io model page, a Zenodo record, a DOI link, a Hypha
   artifact or the URL of an ``rdf.yaml`` or zip: ``bioimage``;
   ``huggingface.co/<org>/<repo>`` (``/tree/<revision>`` too): as 5.

5. ``org/repo`` (``@revision`` optional; not a DOI, which is 6's): a Hugging
   Face repo. Online, its file list is read: with ``metadata.json`` and
   ``model.ts`` it is a cellmap-models export, ``huggingface``; anything
   else is refused with what to do instead. Offline, or when the Hub cannot
   be reached, it is taken to be one, with a note.

6. A BioImage Model Zoo id or nickname (``affable-shark``,
   ``10.5281/zenodo.5764892``): ``bioimage``. Online, it is looked up in
   the zoo's model list; offline, a nickname- or DOI-shaped one is taken to be
   one, with a note.

Anything else is a ValueError that says what was tried.

Importing this module imports only the standard library. Resolving imports
the model config classes (numpy and funlib, as the registry does), and the
online checks huggingface_hub; nothing imports torch or a model's framework,
and ``Resolved.entry()`` is a plain dict.
"""

import inspect
import os
import re
import zipfile
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

# Seconds to wait for the Hub before treating them as
# unreachable: a dashboard request waits this long at most.
ONLINE_TIMEOUT = 20

# What a cellmap-models export holds; the cellmap and huggingface types serve
# the TorchScript.
EXPORT_METADATA, EXPORT_MODEL = "metadata.json", "model.ts"
# A PEFT LoRA adapter's folder has this; the finetune trainer saves one.
ADAPTER_CONFIG = "adapter_config.json"
# What the finetune trainer saves a full finetune (--lora-r 0) as.
FULL_FINETUNE_WEIGHTS = "model_state_dict.pt"
# A bioimage.io model description, by file name.
_RDF_NAMES = ("rdf.yaml", "bioimageio.yaml")
_RDF_SUFFIXES = (".rdf.yaml", ".bioimageio.yaml")
# Keys only Cellpose-SAM's weights have: its SAM image encoder's patch
# embedding and neck. They are plain strings in the pickle inside the zip
# that torch.save writes, so they are found without torch.
_CELLPOSE_SAM_KEYS = (b"encoder.patch_embed", b"encoder.neck")

_PREFIX = re.compile(r"^([A-Za-z]+):(?!//)(.*)$", re.DOTALL)
_HF_REPO = re.compile(r"^[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*(?:@[\w./-]+)?$")
_DOI = re.compile(r"^10\.\d{4,9}/\S+$")
_NICKNAME = re.compile(r"^[a-z]+(?:-[a-z]+)+$")
_VERSION = re.compile(r"^v?\d+(?:\.\d+)*$")


@dataclass
class Resolved:
    """What a reference resolved to.

    Attributes:
        type: the model type's name (``registry.model_types``).
        params: its constructor arguments, as a YAML entry gives them.
        name: the model's name: the one asked for, else one from the reference.
        env: the type's default environment for this model (``default_env``),
            for display; None runs it in cellmap-flow's own. Not in ``entry()``.
        how: one line saying what the reference was taken to be.
        needs: constructor arguments the user must still give (a key of
            ``params`` once given), e.g. ``voxel_size`` for Cellpose.
        notes: anything else worth saying: what was assumed, or ignored.
    """

    type: str
    params: Dict[str, Any]
    name: str
    env: Optional[str]
    how: str
    needs: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def entry(self) -> Dict[str, Any]:
        """The model entry ``registry.build_model`` takes: type, params and name.

        While ``needs`` is not empty, building it fails on what is missing.
        """
        return {"type": self.type, **self.params, "name": self.name}

    def to_json(self) -> Dict[str, Any]:
        """Every field, as the dashboard's route answers with them."""
        return asdict(self)


def resolve(ref: str, *, name: str = None, voxel_size=None, online: bool = True) -> Resolved:
    """The model ``ref`` refers to (see the module docstring for the checks, in order).

    Args:
        ref: what the user pasted.
        name: the model's name; by default one is made from ``ref``.
        voxel_size: nm per voxel (one number, one per axis, or "8,8,8"), for
            the types that do not read it from the model: given to Cellpose
            and bioimage.io models as ``voxel_size`` and to a fly model as
            ``input_voxel_size``, and ignored, with a note, by the others.
        online: whether a Hugging Face repo or zoo id may be checked online.

    Raises:
        ValueError: ``ref`` is none of the things checked, or is one that
            cellmap-flow cannot serve; the message says which and why.
    """
    if not isinstance(ref, str) or not ref.strip():
        raise ValueError("Give a model reference: a path, a URL, a Hugging Face repo or a model name")
    ref = ref.strip()
    request = _Request(name=name or None, voxel_size=_voxel_size(voxel_size), online=online)
    for check in (_local_path, _prefixed, _cellpose_name, _url, _hf_repo, _zoo_id):
        resolved = check(ref, request)
        if resolved is not None:
            return resolved
    raise ValueError(
        f"Could not tell what model {ref!r} is. Tried, in order: a local file or folder (there is "
        f"none{' at ' + _expanded(ref) if _looks_like_path(ref) else ''}); a prefix (hf:, bioimageio:, "
        "cellpose:, fly:, dacapo:, script:); a Cellpose model name; a URL; a Hugging Face repo "
        "(org/repo); and a BioImage Model Zoo id or nickname"
        f"{' (offline: not looked up)' if not online else ''}. Say what it is with a prefix, "
        "e.g. fly:/path/to/checkpoint or dacapo:my_run@100000."
    )


@dataclass
class _Request:
    name: Optional[str]
    voxel_size: Optional[list]
    online: bool


def _voxel_size(value) -> Optional[list]:
    """A voxel size as three numbers, ints kept ints; from one number, one per axis or "8,8,8"."""
    if value is None or value == "" or value == []:
        return None
    if isinstance(value, str):
        # "8", "16,8,8", "(16, 8, 8)" or "[16, 8, 8]", as a form field or a flag has it.
        value = re.findall(r"[^\s,()\[\]]+", value)
    if not isinstance(value, (list, tuple)):
        value = [value]
    try:
        numbers = [float(v) for v in value]
    except (TypeError, ValueError):
        raise ValueError(f"voxel_size must be numbers (one, or one per axis), not {value!r}") from None
    if len(numbers) == 1:
        numbers *= 3
    if len(numbers) != 3 or any(n <= 0 for n in numbers):
        raise ValueError(f"voxel_size must be one positive number or three, not {value!r}")
    return [int(n) if n.is_integer() else n for n in numbers]


# --- what a Resolved says ------------------------------------------------------

def _type_class(type_name: str):
    from cellmap_flow.models import registry

    return registry.model_type(type_name)


def _default_env(type_name: str, params: Dict[str, Any]) -> Optional[str]:
    """The type's ``default_env`` for a model with ``params``, without building one.

    A type that decides per model does so with a property, which reads the
    model's attributes; those are its constructor arguments by name (fly's
    ``checkpoint_path``), so it is asked with them. When it needs more than
    that (a finetune's base model), None.
    """
    from cellmap_flow.models import envs

    declared = envs.declared_default(_type_class(type_name))
    if not envs.decides_per_model(declared):
        return declared
    try:
        return declared.__get__(SimpleNamespace(**params))
    except Exception:
        return None


def _resolved(type_name, params, name, how, request, needs=(), notes=(), takes_voxel_size=False) -> Resolved:
    notes = list(notes)
    if request.voxel_size is not None and not takes_voxel_size:
        notes.append(f"voxel_size is not used: a {type_name} model says its own")
    return Resolved(
        type=type_name,
        params=params,
        name=request.name or name,
        env=_default_env(type_name, params),
        how=how,
        needs=list(needs),
        notes=notes,
    )


def _with_voxel_size(type_name, params, how, name, request, notes=()) -> Resolved:
    """A type whose ``voxel_size`` is required (Cellpose): given, or needed."""
    needs = []
    if request.voxel_size is not None:
        params["voxel_size"] = request.voxel_size
    else:
        needs.append("voxel_size")
    return _resolved(type_name, params, name, how, request, needs=needs, notes=notes, takes_voxel_size=True)


# --- 1. a local file or folder ---------------------------------------------------

def _expanded(ref: str) -> str:
    return os.path.abspath(os.path.expanduser(ref))


def _looks_like_path(ref: str) -> bool:
    """Whether ``ref`` is written as a path, so that its absence is worth saying."""
    return ref.startswith(("/", "./", "../", "~")) or (
        "://" not in ref and ref.lower().endswith((".py", ".ts", ".pt", ".zip", ".yaml", ".yml"))
    )


def _local_path(ref: str, request: _Request) -> Optional[Resolved]:
    path = _expanded(ref)
    if not os.path.exists(path):
        return None
    if os.path.isdir(path):
        return _folder(path, request)
    return _file(path, request)


def _folder(path: str, request: _Request) -> Resolved:
    folder = os.path.basename(path.rstrip("/"))
    if os.path.isfile(os.path.join(path, EXPORT_METADATA)):
        if not os.path.isfile(os.path.join(path, EXPORT_MODEL)):
            raise ValueError(
                f"{path} has a {EXPORT_METADATA} but no {EXPORT_MODEL}, which type: cellmap serves. "
                f"Its model.pt, if it has one, can be served as fly:{os.path.join(path, 'model.pt')}."
            )
        return _resolved("cellmap", {"folder_path": path}, folder, "a cellmap-models export folder", request)
    rdf = _bioimage_source(path)
    if rdf is not None:
        return _bioimage(rdf, folder, "a bioimage.io model folder", request)
    if os.path.isfile(os.path.join(path, ADAPTER_CONFIG)):
        return _finetune("lora_adapter_path", path, "a finetune's LoRA adapter", request)
    checkpoints = sorted(
        (n for n in os.listdir(path) if _is_fly_checkpoint(n)), key=_checkpoint_order
    )
    hint = f", such as {os.path.join(path, checkpoints[-1])}" if checkpoints else ""
    raise ValueError(
        f"{path} is a folder, but not one cellmap-flow knows: no {EXPORT_METADATA} (a cellmap-models "
        f"export), rdf.yaml (a bioimage.io model) or {ADAPTER_CONFIG} (a LoRA adapter). Give the "
        f"model file in it instead{hint}."
    )


def _file(path: str, request: _Request) -> Resolved:
    base = os.path.basename(path)
    if base.endswith(".py"):
        return _resolved("script", {"script_path": path}, base[:-3], "a Python script that defines the model", request)
    rdf = _bioimage_source(path)
    if rdf is not None:
        how = "a bioimage.io model package" if path.endswith(".zip") else "a bioimage.io model description"
        return _bioimage(rdf, _bioimage_name(path), how, request)
    if _is_fly_checkpoint(base):
        return _fly(path, request)
    if base == FULL_FINETUNE_WEIGHTS:
        return _finetune("weights_path", path, "a full finetune's weights", request)
    if _is_cellpose_weights(path):
        return _with_voxel_size("cellpose", {"pretrained_model": path}, "Cellpose-SAM weights", base, request)
    if path.endswith(".zip"):
        raise ValueError(f"{path} is a zip without an rdf.yaml or bioimageio.yaml in it: not a bioimage.io model")
    raise ValueError(
        f"Could not tell what model {path} is from its name or contents. Say it with a prefix: "
        f"fly:{path} (a fly_organelles checkpoint), cellpose:{path} (Cellpose weights) or script:<a .py "
        "that loads it>."
    )


def _bioimage_source(path: str) -> Optional[str]:
    """What bioimageio.core loads for a local ``path``, or None when it is not a model of the zoo's kind.

    A folder's rdf.yaml (or bioimageio.yaml); the file itself when it is
    one; a zip with one inside, as the zip.
    """
    def is_rdf(name):
        name = os.path.basename(name)
        return name in _RDF_NAMES or name.endswith(_RDF_SUFFIXES)

    if os.path.isdir(path):
        for name in sorted(os.listdir(path)):
            if is_rdf(name):
                return os.path.join(path, name)
        return None
    if is_rdf(path):
        return path
    if path.endswith(".zip") and zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as package:
            return path if any(is_rdf(n) for n in package.namelist()) else None
    return None


def _is_fly_checkpoint(name: str) -> bool:
    return name.startswith("model_checkpoint_") or name.endswith(".ts") or name == "model.pt"


def _checkpoint_order(name: str):
    number = name.rsplit("_", 1)[-1]
    return (int(number) if number.isdigit() else -1, name)


def _is_cellpose_weights(path: str) -> bool:
    """Whether ``path`` is Cellpose-SAM's weights, by the key names in its pickle.

    torch.save writes a zip whose ``<name>/data.pkl`` holds the state dict's
    keys as plain strings; the tensors are separate members, so this reads
    tens of kB of a GB file.
    """
    try:
        if not zipfile.is_zipfile(path):
            return False
        with zipfile.ZipFile(path) as weights:
            pickles = [n for n in weights.namelist() if n.endswith("data.pkl")]
            if len(pickles) != 1:
                return False
            with weights.open(pickles[0]) as f:
                keys = f.read(8 << 20)
    except (OSError, zipfile.BadZipFile):
        return False
    return all(key in keys for key in _CELLPOSE_SAM_KEYS)


# --- the types that need more than a path ----------------------------------------

def bioimage_params(model: str, voxel_size=None, cls=None) -> Tuple[Dict[str, Any], List[str]]:
    """``BioModelConfig``'s arguments for ``model``, and the ones still needed, from its signature.

    Its model argument is ``model`` or (before) ``model_name``, and its
    ``voxel_size`` may be required or not; whichever it is, the entry names
    it as the constructor does. ``voxel_size`` goes in when given and taken,
    and is needed when not given and required.

    Raises:
        ValueError: the constructor takes neither ``model`` nor ``model_name``.
    """
    if cls is None:
        cls = _type_class("bioimage")
    parameters = inspect.signature(cls.__init__).parameters
    key = next((k for k in ("model", "model_name") if k in parameters), None)
    if key is None:
        raise ValueError(f"{cls.__name__} takes neither model nor model_name; cannot say which model it is")
    params, needs = {key: model}, []
    takes = parameters.get("voxel_size")
    if takes is not None:
        if voxel_size is not None:
            params["voxel_size"] = voxel_size
        elif takes.default is inspect.Parameter.empty:
            needs.append("voxel_size")
    return params, needs


def _bioimage(model: str, name: str, how: str, request: _Request, notes=()) -> Resolved:
    from cellmap_flow.models.bioimage_catalog import trained_at

    notes = list(notes)
    voxel_size = request.voxel_size
    trained = trained_at(model) or trained_at(name) or {}
    if voxel_size is None and trained.get("voxel_size"):
        voxel_size = trained["voxel_size"]
        notes.append(f"voxel_size {voxel_size} is what it was trained at, on {trained['trained_on']} "
                     f"({trained['confidence']} confidence; {trained['source']})")
    if trained.get("note"):
        notes.append(trained["note"])
    params, needs = bioimage_params(model, voxel_size)
    if request.voxel_size is not None and "voxel_size" not in params:
        notes.append("voxel_size is not used: this bioimage model type does not take one")
    return _resolved("bioimage", params, name, how, request, needs=needs, notes=notes, takes_voxel_size=True)


def _bioimage_name(source: str) -> str:
    """A model name from a bioimage.io source: its nickname, record or folder, not "files" or "1.2"."""
    parts = [p for p in re.split(r"[/:]", source) if p]
    for part in reversed(parts):
        stem = part
        for suffix in (*_RDF_SUFFIXES, ".zip", ".yaml"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        if stem and stem not in ("files", "rdf", "bioimageio") and not _VERSION.match(stem):
            return stem
    return "bioimage_model"


_FLY_NEEDS = re.compile(r"needs (\w+):")


def _fly(path: str, request: _Request) -> Resolved:
    """A fly model: what the constructor reads beside the checkpoint, and what it says is missing.

    The constructor is the one that knows which files say what, so it is
    asked; it reads small files beside the checkpoint and loads no weights.
    It stops at the first argument it lacks, so after missing channels it is
    asked again with stand-ins, to hear about the voxel size too.
    """
    cls = _type_class("fly")
    params = {"checkpoint_path": path}
    if request.voxel_size is not None:
        params["input_voxel_size"] = request.voxel_size
    needs, notes = [], []
    attempt = dict(params)
    while True:
        try:
            model = cls(**attempt)
            break
        except ValueError as e:
            missing = _FLY_NEEDS.search(str(e))
            if missing is None or missing[1] in needs or missing[1] in attempt:
                raise
            needs.append(missing[1])
            attempt[missing[1]] = ["channel"] if missing[1] == "channels" else [1, 1, 1]
    if not needs:
        read = f"channels {', '.join(model.channels)}"
        if "input_voxel_size" not in params:
            read += f" and voxel size {list(model.input_voxel_size)}"
        notes.append(f"reads {read} from the files beside the checkpoint")

    from cellmap_flow.models.configs.fly import PICKLE, TORCHSCRIPT, checkpoint_format

    base, run = os.path.basename(path), os.path.basename(os.path.dirname(path))
    kind = checkpoint_format(path)
    if kind == TORCHSCRIPT:
        how, name = "a TorchScript model, served as a fly model", base[:-3] if base != "model.ts" else run
    elif kind == PICKLE:
        how, name = "a whole pickled fly_organelles model (model.pt)", run
        notes.append("unpickled only when CELLMAP_FLOW_ALLOW_PICKLE=1 is set where it is served")
    else:
        iteration = base[len("model_checkpoint_"):]
        how, name = "a fly_organelles training checkpoint", f"{run}_{iteration}" if run else base
    return _resolved("fly", params, name, how, request, needs=needs, notes=notes, takes_voxel_size=True)


def _finetune(key: str, path: str, how: str, request: _Request) -> Resolved:
    # The trainer saves <run>/lora_adapter and <run>/full_finetune/model_state_dict.pt,
    # so the run's folder names the model when the path is one of those.
    folder = os.path.dirname(path) if key == "weights_path" else path
    if os.path.basename(folder) in ("lora_adapter", "full_finetune"):
        folder = os.path.dirname(folder)
    return _resolved(
        "finetune", {key: path}, os.path.basename(folder), how, request,
        needs=["base_model"],
        notes=["base_model is the model entry it was finetuned from; its serving YAML has it"],
    )


# --- 2. a prefix -------------------------------------------------------------------

def _prefixed(ref: str, request: _Request) -> Optional[Resolved]:
    match = _PREFIX.match(ref)
    if match is None:
        return None
    prefix, value = match[1].lower(), match[2].strip()
    handler = {
        "hf": _hf_prefixed, "huggingface": _hf_prefixed,
        "bioimageio": _bioimage_prefixed, "bioimage": _bioimage_prefixed,
        "cellpose": _cellpose_prefixed,
        "fly": _fly_prefixed,
        "dacapo": _dacapo_prefixed,
        "script": _script_prefixed,
    }.get(prefix)
    if handler is None:
        return None
    if not value:
        raise ValueError(f"{prefix}: needs something after it")
    return handler(value, request)


def _existing(value: str, prefix: str) -> str:
    path = _expanded(value)
    if not os.path.exists(path):
        raise ValueError(f"{prefix}:{value}: there is no {path}")
    return path


def _hf_prefixed(value: str, request: _Request) -> Resolved:
    repo, _, revision = value.partition("@")
    return _huggingface(repo, revision or None, "a cellmap-models export on Hugging Face", request)


def _bioimage_prefixed(value: str, request: _Request) -> Resolved:
    if value.startswith(("http://", "https://")):
        source = _bioimage_url(value) or value
        return _bioimage(source, _bioimage_name(source), "a bioimage.io model at a URL", request)
    path = _expanded(value)
    if os.path.exists(path):
        source = _bioimage_source(path)
        if source is None:
            raise ValueError(f"{path} is not a bioimage.io model: no rdf.yaml or bioimageio.yaml in it")
        return _bioimage(source, _bioimage_name(path), "a bioimage.io model", request)
    return _bioimage(value, _bioimage_name(value), "a BioImage Model Zoo model", request)


def _cellpose_prefixed(value: str, request: _Request) -> Resolved:
    path = _expanded(value)
    if os.path.exists(path):
        return _with_voxel_size("cellpose", {"pretrained_model": path}, "Cellpose weights",
                                os.path.basename(path), request)
    # Not checked here: Cellpose also knows the models a user has added to
    # it by name (models.get_user_models), and says so when the model is built.
    return _with_voxel_size("cellpose", {"pretrained_model": value}, "a Cellpose 4 pretrained model", value, request)


def _fly_prefixed(value: str, request: _Request) -> Resolved:
    return _fly(_existing(value, "fly"), request)


def _script_prefixed(value: str, request: _Request) -> Resolved:
    path = _existing(value, "script")
    return _resolved("script", {"script_path": path}, os.path.splitext(os.path.basename(path))[0],
                     "a Python script that defines the model", request)


def _dacapo_prefixed(value: str, request: _Request) -> Resolved:
    # rpartition: the iteration follows the last @, so a run name may hold one.
    run, at, iteration = value.rpartition("@")
    if not at:
        run, iteration = value, ""
    params, needs = {"run_name": run}, []
    if iteration:
        if not iteration.isdigit():
            raise ValueError(f"dacapo:{value}: the iteration after @ must be a whole number, not {iteration!r}")
        params["iteration"] = int(iteration)
    else:
        needs.append("iteration")
    name = f"{run}_{iteration}" if iteration else run
    return _resolved("dacapo", params, name, "a DaCapo run at one iteration", request, needs=needs)


# --- 3. a Cellpose model name ---------------------------------------------------------

def _cellpose_name(ref: str, request: _Request) -> Optional[Resolved]:
    from cellmap_flow.models.configs.cellpose import PRETRAINED_MODELS

    if ref not in PRETRAINED_MODELS:
        return None
    return _with_voxel_size("cellpose", {"pretrained_model": ref}, "a Cellpose 4 pretrained model", ref, request)


# --- 4. a URL ---------------------------------------------------------------------------

def _url(ref: str, request: _Request) -> Optional[Resolved]:
    if not ref.startswith(("http://", "https://")):
        return None
    parsed = urlparse(ref)
    host = (parsed.hostname or "").lower()
    if host in ("huggingface.co", "www.huggingface.co", "hf.co"):
        repo, revision = _hf_url(parsed)
        return _hf_checked(repo, revision, request)
    source = _bioimage_url(ref)
    if source is not None:
        return _bioimage(source, _bioimage_name(source), "a bioimage.io model at a URL", request)
    raise ValueError(
        f"{ref} is a URL, but not one of a model cellmap-flow knows: a Hugging Face repo, a bioimage.io "
        "or Zenodo page, or the URL of an rdf.yaml or zip"
    )


def _hf_url(parsed) -> Tuple[str, Optional[str]]:
    """(repo, revision) from a huggingface.co URL: /<org>/<repo>, /tree/<rev> or /blob/<rev>/..."""
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2 or parts[0] in ("datasets", "spaces", "models", "docs", "collections"):
        raise ValueError(f"{parsed.geturl()} is not a Hugging Face model repo's page (huggingface.co/<org>/<repo>)")
    revision = parts[3] if len(parts) > 3 and parts[2] in ("tree", "blob", "resolve") else None
    return f"{parts[0]}/{parts[1]}", revision


def _bioimage_url(url: str) -> Optional[str]:
    """What bioimageio.core loads for a bioimage.io-ish URL, or None when it is none.

    A description or package's own URL (rdf.yaml, bioimageio.yaml, .zip) is
    loaded as it is. A page is turned into the id it shows: a bioimage.io
    page's ``id``, a Zenodo record's DOI (``10.5281/zenodo.<record>``), a
    doi.org link's DOI, a Hypha artifact's name.
    """
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    path = parsed.path
    name = os.path.basename(path)
    if name in _RDF_NAMES or name.endswith(_RDF_SUFFIXES) or name.endswith(".zip"):
        return url
    if host == "bioimage.io" or host.endswith(".bioimage.io"):
        # The site routes in its fragment: #/?id=affable-shark or #/artifacts/affable-shark.
        fragment = urlparse(parsed.fragment)
        ids = parse_qs(fragment.query).get("id") or parse_qs(parsed.query).get("id")
        if ids:
            return ids[0]
        artifact = _after(fragment.path, "artifacts") or _after(path, "artifacts")
        return artifact
    if host in ("zenodo.org", "www.zenodo.org"):
        record = re.search(r"/records?/(\d+)", path)
        return f"10.5281/zenodo.{record[1]}" if record else None
    if host in ("doi.org", "dx.doi.org"):
        doi = path.lstrip("/")
        return doi if _DOI.match(doi) else None
    if host.endswith("aicell.io"):
        return _after(path, "artifacts")
    return None


def _after(path: str, segment: str) -> Optional[str]:
    parts = [p for p in path.split("/") if p]
    return parts[parts.index(segment) + 1] if segment in parts[:-1] else None


# --- 5. a Hugging Face repo ---------------------------------------------------------------

def _hf_repo(ref: str, request: _Request) -> Optional[Resolved]:
    # A zoo DOI is org/repo-shaped too, and so is a relative path such as
    # runs/model.ts that is not there; they are 6's and the error's.
    if not _HF_REPO.match(ref) or _DOI.match(ref) or _looks_like_path(ref):
        return None
    repo, _, revision = ref.partition("@")
    return _hf_checked(repo, revision or None, request)


def _hf_files(repo: str, revision: Optional[str]) -> List[str]:
    """The files of a Hugging Face model repo.

    Raises:
        LookupError: there is no such repo (or revision) that can be read.
        Exception: anything else, such as the Hub being unreachable.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError

    try:
        return HfApi().list_repo_files(repo, revision=revision)
    except RevisionNotFoundError as e:
        raise LookupError(f"Hugging Face repo {repo} has no revision {revision!r}") from e
    except RepositoryNotFoundError as e:
        raise LookupError(f"There is no Hugging Face model repo {repo} (or it is private or gated)") from e


def _hf_checked(repo: str, revision: Optional[str], request: _Request) -> Resolved:
    how = "a cellmap-models export on Hugging Face"
    if not request.online:
        return _huggingface(repo, revision, how, request,
                            notes=["not checked (offline): taken to be a cellmap-models export"])
    try:
        files = _hf_files(repo, revision)
    except LookupError as e:
        raise ValueError(str(e)) from None
    except Exception as e:
        return _huggingface(repo, revision, how, request,
                            notes=[f"could not reach Hugging Face to check it ({e}): taken to be a "
                                   "cellmap-models export"])
    if EXPORT_METADATA not in files:
        raise ValueError(
            f"{repo} is on Hugging Face but is not a cellmap-models export (it has no {EXPORT_METADATA}), "
            "so cellmap-flow cannot serve it as it is. Wrap it in a Python script that loads it and "
            "says its geometry (type: script; see the custom script docs), and add that script."
        )
    if EXPORT_MODEL not in files:
        raise ValueError(
            f"{repo} has a {EXPORT_METADATA} but no {EXPORT_MODEL}, which the huggingface type serves"
        )
    return _huggingface(repo, revision, how, request)


def _huggingface(repo, revision, how, request, notes=()) -> Resolved:
    if not re.match(r"^[^/\s]+/[^/\s]+$", repo):
        raise ValueError(f"{repo!r} is not a Hugging Face repo id (org/repo)")
    params = {"repo": repo}
    if revision:
        params["revision"] = revision
    return _resolved("huggingface", params, repo.split("/")[-1], how, request, notes=notes)


# --- 6. a zoo id or nickname -------------------------------------------------------------------


def _zoo_entries() -> List[dict]:
    """The zoo's models, as the Models tab lists them (bioimage.io's server,
    else the legacy index; cached), fetched again when that list is stale.

    Raises:
        Exception: the list cannot be read.
    """
    from cellmap_flow.models import bioimage_catalog

    document = bioimage_catalog.list_bioimage_models()
    if document.get("stale"):
        try:
            document = bioimage_catalog.refresh_bioimage_models()
        except bioimage_catalog.ZooIndexError:
            pass  # the stale list it is
    return document["models"]


def _zoo_id(ref: str, request: _Request) -> Optional[Resolved]:
    shaped = bool(_NICKNAME.match(ref) or _DOI.match(ref))
    how = "a BioImage Model Zoo model"
    if not request.online:
        if not shaped:
            return None
        return _bioimage(ref, _bioimage_name(ref), how, request,
                         notes=["not checked (offline): taken to be a BioImage Model Zoo id or nickname"])
    # Anything one word long may be a zoo id; nothing else is.
    if not shaped and ("/" in ref or any(c.isspace() for c in ref)):
        return None
    try:
        entries = _zoo_entries()
    except Exception as e:
        if not shaped:
            return None
        return _bioimage(ref, _bioimage_name(ref), how, request,
                         notes=[f"could not read the zoo's model list to check it ({e}): taken to be a zoo model"])
    wanted = ref.lower()
    for entry in entries:
        if wanted in (str(entry.get(k) or "").lower() for k in ("id", "nickname", "key", "concept_doi")):
            name = entry.get("nickname") or _bioimage_name(str(entry.get("id") or ref))
            return _bioimage(ref, name, how, request)
    if shaped:
        raise ValueError(f"{ref!r} is not a model in the BioImage Model Zoo (by id, nickname or DOI)")
    return None
