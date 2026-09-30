"""One description of a normalization + postprocessing chain.

A chain travels as data far more often than it runs: in every layer URL's
args blob (read by inference servers that may be older or newer than the
dashboard), in model YAMLs' ``json_data``, in the finetune manifest, and in
what the dashboard remembers between requests. Each of those used to build
and read the ``{"input_norm": [...], "postprocess": [...]}`` form on its own.
``PipelineSpec`` is that form, with the readers for every shape it arrives
in.

A step is a flat ``{"name": <class name>, **constructor arguments}`` dict,
kept as given: values are not coerced (the dashboard's forms send strings,
and the constructors parse them), and the step is never nested as
``{"name", "params"}`` on the wire, because older servers pass every key
but ``name`` straight to the constructor.

Importing this module is cheap: the op classes (and the libraries some of
them need) are imported only when a spec is built.
"""

import hashlib
import inspect
import json
import re
from dataclasses import dataclass
from typing import Literal, Optional

from cellmap_flow.serving.protocol import (
    INPUT_NORM_KEY,
    POSTPROCESS_KEY,
    decode_to_json,
    encode_to_str,
)

__all__ = [
    "PipelineSpec",
    "builder_steps",
    "chain_is_segmentation",
    "chain_num_channels",
    "chain_output_dtype",
    "normalize_steps",
    "op_schemas",
]

_CHAIN_KEYS = (INPUT_NORM_KEY, POSTPROCESS_KEY)


def _copy_step(step):
    return dict(step) if isinstance(step, dict) else step


def normalize_steps(steps) -> tuple:
    """A chain in the ordered ``({"name", **params}, ...)`` form.

    Accepts the ordered list (copied; each step dict is copied too), the
    older ``{Name: {params}}`` dict (one step per key, in the dict's order),
    or nothing (``None``, ``{}``, ``[]``). In the dict form the key names the
    class, as it did for the reader that dict was written for, so it wins
    over a ``name`` inside the params.

    A list element that is not a dict is kept as it is; the op readers skip
    it with a warning, as before.
    """
    if not steps:
        return ()
    if isinstance(steps, (list, tuple)):
        return tuple(_copy_step(step) for step in steps)
    if isinstance(steps, dict):
        normalized = []
        for name, params in steps.items():
            step = {"name": name, **(params or {})}
            step["name"] = name
            normalized.append(step)
        return tuple(normalized)
    raise ValueError(f"Expected dict or list, got {type(steps)}")


def _step_dicts(ops) -> list:
    """Live normalizers or postprocessors as ``[{name, **params}]`` steps.

    A list, not a dict keyed by class name: two steps of the same class (e.g.
    two LambdaPostprocessors) collapsed into one under a dict. Values keep
    their types -- stringifying them made bool("False") read back as True.
    """
    steps = []
    for n in ops:
        name = n.name()
        elms = dict(n.to_dict())
        if "name" not in elms:
            raise ValueError(f"Normalizer {name} does not have a name key. {elms}")
        elms["name"] = name
        steps.append(elms)
    return steps


def builder_steps(nodes) -> tuple:
    """Steps from the pipeline builder's ``[{"name", "params", ...}]`` nodes.

    Built as ``{**params, "name": name}``, the key order
    ``/api/pipeline/apply`` has always stored. Nodes that are not dicts or
    have no name are skipped; the builder's own validation rejects those
    before they get here.
    """
    steps = []
    for node in nodes or ():
        if not isinstance(node, dict) or not node.get("name"):
            continue
        steps.append({**(node.get("params") or {}), "name": node["name"]})
    return tuple(steps)


@dataclass(frozen=True)
class PipelineSpec:
    """An input-normalization chain and a postprocessing chain, as data.

    Both are tuples of step dicts in the order they run. Build the live op
    instances with ``build()``; those can hold state (SimpleBlockwiseMerger's
    equivalences), so a spec is never a substitute for the instances a
    server or the dashboard already has.
    """

    input_norm: tuple = ()
    postprocess: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "input_norm", normalize_steps(self.input_norm))
        object.__setattr__(self, "postprocess", normalize_steps(self.postprocess))

    @classmethod
    def from_json_data(cls, data, strict: bool = False) -> "PipelineSpec":
        """From a ``json_data`` dict (or its JSON text).

        Either chain may be in the list or the legacy dict form. By default a
        missing or null chain is empty. ``strict`` instead requires both keys
        (KeyError) and a list or dict under each (ValueError), which is what
        the readers before PipelineSpec enforced, and what the blockwise
        precheck relies on to reject a misspelt ``json_data``.
        """
        if data is None and not strict:
            return cls()
        if isinstance(data, str):
            data = json.loads(data)
        if not strict:
            return cls(data.get(INPUT_NORM_KEY), data.get(POSTPROCESS_KEY))
        chains = (data[INPUT_NORM_KEY], data[POSTPROCESS_KEY])
        for chain in chains:
            if not isinstance(chain, (list, dict)):
                raise ValueError(f"Expected dict or list, got {type(chain)}")
        return cls(*chains)

    @classmethod
    def from_steps(cls, input_norms=(), postprocess=()) -> "PipelineSpec":
        """From live normalizer/postprocessor instances, via their to_dict()."""
        return cls(_step_dicts(input_norms or ()), _step_dicts(postprocess or ()))

    @classmethod
    def from_builder(cls, input_normalizers=(), postprocessors=()) -> "PipelineSpec":
        """From the pipeline builder's normalizer and postprocessor nodes."""
        return cls(builder_steps(input_normalizers), builder_steps(postprocessors))

    @classmethod
    def from_url_blob(cls, blob: str):
        """``(spec, extras)`` from a layer URL's args blob.

        ``extras`` is every other key the blob carries (``dashboard_url``,
        ``digest``, and ``time`` from older dashboards).
        """
        data = decode_to_json(blob)
        extras = {k: v for k, v in data.items() if k not in _CHAIN_KEYS}
        return cls.from_json_data(data), extras

    def to_json_data(self) -> dict:
        """``{"input_norm": [...], "postprocess": [...]}``, in that order."""
        return {
            INPUT_NORM_KEY: [_copy_step(step) for step in self.input_norm],
            POSTPROCESS_KEY: [_copy_step(step) for step in self.postprocess],
        }

    def to_url_blob(self, **extras) -> str:
        """The args blob for a layer URL. Extras follow the two chains."""
        clash = [k for k in extras if k in _CHAIN_KEYS]
        if clash:
            raise ValueError(f"extras may not replace a chain: {clash}")
        return encode_to_str({**self.to_json_data(), **extras})

    def digest(self) -> str:
        """A short hash of the chains' content, stable across processes.

        Keys are sorted, so two specs that differ only in the order of a
        step's parameters share a digest; the order of the steps counts.
        """
        text = json.dumps(
            self.to_json_data(), sort_keys=True, separators=(",", ":"), default=str
        )
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def build(self):
        """``(input_norms, postprocess)``: new op instances for both chains."""
        from cellmap_flow.norm.input_normalize import get_normalizations
        from cellmap_flow.post.postprocessors import get_postprocessors

        return (
            get_normalizations(list(self.input_norm)),
            get_postprocessors(list(self.postprocess)),
        )

    def is_empty(self) -> bool:
        return not self.input_norm and not self.postprocess


# --- what a chain of live steps produces ---------------------------------------
#
# Each step says what it does to (dtype, channels) through output_info(), and a
# chain is those steps applied in order. So a later step's declaration wins,
# the same "last step that says" the separate scans in the server, globals and
# the dashboard each implemented.


def _output_info(step, dtype, channels):
    info = getattr(step, "output_info", None)
    if info is not None:
        return info(dtype, channels)
    # Not an op class (a stand-in, say): read the same attributes it would.
    own_dtype = getattr(step, "dtype", None)
    return (
        own_dtype if own_dtype else dtype,
        getattr(step, "num_channels", channels),
        getattr(step, "is_segmentation", None),
    )


def _run_output_info(postprocess, dtype, channels):
    is_segmentation = None
    for step in postprocess or ():
        dtype, channels, step_is_segmentation = _output_info(step, dtype, channels)
        if step_is_segmentation is not None:
            is_segmentation = step_is_segmentation
    return dtype, channels, is_segmentation


def chain_output_dtype(postprocess, model_dtype):
    """The dtype the client receives: the last step's that declares one,
    else the model's own."""
    return _run_output_info(postprocess, model_dtype, None)[0]


def chain_num_channels(postprocess, model_channels) -> int:
    """How many channels the client receives, after e.g. ChannelSelection."""
    return int(_run_output_info(postprocess, None, model_channels)[1])


def chain_is_segmentation(postprocess) -> Optional[bool]:
    """Whether the output is labels: the last step's that says.

    False for an empty chain (raw model output is never labels), None when
    steps are present but none of them says.
    """
    if not postprocess:
        return False
    return _run_output_info(postprocess, None, None)[2]


# --- describing the ops -----------------------------------------------------------

# bool before int: bool is an int subclass, and a flag is not a count.
_JSON_TYPES = (
    (bool, "boolean"),
    (int, "integer"),
    (float, "number"),
    (str, "string"),
    ((list, tuple), "array"),
    (dict, "object"),
)


def _json_type(annotation, default):
    """The JSON Schema type for a constructor argument, or None if unknown.

    The annotation decides when there is one; otherwise the default's type.
    """
    if isinstance(annotation, type):
        for python_type, name in _JSON_TYPES:
            if issubclass(annotation, python_type):
                return name
    if default is not None and default is not inspect.Parameter.empty:
        for python_type, name in _JSON_TYPES:
            if isinstance(default, python_type):
                return name
    return None


def _op_classes(kind):
    if kind == INPUT_NORM_KEY:
        from cellmap_flow.norm.input_normalize import InputNormalizer

        return InputNormalizer.__subclasses__()
    if kind == POSTPROCESS_KEY:
        from cellmap_flow.post.postprocessors import PostProcessor

        return PostProcessor.__subclasses__()
    raise ValueError(f"kind must be {INPUT_NORM_KEY!r} or {POSTPROCESS_KEY!r}, got {kind!r}")


def _title(name):
    # "MinMaxNormalizer" -> "Min Max Normalizer"; "ZScoreNormalizer" keeps "ZScore".
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)


def _jsonable_default(value):
    if isinstance(value, tuple):
        value = list(value)
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return str(value)
    return value


def _op_schema(cls):
    properties = {}
    required = []
    extra = False
    for pname, param in inspect.signature(cls.__init__).parameters.items():
        if pname == "self" or param.kind is inspect.Parameter.VAR_POSITIONAL:
            continue
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            extra = True
            continue
        prop = {"title": pname}
        json_type = _json_type(param.annotation, param.default)
        if json_type:
            prop["type"] = json_type
        if param.default is inspect.Parameter.empty:
            required.append(pname)
        else:
            prop["default"] = _jsonable_default(param.default)
        properties[pname] = prop

    schema = {"type": "object", "title": _title(cls.name())}
    # The class's own docstring only; an inherited one describes the base.
    doc = inspect.cleandoc(cls.__dict__.get("__doc__") or "")
    if doc:
        schema["description"] = doc.split("\n\n")[0]
    schema["properties"] = properties
    schema["required"] = required
    # Older servers pass every key but "name" to the constructor, so an
    # unknown one is an error there.
    schema["additionalProperties"] = extra
    return schema


def op_schemas(kind: Literal["input_norm", "postprocess"]) -> list:
    """A JSON Schema for each registered op's parameters.

    ``[{"name", "title", "schema"}]``, one per op, in the order
    get_input_normalizers() / get_postprocessors_list() list them. The
    schema describes a step's parameters, i.e. its dict without ``name``,
    taken from the constructor's signature: a type from the annotation or
    the default's type, the default, and which arguments are required.
    Values arrive from the dashboard's forms as strings and the constructors
    parse them, so the types say what a value means, not how it must be
    sent.
    """
    schemas = []
    for cls in _op_classes(kind):
        schema = _op_schema(cls)
        schemas.append({"name": cls.name(), "title": schema["title"], "schema": schema})
    return schemas
