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
import json
from dataclasses import dataclass
from typing import Optional

from cellmap_flow.utils.web_utils import (
    ARGS_KEY,
    INPUT_NORM_DICT_KEY as INPUT_NORM_KEY,
    POSTPROCESS_DICT_KEY as POSTPROCESS_KEY,
    decode_to_json,
    encode_to_str,
    list_cls_to_dict,
)

__all__ = [
    "INPUT_NORM_KEY",
    "POSTPROCESS_KEY",
    "PipelineSpec",
    "builder_steps",
    "normalize_steps",
    "split_dataset_url",
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


def split_dataset_url(dataset: str) -> Optional[str]:
    """The args blob between a layer URL's two ``ARGS_KEY`` markers.

    ``None`` when the URL carries no blob at all. Any other number of markers
    than two is a malformed URL.
    """
    if ARGS_KEY not in dataset:
        return None
    parts = dataset.split(ARGS_KEY)
    if len(parts) != 3:
        raise ValueError(
            f"Invalid dataset format. Expected two occurrences of {ARGS_KEY}. found {len(parts)} {dataset}"
        )
    return parts[1]


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
    def from_json_data(cls, data) -> "PipelineSpec":
        """From a ``json_data`` dict (or its JSON text); either chain may be
        in the list or the legacy dict form, or missing."""
        if data is None:
            return cls()
        if isinstance(data, str):
            data = json.loads(data)
        return cls(data.get(INPUT_NORM_KEY), data.get(POSTPROCESS_KEY))

    @classmethod
    def from_steps(cls, input_norms=(), postprocess=()) -> "PipelineSpec":
        """From live normalizer/postprocessor instances, via their to_dict()."""
        return cls(
            list_cls_to_dict(input_norms or ()),
            list_cls_to_dict(postprocess or ()),
        )

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
