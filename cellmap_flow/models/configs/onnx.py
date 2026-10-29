"""``OnnxModelConfig``: an ONNX export as the browser mode runs it, served with ONNX Runtime.

The model is one of:

- ``path``: a local ``.onnx`` file. A model finetuned in the browser (the
  file its *Download finetuned model* saves) carries its manifest in the
  file's ``metadata_props``, under ``cellmap_flow.manifest`` (MANIFEST_KEY),
  so the file alone says its voxel sizes, block sizes and channels. Any
  other file says none of that: the entry gives ``input_voxel_size``, and the
  block sizes and the channel count are read from the graph's input and
  output shapes when they are fixed, else from ``input_size``,
  ``output_size`` and ``channels``. A relative ``path`` in a YAML is taken
  from the YAML file's folder (``config_relative_paths``), so an exported
  YAML can name the downloaded file beside it.
- ``folder``: one model of a Hugging Face repo of web exports, ``repo``
  (cellmap/onnx-web-models by default) at ``revision``: a folder with a
  ``manifest.json`` and one ``.onnx`` per variant, a precision and a block
  size. ``file`` names the variant's file (what the browser's Export Config
  writes: the one it ran); without it, ``variant`` (fp32 by default) and
  ``input_size`` (by default the largest block) choose one. Both files are
  downloaded with huggingface_hub, as the huggingface type's are.

A manifest's fields are web/src/models/manifest.ts's, which ``parse_manifest``
reads the same way: the voxel sizes in nm, the channels, ``output`` (what the
output is: the exports' "raw logits"), ``input_norm`` (cellmap-flow steps),
``io`` (the input and output tensors' names) and the variants. Anything the
entry gives (voxel sizes, sizes, channels, output) wins over the manifest.

Each forward is what the browser feeds (web/src/infer/stages.ts): one block
of float32 ``[1, 1, z, y, x]`` at the input size, normalized by the chain,
in and ``[1, C, z', y', x']`` out (``[1, z', y', x']`` is one channel); an
fp16 export whose graph takes float16 is given float16, and its output is
served as float32. A 2D student (input and output one slice thick, such as
setup_53_2d) is given one slice a call, ``slices_per_chunk`` slices a chunk,
as the bioimage type runs a 2D model. The session is wrapped in a torch
module (``_onnx_module``), so that the inference runner serves it as it
serves any network: the warmup forward checks the declared shapes against
the graph's. What its output is, the runner reads off the file's graph
(``output_graph``, ``read_graph``): an export's last Conv says logits, so
the dashboard suggests a SigmoidPostprocessor (or the affinity chain for
affinities), as for any model whose output is logits. The price is one more
copy each way: the runner puts a block on the device, and the module hands
it to ONNX Runtime from the host (148³ float32 is 13 MB, milliseconds a
block next to the forward's tenths of a second).

The manifest's ``input_norm`` is what the browser feeds the model through;
here, as for every type, the session's chain (a YAML's json_data) does the
normalizing. ``declared_input_norm`` gives it to the dashboard's model
advice, which offers it with an Apply button, and the browser's Export
Config writes the chain the page ran with.

Not finetunable: the finetune trainer trains a torch module, with LoRA's
adapters or in full, and an ONNX graph is not one; its weights are tensors
in a protobuf, and no backward pass runs through ONNX Runtime's inference
sessions. The browser's own finetune trains an export's last layer, and
its result is served here.

``onnxruntime`` is imported only when the model is built: the CLIs and the
dashboard's model form import every type. The server runs in pixi.toml's
``onnx`` environment (onnxruntime-gpu on Linux, onnxruntime on the Mac)
unless the entry names another ``env``. The manifest is read without it
(``read_metadata``, ``read_manifest``), so ``cellmap_flow add`` and the
dashboard's *Add a model* can say what a file is.
"""

import functools
import io
import json
import logging
import os
import re
import struct
import time
from typing import Dict, Optional

import numpy as np

from cellmap_flow.models import entry
from cellmap_flow.models.configs.base import Config, ModelConfig
from cellmap_flow.models.geometry import _numbers
from cellmap_flow.models.resolve import ONNX_REPO

logger = logging.getLogger(__name__)

# The metadata_props key a model finetuned in the browser carries its
# manifest under, as JSON (web/src/contracts.ts MANIFEST_METADATA_KEY).
MANIFEST_KEY = "cellmap_flow.manifest"
# web/src/models/manifest.ts's VARIANT_NAMES, in its order: a variant of
# another precision is left out, as the browser leaves it out.
VARIANTS = ("fp32", "fp16", "int8w", "int8w_fp16")
DEFAULT_VARIANT = "fp32"
# As the bioimage type gives a 2D model: z slices a chunk, each its own call.
DEFAULT_SLICES_PER_CHUNK = 8
# How long a failed manifest download is remembered before declared_input_norm
# tries again, as HuggingFaceModelConfig remembers its metadata.json's: the
# dashboard's model advice asks on every poll.
MANIFEST_RETRY_SECONDS = 60


# --- the file's metadata and graph, without onnx or onnxruntime -----------------

def _varint(f) -> int:
    """The varint at ``f``'s position, which is then past it."""
    value, shift = 0, 0
    while True:
        byte = f.read(1)
        if not byte:
            raise ValueError("a varint runs past the end of the file")
        value |= (byte[0] & 0x7F) << shift
        if not byte[0] & 0x80:
            return value
        shift += 7
        if shift > 63:
            raise ValueError("a varint is longer than ten bytes")


def _fields(f, end, where):
    """Each field of the protobuf message from ``f``'s position to ``end``, as
    ``(number, wire type, value)``: a run of fields, each a tag (its number
    and wire type) and its value, a varint, 4 or 8 bytes, or a
    length-prefixed run of bytes. For the last (wire type 2) ``value`` is the
    run's length and ``f`` is at its first byte, for the caller to read or
    not: the next field is read from past it either way, so a 1 GB
    initializer is stepped over unread.

    Raises:
        ValueError: what is there is not a protobuf message, so not an ONNX
            model (a failed download's HTML page, say); ``where`` names it.
    """
    while f.tell() < end:
        tag = _varint(f)
        number, wire = tag >> 3, tag & 7
        if number == 0:
            raise ValueError(f"{where} is not an ONNX model (a field numbered 0)")
        if wire == 0:
            yield number, wire, _varint(f)
        elif wire in (1, 5):  # a fixed 64- or 32-bit value
            yield number, wire, f.read(8 if wire == 1 else 4)
        elif wire == 2:
            length = _varint(f)
            start = f.tell()
            if start + length > end:
                raise ValueError(f"{where} is not an ONNX model (field {number} runs past its end)")
            yield number, wire, length
            f.seek(start + length)
        else:
            raise ValueError(f"{where} is not an ONNX model (wire type {wire})")


def _string_entry(message: bytes):
    """(key, value) of a StringStringEntryProto: field 1 the key, field 2 the value."""
    entry, f = {}, io.BytesIO(message)
    while f.tell() < len(message):
        tag = _varint(f)
        if tag & 7 != 2:
            raise ValueError("a metadata_props entry holds a field that is not a string")
        length = _varint(f)
        entry[tag >> 3] = f.read(length).decode("utf-8")
    return entry.get(1, ""), entry.get(2, "")


def read_metadata(path) -> Dict[str, str]:
    """The ``metadata_props`` of the ONNX model at ``path`` (ModelProto field 14), by key.

    Read without onnx or onnxruntime, so the dashboard's environment can
    read it: each of the ModelProto's fields is stepped over (``_fields``)
    but the metadata's entries; the graph (field 7), nearly all of the file,
    is never read.

    Raises:
        ValueError: the file is not a protobuf message with a graph, so not
            an ONNX model (a failed download's HTML page, say).
    """
    metadata, graph = {}, False
    with open(path, "rb") as f:
        for number, wire, value in _fields(f, os.path.getsize(path), path):
            if wire == 2 and number == 14:
                key, text = _string_entry(f.read(value))
                metadata[key] = text
            graph |= wire == 2 and number == 7
    if not graph:
        raise ValueError(f"{path} is not an ONNX model: it has no graph")
    return metadata


# A TensorProto read for its value: one number, its name and its type are a
# few dozen bytes. A longer one (a weight) is stepped over.
SCALAR_TENSOR_BYTES = 256
# TensorProto.DataType's numbers of the types a scalar is read in, as struct formats.
_SCALAR_TYPES = {1: "<f", 6: "<i", 7: "<q", 10: "<e", 11: "<d"}


def _tensor(message: bytes, where):
    """(name, its one number or None) of a TensorProto: field 8 the name, 2
    the type, 9 the raw bytes, 4, 5, 7 and 10 the typed values."""
    f, name, kind, value = io.BytesIO(message), "", None, None
    for number, wire, field in _fields(f, len(message), where):
        if number == 8 and wire == 2:
            name = f.read(field).decode("utf-8")
        elif number == 2 and wire == 0:
            kind = field
        elif number == 9 and wire == 2:
            raw = f.read(field)
            fmt = _SCALAR_TYPES.get(kind)
            if fmt is not None and len(raw) == struct.calcsize(fmt):
                value = struct.unpack(fmt, raw)[0]
        elif number == 4 and wire == 5:
            value = struct.unpack("<f", field)[0]
        elif number == 10 and wire == 1:
            value = struct.unpack("<d", field)[0]
        elif number in (4, 10) and wire == 2 and field in (4, 8):  # one packed float or double
            value = struct.unpack("<f" if field == 4 else "<d", f.read(field))[0]
        elif number in (5, 7) and wire == 0:
            value = field
        elif number in (5, 7) and wire == 2:  # packed ints: the first
            value = _varint(f)
    return name, None if value is None else float(value)


def _node(message: bytes, where):
    """A NodeProto as portable.activation's node, with its attributes'
    numbers (a Constant's value, an old Clip's min and max) under
    ``attributes``: field 1 its inputs, 2 its outputs, 4 its op_type, 7 its
    domain, 5 its AttributeProtos (1 the name, 2 a float, 3 an int, 5 a
    tensor)."""
    f = io.BytesIO(message)
    node, domain, attributes = {"op": "", "inputs": [], "outputs": []}, "", {}
    for number, wire, field in _fields(f, len(message), where):
        if wire != 2:
            continue
        if number in (1, 2):
            node["inputs" if number == 1 else "outputs"].append(f.read(field).decode("utf-8"))
        elif number == 4:
            node["op"] = f.read(field).decode("utf-8")
        elif number == 7:
            domain = f.read(field).decode("utf-8")
        elif number == 5:
            attribute = f.read(field)
            g, key, value = io.BytesIO(attribute), "", None
            for n, w, v in _fields(g, len(attribute), where):
                if n == 1 and w == 2:
                    key = g.read(v).decode("utf-8")
                elif n == 2 and w == 5:
                    value = struct.unpack("<f", v)[0]
                elif n == 3 and w == 0:
                    value = v
                elif n == 5 and w == 2:
                    value = _tensor(g.read(v), where)[1]
            attributes[key] = value
    if domain not in ("", "ai.onnx"):
        # Another domain's op (com.microsoft's) is not the standard op of its name.
        node["op"] = f"{domain}.{node['op']}"
    node["attributes"] = attributes
    return node


def read_graph(path):
    """The graph of the ONNX model at ``path``, as ``portable.activation``
    reads it: ``(nodes, outputs)``, each node ``{"op" (its op_type),
    "inputs", "outputs"}``, a Clip's with ``min`` and ``max`` where they are
    constants, and the graph's output names.

    Read without onnx, as read_metadata reads the metadata (the onnx
    environment has onnxruntime, not onnx): the ModelProto's graph (field
    7) is a GraphProto, whose nodes (field 1) and outputs (field 12, a
    ValueInfoProto named by its field 1) are read, and whose initializers
    (field 5), the weights, are stepped over unless one is a single number
    (SCALAR_TENSOR_BYTES), as a Clip's bounds are.

    Raises:
        ValueError: the file is not an ONNX model.
    """
    nodes, outputs, numbers = [], [], {}
    with open(path, "rb") as f:
        for number, wire, length in _fields(f, os.path.getsize(path), path):
            if number != 7 or wire != 2:
                continue
            for field, kind, size in _fields(f, f.tell() + length, path):
                if kind != 2:
                    continue
                if field == 1:
                    nodes.append(_node(f.read(size), path))
                elif field == 12:
                    info = f.read(size)
                    g = io.BytesIO(info)
                    outputs += [g.read(v).decode("utf-8") for n, w, v in _fields(g, len(info), path) if n == 1 and w == 2]
                elif field == 5 and size <= SCALAR_TENSOR_BYTES:
                    name, value = _tensor(f.read(size), path)
                    numbers[name] = value
    for node in nodes:
        if node["op"] == "Constant" and node["outputs"]:
            attributes = node["attributes"]
            numbers[node["outputs"][0]] = next((attributes[k] for k in ("value", "value_float", "value_int")
                                                if attributes.get(k) is not None), None)
    for node in nodes:
        attributes = node.pop("attributes")
        if node["op"] == "Clip":
            # Its bounds: inputs 2 and 3 since opset 11, attributes before.
            for i, key in enumerate(("min", "max")):
                value = attributes.get(key, numbers.get(node["inputs"][i + 1]) if len(node["inputs"]) > i + 1 else None)
                if value is not None:
                    node[key] = float(value)
        # "" is an optional input left out, held till here for the Clip's places.
        node["inputs"] = [name for name in node["inputs"] if name]
    return nodes, outputs


def read_manifest(path) -> Optional[dict]:
    """The cellmap-flow manifest the ONNX file at ``path`` carries (MANIFEST_KEY), parsed; None without one.

    Raises:
        ValueError: the file is not an ONNX model, or its manifest is not one.
    """
    text = read_metadata(path).get(MANIFEST_KEY)
    if text is None:
        return None
    try:
        raw = json.loads(text)
    except ValueError as e:
        raise ValueError(f"{path}'s {MANIFEST_KEY} is not JSON: {e}") from None
    return parse_manifest(raw, f"{path}'s {MANIFEST_KEY}")


# --- the manifest -----------------------------------------------------------------

def _positive_numbers(raw, key, where):
    value = raw.get(key)
    if (not isinstance(value, list) or len(value) != 3
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0 for v in value)):
        raise ValueError(f"{where}: {key} is not three positive numbers (z, y, x)")
    return list(_numbers(value))


def _block(value, where) -> list:
    """A variant's block size as z, y, x voxels: one number is a cube, as the manifests write them."""
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return [value] * 3
    if isinstance(value, list) and len(value) == 3 and all(isinstance(v, int) and v > 0 for v in value):
        return list(value)
    raise ValueError(f"{where}: expected a positive whole number or three (z, y, x), got {value!r}")


def _io(value) -> dict:
    """The input and output tensors' names: an object, or the exports' sentence with the names in quotes."""
    if isinstance(value, dict):
        return {"input": value.get("input") or "input", "output": value.get("output") or "output"}
    text = value if isinstance(value, str) else ""
    found = {k: re.search(rf"{k}\s+'([^']+)'", text) for k in ("input", "output")}
    return {k: m[1] if m else k for k, m in found.items()}


def parse_manifest(raw, where="manifest.json") -> dict:
    """``raw``, a manifest's JSON, checked and widened as the browser's ``parseManifest`` does.

    The exports write a variant's block as one number (a cube) and ``io`` as
    a sentence; here they are three numbers and ``{"input", "output"}``. A
    variant of a precision not in VARIANTS is left out.

    Raises:
        ValueError: a field is missing or of the wrong kind; it says which.
    """
    if not isinstance(raw, dict):
        raise ValueError(f"{where} is not a JSON object")
    channels = raw.get("channels")
    if not isinstance(channels, list) or not channels or not all(isinstance(c, str) for c in channels):
        raise ValueError(f"{where}: channels is not a list of names")
    input_voxel_size = _positive_numbers(raw, "input_voxel_size_nm", where)
    output_voxel_size = _positive_numbers(raw, "output_voxel_size_nm", where)
    variants = []
    for i, v in enumerate(raw.get("variants") or []):
        if not isinstance(v, dict) or v.get("variant") not in VARIANTS:
            continue
        if not isinstance(v.get("file"), str) or not v["file"]:
            raise ValueError(f"{where}: variants[{i}] names no file")
        variants.append({
            "variant": v["variant"],
            "input_size": _block(v.get("input_size"), f"{where} variants[{i}].input_size"),
            "output_size": _block(v.get("output_size"), f"{where} variants[{i}].output_size"),
            "file": v["file"],
        })
    if not variants:
        raise ValueError(f"{where}: no variant of a known precision ({', '.join(VARIANTS)})")
    steps = raw.get("input_norm") or []
    if not isinstance(steps, list) or not all(isinstance(s, dict) and s.get("name") for s in steps):
        raise ValueError(f"{where}: input_norm is not a list of steps, each with a name")
    return {
        "model": raw.get("model") or "",
        "channels": list(channels),
        "input_voxel_size": input_voxel_size,
        "output_voxel_size": output_voxel_size,
        "output": raw.get("output") if isinstance(raw.get("output"), str) else "",
        "input_norm": [dict(s) for s in steps],
        "io": _io(raw.get("io")),
        "variants": variants,
    }


def _matches(block, size) -> bool:
    """Whether a variant's input ``block`` is ``size``: three numbers, or one, a cube's or a slice's side."""
    if len(size) == 1:
        return block in ([size[0]] * 3, [1, size[0], size[0]])
    return list(block) == list(size)


def choose_variant(manifest, file=None, variant=None, input_size=None) -> dict:
    """The manifest's variant to serve: the one in ``file``, else of precision
    ``variant`` (fp32 by default, else the first there is) with an input of
    ``input_size`` voxels, by default the largest block.

    Raises:
        ValueError: no variant is the one asked for; the message lists them.
    """
    variants = manifest["variants"]
    listed = ", ".join(f"{v['file']} ({v['variant']}, in {'x'.join(map(str, v['input_size']))})" for v in variants)
    if file is not None:
        for v in variants:
            if v["file"] == file:
                return v
        raise ValueError(f"The manifest has no variant in {file}; it has {listed}")
    precisions = [v["variant"] for v in variants]
    if variant is None:
        variant = DEFAULT_VARIANT if DEFAULT_VARIANT in precisions else precisions[0]
    candidates = [v for v in variants if v["variant"] == variant]
    if input_size is not None:
        candidates = [v for v in candidates if _matches(v["input_size"], input_size)]
    if not candidates:
        wanted = variant + (f" with an input of {'x'.join(map(str, input_size))}" if input_size else "")
        raise ValueError(f"The manifest has no {wanted} variant; it has {listed}")
    return max(candidates, key=lambda v: int(np.prod(v["input_size"])))


# --- serving ------------------------------------------------------------------------

def _fixed(shape) -> list:
    """A graph tensor's dims, None where it is not a fixed whole number (a symbolic or unknown dim)."""
    return [d if isinstance(d, int) and d > 0 else None for d in (shape or [])]


@functools.lru_cache(maxsize=None)
def _onnx_module():
    """The torch module that runs an ONNX Runtime session, defined on first use:
    this module must not import torch, since the CLIs import every model type."""
    import torch

    class OnnxModule(torch.nn.Module):
        """An inference session as a torch module: ``(B, 1, Z, Y, X)`` in, ``(B, C, Z', Y', X')`` out.

        Each image of the batch is one call, ``[1, 1, z, y, x]`` as the
        browser feeds it, or one call a z slice for a 2D student
        (``slice_wise``). The input goes to the session as the graph's input
        type and the output comes back as float32, on the input's device.
        No parameters: there is nothing here to move, to train or to cast.
        """

        def __init__(self, session, input_name, output_name, input_dtype, slice_wise):
            super().__init__()
            self.session = session
            self.input_name, self.output_name = input_name, output_name
            self.input_dtype = input_dtype
            self.slice_wise = slice_wise

        def forward(self, x):
            array = x.detach().to("cpu", torch.float32).numpy()
            if array.ndim != 5 or array.shape[1] != 1:
                raise ValueError(f"an ONNX model takes (batch, 1, z, y, x), not {tuple(array.shape)}")
            images = []
            for image in array:
                calls = [image[None, :, z:z + 1] for z in range(image.shape[1])] if self.slice_wise else [image[None]]
                images.append(np.concatenate([self._run(block) for block in calls], axis=2))
            return torch.from_numpy(np.concatenate(images, axis=0)).to(x.device)

        def _run(self, block):
            feed = {self.input_name: np.ascontiguousarray(block, dtype=self.input_dtype)}
            (out,) = self.session.run([self.output_name], feed)
            out = np.asarray(out, dtype=np.float32)
            if out.ndim == 4:  # [1, z, y, x]: one channel
                out = out[:, None]
            if out.ndim != 5 or out.shape[0] != 1:
                raise ValueError(f"the model wrote {list(out.shape)}, not [1, C, z, y, x]")
            return out

    return OnnxModule


def _session(path, label):
    """An onnxruntime session of the model at ``path``: on CUDA where the
    runner serves on a GPU and onnxruntime has its CUDA provider, else on
    the CPU, saying which and why.

    torch is imported before it: onnxruntime-gpu's CUDA provider loads
    cuDNN and cuBLAS by name, and the torch wheels' nvidia packages, which
    torch loads when it is imported, are where the environment has them.

    onnxruntime sends usage events to Microsoft unless told not to (1.31
    uploads to mobile.events.data.microsoft.com from a thread it starts on
    import), and an inference server on the cluster has no business doing
    so. ORT_DISABLE_TELEMETRY=1 stops it, and only when it is set before the
    import (disable_telemetry_events afterwards does not); it is set here
    unless the environment sets it already.
    """
    os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
    import onnxruntime as ort

    from cellmap_flow.utils.device import best_device

    if hasattr(ort, "disable_telemetry_events"):  # imported before this ran
        ort.disable_telemetry_events()
    device = best_device()
    available = ort.get_available_providers()
    cuda = device.type == "cuda" and "CUDAExecutionProvider" in available
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if cuda else ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(path), providers=providers)
    used = session.get_providers()
    if cuda and "CUDAExecutionProvider" not in used:
        logger.warning(f"{label}: onnxruntime could not start its CUDA provider (see its log above); "
                       "running on the CPU, far slower")
    elif device.type == "cuda" and not cuda:
        logger.warning(f"{label}: this onnxruntime has no CUDA provider ({', '.join(available)}); running on "
                       "the CPU, far slower. pixi.toml's onnx environment has onnxruntime-gpu.")
    logger.info(f"{label}: onnxruntime {ort.__version__} on {', '.join(used)}")
    return session


class OnnxModelConfig(ModelConfig):
    """An ONNX export, the browser mode's kind of model, served with ONNX Runtime.

    Args:
        path: a local .onnx file. Relative in a YAML: from the YAML's folder.
        folder: a model's folder in ``repo`` (setup_54), instead of ``path``.
        repo: the Hugging Face repo of ``folder``; cellmap/onnx-web-models.
        revision: the repo's branch, tag or commit; its main branch.
        file: the ``.onnx`` of ``folder`` to serve: one of its manifest's
            variants (what the browser's Export Config writes).
        variant: the precision to serve when ``file`` is not given: fp32
            (the default), fp16, int8w or int8w_fp16.
        input_voxel_size: nm per input voxel (one number, or z, y, x). The
            manifest's; needed for a file without one.
        output_voxel_size: nm per output voxel: by default the input's times
            the manifest's ratio of the two, else the input's.
        input_size: voxels of the block the model reads (one number or z, y,
            x): the manifest's variant's, else the graph's input. For a
            ``folder``, it chooses the variant (the largest by default).
        output_size: voxels it writes: the variant's, else the graph's output.
        channels: the output channels' names; the manifest's, else
            channel_0, channel_1, ... for as many as the graph outputs.
        output: what the output is, as the manifests say it ("raw logits").
        slices_per_chunk: z slices a chunk of a 2D student (a model whose
            input and output are one slice thick), each one call; 8 by
            default. A 3D model ignores it.
    """

    cli_name = "onnx"
    # onnxruntime is in none of cellmap-flow's other environments but
    # bioimageio's, and there without its CUDA provider; pixi.toml's onnx
    # environment has onnxruntime-gpu.
    default_env = "onnx"
    # The trainer trains torch modules; see the module's docstring.
    finetunable = False
    # A YAML entry's relative path is the YAML's folder's (registry.build_model).
    config_relative_paths = ("path",)

    def __init__(
        self,
        path: str = None,
        folder: str = None,
        repo: str = None,
        revision: str = None,
        file: str = None,
        variant: str = None,
        input_voxel_size: tuple = None,
        output_voxel_size: tuple = None,
        input_size: tuple = None,
        output_size: tuple = None,
        channels: list[str] = None,
        output: str = None,
        slices_per_chunk: int = None,
        name: str = None,
        scale=None,
    ):
        super().__init__()
        if (path is None) == (folder is None):
            raise ValueError("An onnx model is a path (a local .onnx file) or a folder of a Hugging Face repo "
                             f"({ONNX_REPO} by default), and not both")
        if path is not None and (repo is not None or revision is not None or file is not None):
            raise ValueError("repo, revision and file name a Hugging Face model's file; a path is the file itself")
        if variant is not None and variant not in VARIANTS:
            raise ValueError(f"variant must be one of {', '.join(VARIANTS)}, not {variant!r}")
        self.path = None if path is None else os.path.abspath(os.path.expanduser(str(path)))
        self.folder = None if folder is None else str(folder).strip("/")
        self.repo = repo
        self.revision = revision
        self.file = file
        self.variant = variant
        self.input_voxel_size = entry.voxel_size(input_voxel_size, "input_voxel_size")
        self.output_voxel_size = entry.voxel_size(output_voxel_size, "output_voxel_size")
        self.input_size = entry.sizes(input_size, min=1, lengths=(1, 3), what="input_size")
        self.output_size = entry.sizes(output_size, min=1, lengths=(1, 3), what="output_size")
        self.channels = entry.names(channels)
        self.output = output
        self.slices_per_chunk = None if slices_per_chunk is None else int(slices_per_chunk)
        if self.slices_per_chunk is not None and self.slices_per_chunk < 1:
            raise ValueError(f"slices_per_chunk must be at least 1, got {self.slices_per_chunk}")
        if name is None:
            name = self.folder if self.folder else os.path.splitext(os.path.basename(self.path))[0]
        self.name = name
        self.scale = scale
        self._manifest_cache = None
        self._manifest_failed_at = None

    # --- what the model is ---------------------------------------------------------

    @property
    def hub_repo(self) -> str:
        return self.repo or ONNX_REPO

    def _hub_download(self, filename: str) -> str:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(self.hub_repo, f"{self.folder}/{filename}", revision=self.revision)

    def _manifest(self) -> Optional[dict]:
        """The model's manifest, parsed: the folder's manifest.json, or the
        file's own; None for a file without one. Kept once read."""
        if self._manifest_cache is None:
            if self.folder is not None:
                where = f"{self.hub_repo}/{self.folder}/manifest.json"
                with open(self._hub_download("manifest.json")) as f:
                    self._manifest_cache = parse_manifest(json.load(f), where)
            else:
                self._manifest_cache = read_manifest(self.path) or {}
        return self._manifest_cache or None

    def _declared_manifest(self) -> Optional[dict]:
        """The manifest ({} for a file without one) for what the dashboard
        asks of it on every poll, or None when it cannot be read: a failed
        download is not tried again for MANIFEST_RETRY_SECONDS."""
        if self._manifest_failed_at is not None and time.monotonic() - self._manifest_failed_at < MANIFEST_RETRY_SECONDS:
            return None
        try:
            return self._manifest() or {}
        except Exception as e:
            self._manifest_failed_at = time.monotonic()
            logger.info(f"Could not read {self.name}'s manifest ({e}); trying again after {MANIFEST_RETRY_SECONDS} s")
            return None

    def declared_input_norm(self, data_range=None) -> Optional[list]:
        """The input normalization the model's manifest declares, as
        cellmap-flow steps (``[{"name": ..., **params}]``), or None. The
        manifest gives each step's values, so ``data_range`` is not used.

        For the dashboard's model advice (``suggest_input_norm``'s
        ``declared``), which asks on every poll (``_declared_manifest``).
        """
        return (self._declared_manifest() or {}).get("input_norm") or None

    def declared_geometry(self) -> Optional[dict]:
        """``ModelConfig.declared_geometry`` from the entry, the manifest and its
        variant, as the server works it out (``_geometry``) but without the
        graph: None when they leave a size to the graph, or the manifest
        cannot be read now (``_declared_manifest``)."""
        manifest = self._declared_manifest()
        if manifest is None:
            return None
        geometry = Config()
        try:
            self._geometry(geometry, manifest, self._variant(manifest))
        except (TypeError, ValueError):
            return None
        return {"write_shape": list(geometry.write_shape), "input_voxel_size": list(geometry.input_voxel_size),
                "output_voxel_size": list(geometry.output_voxel_size), "output_channels": geometry.output_channels,
                "channels": list(geometry.channels)}

    def output_graph(self):
        """The served file's graph (``read_graph``), read back from the
        output it is fetched by; the manifest's ``output`` words are not
        consulted, as the graph says it for certain."""
        config = self.config
        nodes, _ = read_graph(config.onnx_file)
        return nodes, config.output_name

    def _variant(self, manifest) -> Optional[dict]:
        """The manifest's variant this model serves: the folder's chosen by
        file, variant and input size; the file's own by its name, else the
        only one, else chosen as a folder's."""
        if not manifest:
            return None
        if self.folder is not None:
            return choose_variant(manifest, self.file, self.variant, self.input_size)
        own = [v for v in manifest["variants"] if v["file"] == os.path.basename(self.path)]
        if own:
            return own[0]
        if len(manifest["variants"]) == 1:
            return manifest["variants"][0]
        return choose_variant(manifest, None, self.variant, self.input_size)

    def _get_config(self) -> Config:
        manifest = self._manifest()
        variant = self._variant(manifest)
        path = self._hub_download(variant["file"]) if self.folder is not None else self.path
        label = self.name or os.path.basename(path)
        session = _session(path, label)
        config = Config()
        self._describe(config, manifest or {}, variant, session)
        config.model = _onnx_module()(session, config.input_name, config.output_name, config.input_dtype,
                                      config.slice_wise)
        config.onnx_file = path  # what output_graph reads
        output = config.output_kind
        if "logit" in output.lower():
            logger.info(f"{label}: its output is {output!r}. For probabilities add a SigmoidPostprocessor to the "
                        "postprocessing (an affinity model: the affinity chain the dashboard suggests)")
        return config

    def _describe(self, config, manifest, variant, session):
        """Set ``config`` from the entry, the manifest and its variant, and the
        graph's input and output: its geometry (``_geometry``), and the
        tensors the model is fed and fetched by."""
        what = self.name or "the model"
        inputs, outputs = session.get_inputs(), session.get_outputs()
        # The manifest's tensor names, as the browser feeds and fetches by
        # them; else the graph's first input and output.
        names = manifest.get("io") or {}
        input_name = names.get("input") if names.get("input") in [i.name for i in inputs] else inputs[0].name
        output_name = names.get("output") if names.get("output") in [o.name for o in outputs] else outputs[0].name
        graph_in = next(i for i in inputs if i.name == input_name)
        graph_out = next(o for o in outputs if o.name == output_name)
        in_dims, out_dims = _fixed(graph_in.shape), _fixed(graph_out.shape)
        if len(in_dims) != 5 or in_dims[0] not in (1, None) or in_dims[1] not in (1, None):
            raise ValueError(
                f"{what}'s input {input_name} is {graph_in.shape}; cellmap-flow feeds an ONNX model [1, 1, z, y, x], "
                "as the browser does. Serve another layout with a script model (type: script)"
            )
        if len(out_dims) not in (4, 5):
            raise ValueError(f"{what}'s output {output_name} is {graph_out.shape}, not [1, C, z, y, x]")
        graph_channels = out_dims[1] if len(out_dims) == 5 else 1
        self._geometry(config, manifest, variant, in_dims[2:], out_dims[-3:], graph_channels)
        config.output_dtype = np.float32
        config.output_kind = self.output or manifest.get("output") or ""
        config.input_name, config.output_name = input_name, output_name
        config.input_dtype = np.float16 if graph_in.type == "tensor(float16)" else np.float32

    def _geometry(self, config, manifest, variant, graph_in=(None,) * 3, graph_out=(None,) * 3, graph_channels=None):
        """Set ``config``'s geometry from the entry, the manifest and its
        variant, and the graph's input and output sizes (z, y, x voxels, None
        where not fixed) and channel count, each where the one before says
        nothing; check that they agree with the graph. Without the graph's
        (declared_geometry), the others must say it all.

        Raises:
            ValueError: none of them says a size, or they disagree.
        """
        what = self.name or "the model"
        variant = variant or {}

        def first(given, *found, what_is_it):
            for value in (given, *found):
                if value is not None and (np.ndim(value) == 0 or all(v is not None for v in value)):
                    return value
            raise ValueError(f"{what} does not say its {what_is_it}: give it in the entry")

        in_voxel = first(self.input_voxel_size, manifest.get("input_voxel_size"), what_is_it="input_voxel_size")
        if self.output_voxel_size is not None:
            out_voxel = self.output_voxel_size
        elif manifest.get("output_voxel_size"):
            # The manifest's ratio of output to input, at the input voxel
            # size the entry may have changed: a 16 nm model run at 8 nm
            # writes 8 nm voxels, as it wrote 16.
            out_voxel = (np.asarray(in_voxel, dtype=float) * np.asarray(manifest["output_voxel_size"], dtype=float)
                         / np.asarray(manifest["input_voxel_size"], dtype=float))
        else:
            out_voxel = in_voxel
        # In the folder's case input_size chose the variant; a file's
        # manifest's variant is overruled by the entry's sizes.
        given_in = self.input_size if self.folder is None else None
        in_size = first(entry.per_axis(given_in, 3, "input_size"), variant.get("input_size"), list(graph_in),
                        what_is_it="input_size")
        out_size = first(entry.per_axis(self.output_size, 3, "output_size"), variant.get("output_size"),
                         list(graph_out), what_is_it="output_size")
        for kind, size, dims in (("input", in_size, list(graph_in)), ("output", out_size, list(graph_out))):
            if any(d is not None and d != s for d, s in zip(dims, size)):
                raise ValueError(f"{what}'s graph has an {kind} of {dims} (z, y, x); the entry or manifest says "
                                 f"{list(size)}")
        channels = self.channels or manifest.get("channels")
        if channels is None:
            if graph_channels is None:
                raise ValueError(f"{what}'s output has no fixed number of channels: give channels in the entry")
            channels = [f"channel_{i}" for i in range(graph_channels)]
        if graph_channels is not None and graph_channels != len(channels):
            raise ValueError(f"{what} outputs {graph_channels} channel(s), and {len(channels)} are named: {channels}")

        slice_wise = in_size[0] == 1 and out_size[0] == 1
        in_voxel = np.asarray(in_voxel, dtype=float)
        out_voxel = np.asarray(out_voxel, dtype=float)
        if slice_wise:
            # Each slice its own call: no context in z, and a slice in is a
            # slice out, at the input's z spacing.
            slices = self.slices_per_chunk or DEFAULT_SLICES_PER_CHUNK
            read, write = (slices, *in_size[1:]), (slices, *out_size[1:])
            out_voxel[0] = in_voxel[0]
        else:
            read, write = tuple(in_size), tuple(out_size)
        read_shape, write_shape = np.asarray(read) * in_voxel, np.asarray(write) * out_voxel
        context = (read_shape - write_shape) / 2
        in_voxels = context / in_voxel
        if (context < 0).any() or not np.allclose(in_voxels, np.rint(in_voxels)):
            # The chunk is read around the written region on the input's
            # grid; half a voxel off, every output would be shifted.
            raise ValueError(
                f"{what}'s output ({list(write)} voxels of {list(_numbers(out_voxel))} nm) is not centred on whole "
                f"voxels of its input ({list(read)} of {list(_numbers(in_voxel))} nm)"
            )

        config.input_voxel_size = _numbers(in_voxel)
        config.output_voxel_size = _numbers(out_voxel)
        config.read_shape = _numbers(read_shape)
        config.write_shape = _numbers(write_shape)
        config.input_size = tuple(int(v) for v in read)
        config.output_size = tuple(int(v) for v in write)
        config.output_channels = len(channels)
        config.channels = list(channels)
        config.block_shape = np.array((*write, len(channels)))
        config.slice_wise = slice_wise

    def to_dict(self):
        """This config as a model entry, which ``registry.build_model`` rebuilds.

        What the manifest or the graph decides unless told otherwise is
        written only when it was given, so that an exported entry keeps
        following the model.
        """
        result = {"type": "onnx"}
        # In the constructor's order, as the server's command gives them.
        for key in ("path", "folder", "repo", "revision", "file", "variant", "input_voxel_size",
                    "output_voxel_size", "input_size", "output_size", "channels", "output", "slices_per_chunk"):
            value = getattr(self, key)
            if value is None:
                continue
            if key in ("input_size", "output_size"):
                value = entry.as_given(value)
            elif isinstance(value, (tuple, list)):
                value = list(value)  # a copy, and no tuple: yaml.safe_load refuses !!python/tuple
            result[key] = value
        return self._with_name_scale(result)
