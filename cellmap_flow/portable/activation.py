"""A model's output activation, read from its network: the rule both modes run.

What a model's output is, ``UNIT`` (a sigmoid or softmax already applied),
``SIGNED_UNIT`` (a tanh) or ``UNBOUNDED`` (logits, distances: no
activation), decides the sigmoid a model's start sets, Use this model's
threshold, the ranking's cut-offs and a seed's probability. Measured, it is
a guess: the inference server's probe runs the network on noise and buckets
what comes out (portable/label_it.py classify_output_range), and
jrc_mus-livers_16nm_to_8nm_peroxisome answered -0.9595 everywhere, read as a
tanh's [-1, 1], when its network ends in a Conv. The network itself says it
for certain: ``output_activation`` walks back from the graph's output, past
the ops that only move values about, to the last op that computes them.

The graph comes as ``nodes``, each ``{"op": str, "inputs": [str],
"outputs": [str]}``, its inputs and outputs the names of the values it
reads and writes, as each format's extractor reads them: the inference
server's (inference/runner.py, models/configs: TorchScript, torch.export,
an ONNX file, a torch.fx trace) and the browser's (web/src/finetune/onnx-proto.ts,
through ``calls.output_activation``). ``op`` is the format's own name for
the op, which ``canonical`` reads: an ONNX op_type (``Sigmoid``), a
TorchScript kind (``aten::sigmoid``), a torch.export target
(``aten.sigmoid.default``), a torch.fx target or module class (``sigmoid``,
``Conv3d``) all come to one name (``sigmoid``, ``conv``), so each
extractor passes its names as they are and the sets below are the only
place that knows them. A clip's bounds, when its extractor can read them,
are the node's ``min`` and ``max`` (numbers).

A value no node writes is a graph input, a weight or a constant. So is one
a ``CONSTANT_OPS`` node writes (an ONNX Constant, a TorchScript or fx
attribute read): an add of one of those to a linear layer's output is the
layer's bias.

The standard library alone, beside label_it's names for the classes: the
browser's Python runs it as it is.
"""

import re
from typing import List, Optional

from cellmap_flow.portable.label_it import SIGNED_UNIT, UNBOUNDED, UNIT

# The ops that end a network with a known range, by canonical name, with the
# name the page says ("its network ends in a Sigmoid").
UNIT_OPS = {"sigmoid": "Sigmoid", "softmax": "Softmax", "hardsigmoid": "HardSigmoid"}
SIGNED_UNIT_OPS = {"tanh": "Tanh"}
# Bounded to [min, max] by the node's own bounds: [0, 1] is UNIT, [-1, 1] is
# SIGNED_UNIT, others (a ReLU6's [0, 6]) and unread ones decide nothing.
CLIP_OPS = {"clip": "Clip", "clamp": "Clip", "hardtanh": "Clip"}
# Linear layers: an output straight out of one is unbounded, logits or a
# distance, whatever its weights. The normalizations among them are affine
# per channel once trained.
LINEAR_OPS = {
    # aten::_convolution_mode is a TorchScript conv with padding="same"
    # (mito-aff-unet-setup-16's last).
    "conv": "Conv", "convolution": "Conv", "convolutionmode": "Conv", "convtranspose": "ConvTranspose",
    "gemm": "Linear", "linear": "Linear", "addmm": "Linear", "matmul": "MatMul", "mm": "MatMul", "bmm": "MatMul",
    "batchnorm": "BatchNorm", "batchnormalization": "BatchNorm", "nativebatchnorm": "BatchNorm",
    "nativebatchnormlegitnotraining": "BatchNorm", "cudnnbatchnorm": "BatchNorm",
    "instancenorm": "InstanceNorm", "instancenormalization": "InstanceNorm",
    "groupnorm": "GroupNorm", "groupnormalization": "GroupNorm", "nativegroupnorm": "GroupNorm",
    "layernorm": "LayerNorm", "layernormalization": "LayerNorm", "nativelayernorm": "LayerNorm",
}
# A bias added to a linear layer's output (an ONNX MatMul's, an fx trace's
# matmul + parameter): still the layer's range.
BIAS_OPS = {"add"}
# The ops that only move values about, crop or copy them, read through to
# their first input: every value out of one is a value of its input (or a
# pad's constant). A concatenation is one of them only with one input; of
# several, each could be another range. In eval mode a dropout is the
# identity, and a getitem is an index or a tuple's part (a torch.export
# batch norm's output).
PASS_THROUGH_OPS = {
    "reshape", "view", "viewas", "flatten", "unflatten", "squeeze", "unsqueeze", "expand", "expandas",
    "transpose", "permute", "t", "contiguous", "identity", "clone", "detach", "alias",
    "cast", "to", "tocopy", "typeas", "float", "half",
    "slice", "crop", "centercrop", "narrow", "select", "gather", "index", "indexselect", "getitem",
    "pad", "constantpadnd", "reflectionpad", "replicationpad",
    "concat", "cat",
    "dropout", "featuredropout", "alphadropout",
}
CONCAT_OPS = {"concat", "cat"}
# The ops whose output is a constant or a weight, not a computed value.
CONSTANT_OPS = {"constant", "getattr"}

_NAMESPACES = ("aten", "prims", "prim", "torch")


def canonical(op: str) -> str:
    """``op``, any format's name for it, as the sets above have it:
    ``aten::_convolution``, ``aten.convolution.default`` and ``Conv`` are
    ``convolution`` and ``conv``; ``Conv3d`` (an fx module's class) and
    ``aten::conv3d`` are ``conv``; ``aten::sigmoid_`` is ``sigmoid``.

    The namespace and a torch.export overload go, then the case, the
    underscores (``batch_norm`` is ONNX's ``BatchNormalization``'s
    ``batchnorm...``) and a dimensionality (``3d``)."""
    name = str(op or "")
    if "::" in name:
        name = name.split("::", 1)[1]
    parts = name.split(".")
    if len(parts) > 1 and parts[0] in _NAMESPACES:
        name = parts[1]
    name = name.lower().replace("_", "")
    return re.sub(r"[123]d$", "", name)


def _bounded(node) -> Optional[str]:
    """A clip's class from its bounds, None when they are not [0, 1] or [-1, 1] or not read."""
    bounds = (node.get("min"), node.get("max"))
    if not all(isinstance(b, (int, float)) and not isinstance(b, bool) for b in bounds):
        return None
    return {(0.0, 1.0): UNIT, (-1.0, 1.0): SIGNED_UNIT}.get(tuple(float(b) for b in bounds))


def output_activation(nodes: List[dict], output: str) -> Optional[dict]:
    """``{"output_class", "op"}``: the class of the graph value ``output``,
    from the last op that computes it, and that op's name for the page; None
    when the graph does not say.

    From ``output``, back through each PASS_THROUGH_OPS op's first input to
    the op that changes values:

    - a UNIT_OPS op (Sigmoid, Softmax, HardSigmoid), or a clip to [0, 1]: UNIT;
    - a SIGNED_UNIT_OPS op (Tanh), or a clip to [-1, 1]: SIGNED_UNIT;
    - a LINEAR_OPS op (Conv, ConvTranspose, Linear, MatMul, a
      normalization), or an add of a bias to one (one input from the graph,
      read back to the linear op; the other a weight or a constant):
      UNBOUNDED, its op the linear one;
    - anything else: None. Arithmetic after an activation (a sigmoid times
      2), a ReLU (no upper bound, but no logits either), a custom op, a
      concatenation of several values, or a value no node writes (the
      walk ran off the graph): what the output is, the graph cannot say.
    """
    producer = {}
    for node in nodes or []:
        if canonical(node.get("op")) in CONSTANT_OPS:
            continue
        for name in node.get("outputs") or []:
            producer[name] = node

    def computing(name):
        """The node that computes value ``name``, past the pass-through ops; None off the graph."""
        seen = set()
        while name in producer and name not in seen:
            seen.add(name)
            node = producer[name]
            op = canonical(node.get("op"))
            if op not in PASS_THROUGH_OPS:
                return node
            inputs = node.get("inputs") or []
            if not inputs or (op in CONCAT_OPS and len(inputs) != 1):
                return node
            name = inputs[0]
        return None

    node = computing(output)
    if node is None:
        return None
    op = canonical(node.get("op"))
    if op in UNIT_OPS:
        return {"output_class": UNIT, "op": UNIT_OPS[op]}
    if op in SIGNED_UNIT_OPS:
        return {"output_class": SIGNED_UNIT, "op": SIGNED_UNIT_OPS[op]}
    if op in CLIP_OPS:
        output_class = _bounded(node)
        return None if output_class is None else {"output_class": output_class, "op": CLIP_OPS[op]}
    if op in BIAS_OPS:
        computed = [name for name in node.get("inputs") or [] if name in producer]
        node = computing(computed[0]) if len(computed) == 1 else None
        op = canonical(node.get("op")) if node is not None else None
    if op in LINEAR_OPS:
        return {"output_class": UNBOUNDED, "op": LINEAR_OPS[op]}
    return None
