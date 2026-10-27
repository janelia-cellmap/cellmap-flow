"""The output activation read from a network's graph (portable/activation.py)."""

import pytest

from cellmap_flow.portable.activation import output_activation
from cellmap_flow.portable.label_it import SIGNED_UNIT, UNBOUNDED, UNIT


def chain(*ops):
    """A graph of ``ops`` one after another from the input ``x``: each op's
    output ``v<i>``, a weight ``w<i>`` beside each first input, and the last
    output ``v<n>``."""
    return [{"op": op, "inputs": [f"v{i}" if i else "x", f"w{i}"], "outputs": [f"v{i + 1}"]}
            for i, op in enumerate(ops)], f"v{len(ops)}"


@pytest.mark.parametrize("ops, expected", [
    # The cached Hugging Face exports' ONNX tails: jrc_mus-livers' Conv,
    # setup-54-2d's Unsqueeze <- Conv, the salivary distance models' Sigmoid.
    (["Relu", "Conv"], (UNBOUNDED, "Conv")),
    (["Conv", "Unsqueeze"], (UNBOUNDED, "Conv")),
    (["Conv", "Sigmoid"], (UNIT, "Sigmoid")),
    # TorchScript's and torch.export's names for the same ops.
    (["aten::_convolution", "aten::slice", "aten::tanh"], (SIGNED_UNIT, "Tanh")),
    (["aten.convolution.default", "aten.sigmoid.default", "aten.view.default"], (UNIT, "Sigmoid")),
    # Arithmetic after the activation: [0, 2], or anything; not known.
    (["Conv", "Sigmoid", "Mul"], None),
    (["Conv", "Relu"], None),
])
def test_the_last_op_that_computes_the_output_decides(ops, expected):
    nodes, output = chain(*ops)
    found = output_activation(nodes, output)
    assert (found and (found["output_class"], found["op"])) == expected


def test_a_bias_added_to_a_matmul_is_the_layers_and_two_computed_values_are_not():
    nodes = [
        {"op": "MatMul", "inputs": ["x", "weight"], "outputs": ["y"]},
        {"op": "Constant", "inputs": [], "outputs": ["bias"]},
        {"op": "Add", "inputs": ["y", "bias"], "outputs": ["out"]},
    ]
    assert output_activation(nodes, "out") == {"output_class": UNBOUNDED, "op": "MatMul"}
    nodes.append({"op": "Sigmoid", "inputs": ["x"], "outputs": ["s"]})
    nodes.append({"op": "Add", "inputs": ["y", "s"], "outputs": ["sum"]})
    assert output_activation(nodes, "sum") is None


def test_a_clip_decides_by_its_bounds():
    nodes, output = chain("Conv", "Clip")
    assert output_activation(nodes, output) is None  # bounds not read
    nodes[-1].update(min=0, max=1)
    assert output_activation(nodes, output)["output_class"] == UNIT
    nodes[-1].update(min=0, max=6)
    assert output_activation(nodes, output) is None
