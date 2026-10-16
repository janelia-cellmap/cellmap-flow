"""Generate the small ONNX models the browser finetune's tests run.

The browser trains a model's head, its last 1×1 convolution, and writes the
result back into the model's bytes (web/src/finetune/onnx-head.ts, features.ts,
head-trainer.ts, run.ts). The tests check each step against onnxruntime-web
running a real graph, and the real exports (cellmap/onnx-web-models, 4 to
60 MB) are too big to commit, so this writes two models of a few kilobytes
shaped like their tails as inspected on 2026-10-15 (setup_53_2d's
in556_fp32 and in300_fp16 exports; PLAN_browser_finetune.md, "What trains"):

``head.onnx``: ``input`` float32 [1, 1, 1, 8, 8] → Gather (axis 2, index 0)
to 4-D → Conv 3×3, 1 → 4 channels, with bias → Relu (the body, standing in
for the U-Net) → Conv 1×1, 4 → 2 channels, with bias (the head,
``head.weight`` [2, 4, 1, 1] and ``head.bias`` [2]) → Unsqueeze (axis 2) →
``output`` float32 [1, 2, 1, 6, 6]. As in the exports the Gather index and
the Unsqueeze axes are Constant nodes, not initializers.

``head_fp16.onnx``: the same network as the fp16 exports store it: every
initializer float16, a Cast to float16 after the float32 input and a Cast
back to float32 before the output; and, as in those exports, the two Cast
nodes come last in the node list, after the nodes that consume and produce
their tensors, so a reader must find producers by name rather than trust
the order.

Deterministic (``numpy.random.default_rng(0)``), opset 18 and ir_version 8
as the exports. The tests read the weights from the files; only the channel
counts (4 in, 2 out) and the output's 6×6 are named in them. Regenerate with
any Python that has numpy and the onnx package, from anywhere::

    python3 web/test/fixtures/make_onnx_fixture.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HERE = Path(__file__).resolve().parent
SEED = 0
INPUT_SIZE = 8
BODY_CHANNELS = 4
HEAD_CHANNELS = 2


def weights(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """The network's parameters, float32, in a fixed order from the seed."""
    return {
        "body.weight": rng.normal(0.0, 0.5, size=(BODY_CHANNELS, 1, 3, 3)).astype(np.float32),
        "body.bias": rng.normal(0.0, 0.1, size=(BODY_CHANNELS,)).astype(np.float32),
        "head.weight": rng.normal(0.0, 1.0, size=(HEAD_CHANNELS, BODY_CHANNELS, 1, 1)).astype(np.float32),
        "head.bias": rng.normal(0.0, 0.5, size=(HEAD_CHANNELS,)).astype(np.float32),
    }


def make_model(params: dict[str, np.ndarray], fp16: bool) -> onnx.ModelProto:
    out_size = INPUT_SIZE - 2  # the 3×3 convolution, unpadded
    dtype = np.float16 if fp16 else np.float32
    initializers = [numpy_helper.from_array(v.astype(dtype), k) for k, v in params.items()]
    # The tensor the body reads: the input itself, or its float16 cast.
    body_input = "graph_input_cast_0" if fp16 else "input"
    head_output = "graph_output_cast_0" if fp16 else "output"
    nodes = [
        helper.make_node("Constant", [], ["gather_index"], value=helper.make_tensor("gather_index_value", TensorProto.INT64, [], [0])),
        helper.make_node("Gather", [body_input, "gather_index"], ["body_input_4d"], axis=2),
        helper.make_node("Conv", ["body_input_4d", "body.weight", "body.bias"], ["body_conv"], kernel_shape=[3, 3], pads=[0, 0, 0, 0], strides=[1, 1], dilations=[1, 1], group=1),
        helper.make_node("Relu", ["body_conv"], ["features"]),
        helper.make_node("Conv", ["features", "head.weight", "head.bias"], ["logits"], kernel_shape=[1, 1], pads=[0, 0, 0, 0], strides=[1, 1], dilations=[1, 1], group=1, name="head"),
        helper.make_node("Constant", [], ["unsqueeze_axes"], value=helper.make_tensor("unsqueeze_axes_value", TensorProto.INT64, [1], [2])),
        helper.make_node("Unsqueeze", ["logits", "unsqueeze_axes"], [head_output]),
    ]
    casts = []
    if fp16:
        casts = [
            helper.make_node("Cast", ["input"], [body_input], to=TensorProto.FLOAT16, name="graph_input_cast0"),
            helper.make_node("Cast", [head_output], ["output"], to=TensorProto.FLOAT, name="graph_output_cast0"),
        ]
    graph = helper.make_graph(
        [casts[0], *nodes, casts[1]] if fp16 else nodes,
        "head_fixture_fp16" if fp16 else "head_fixture",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 1, INPUT_SIZE, INPUT_SIZE])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, HEAD_CHANNELS, 1, out_size, out_size])],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], producer_name="make_onnx_fixture.py")
    model.ir_version = 8
    # Checked in topological order, which the checker insists on; then the
    # Casts moved last, after the nodes that use their tensors, as the fp16
    # exports have them (onnxruntime sorts the graph itself and runs them).
    onnx.checker.check_model(model, full_check=True)
    if fp16:
        del model.graph.node[:]
        model.graph.node.extend([*nodes, *casts])
    return model


def main() -> None:
    params = weights(np.random.default_rng(SEED))
    for name, fp16 in (("head.onnx", False), ("head_fp16.onnx", True)):
        path = HERE / name
        onnx.save(make_model(params, fp16), str(path))
        print(f"wrote {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
