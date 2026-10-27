// What /__control__/model_info says about a model served in the browser:
// the answer the Python server builds in models/geometry.py
// (ModelGeometry.to_model_info) and serving/chunkmirage_server.py
// (ChunkmirageServer.model_info), built here from the manifest and the
// chosen variant instead of from a loaded model and an output probe.
//
// Field by field, as the dashboard's layer code reads them:
//   write_shape / read_shape    the output and input block in nm (z, y, x),
//                               as ModelGeometry sends them and as the
//                               Finetune tab shows them ("Output Size (nm)");
//                               the ONNX graph fixes the voxel counts, so
//                               they are the variant's sizes times the voxel
//                               size, not something to derive;
//   output_voxel_size, input_voxel_size, effective_output_voxel_size
//                               nm; effective equals output here because the
//                               browser reads the raw at the model's own
//                               voxel size (no relabelled level);
//   has_channel, output_axes    channels first, as chunkmirage serves them;
//   output_class                the output activation's class in the Python
//                               vocabulary (serving/probe.py: "unit",
//                               "signed_unit", "unbounded"), read off the
//                               network the engine runs (networkActivation:
//                               its ONNX graph's last op that computes
//                               values), else off the manifest's `output`
//                               text, the way the server reads its model's
//                               graph, else probes it;
//   output_class_from, output_op
//                               which of the two it was ("graph", with the
//                               op, or "manifest"; the server's other is
//                               "probe"), null when neither says;
//   output_min / output_max     the probed range: null, nothing was probed;
//   display_channel             the channel to show for a flow model
//                               (Cellpose): none of these models has one;
//   available                   true: the geometry is known without a probe.
import type { Inference, ModelEntry, ModelInfo, ModelManifest, OutputActivation } from "../contracts";
import { readGraph, readModel } from "../finetune/onnx-proto";

/** The portable rule a graph's nodes go to (cellmap_flow/portable/calls.py, over portable/activation.py). */
const OUTPUT_ACTIVATION = "cellmap_flow.portable.calls:output_activation";

const SPATIAL_AXES = ["z", "y", "x"];

/** The served model's info; `activation` is what its network ends in
 * (networkActivation), or null when that was not read or does not say. */
export function modelInfo(entry: ModelEntry, activation: OutputActivation | null = null): ModelInfo {
  const m = entry.manifest;
  const v = entry.variant;
  const spatial = SPATIAL_AXES.slice(SPATIAL_AXES.length - m.input_voxel_size_nm.length);
  return {
    output_channels: m.channels.length,
    channels: m.channels.slice(),
    write_shape: v.output_size.map((n, a) => n * m.output_voxel_size_nm[a]),
    read_shape: v.input_size.map((n, a) => n * m.input_voxel_size_nm[a]),
    output_voxel_size: m.output_voxel_size_nm.slice(),
    input_voxel_size: m.input_voxel_size_nm.slice(),
    effective_output_voxel_size: m.output_voxel_size_nm.slice(),
    has_channel: true,
    output_axes: ["c", ...spatial],
    available: true,
    ...outputClass(m, activation),
    output_min: null,
    output_max: null,
    display_channel: null,
    engine: "browser",
  };
}

/** What the network of the ONNX model in `bytes` ends in, by the rule both
 * modes run (portable/activation.py output_activation, through the engine's
 * Python): its graph's nodes as they are, each op_type with the names of the
 * values it reads and writes (an initializer is a name no node writes), and
 * `output`, the graph output the engine reads (the manifest's io.output),
 * else the graph's first. Null when the graph does not say: a Relu last,
 * arithmetic after a sigmoid, a custom op. A Clip's bounds are not read
 * (from opset 11 they are inputs, a Constant's or an initializer's), so a
 * Clip last says nothing either: none of the exports measured on
 * 2026-10-27 ends in one (they end in a Conv or a Sigmoid). Throws when
 * `bytes` are not an ONNX model. The weights are stepped over, not decoded
 * (onnx-proto.ts readGraph), so a big model costs no more than a small one. */
export async function networkActivation(bytes: ArrayBuffer, output: string | undefined, infer: Pick<Inference, "python">): Promise<OutputActivation | null> {
  const graph = readGraph(readModel(new Uint8Array(bytes)).graph);
  const nodes = graph.nodes.map((n) => ({ op: n.opType, inputs: n.inputs, outputs: n.outputs }));
  return infer.python<OutputActivation | null>(OUTPUT_ACTIVATION, { nodes, output: output || graph.outputs[0] || "" });
}

/** A served model's output class and where it came from: its network's
 * (`activation`) when that says, else the manifest's `output` text
 * (outputClassOf), else null from nowhere. What the Python server's
 * model_info answers, with "manifest" where it says "probe": the advice,
 * the start's postprocessing and the ranking read the class alike, and the
 * advice says which it was. */
export function outputClass(manifest: ModelManifest, activation: OutputActivation | null): Pick<ModelInfo, "output_class" | "output_class_from" | "output_op"> {
  if (activation) return { output_class: activation.output_class, output_class_from: "graph", output_op: activation.op };
  const declared = outputClassOf(manifest.output);
  return { output_class: declared, output_class_from: declared ? "manifest" : null, output_op: null };
}

/** The activation class a manifest's `output` text describes, in
 * serving/probe.py's vocabulary: "unit" for probabilities ([0, 1], a sigmoid
 * or softmax already applied), "signed_unit" for a tanh ([-1, 1]),
 * "unbounded" for logits, distances and the like (no output activation), or
 * null when the text says nothing recognisable. Marwan's exports all say
 * "raw logits (apply sigmoid / threshold at 0)". */
export function outputClassOf(output: string): string | null {
  const text = output.toLowerCase();
  if (!text) return null;
  if (/\blogits?\b|\bdistances?\b|\bunbounded\b|\braw\b/.test(text)) return "unbounded";
  if (/\btanh\b|signed/.test(text)) return "signed_unit";
  if (/\bsigmoid\b|\bsoftmax\b|probabilit|\[0,\s*1\]/.test(text)) return "unit";
  return null;
}
