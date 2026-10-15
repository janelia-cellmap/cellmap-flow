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
//                               "signed_unit", "unbounded"), which the
//                               server observes by probing and the browser
//                               reads off the manifest's `output` text;
//   output_min / output_max     the probed range: null, nothing was probed;
//   display_channel             the channel to show for a flow model
//                               (Cellpose): none of these models has one;
//   available                   true: the geometry is known without a probe.
import type { ModelEntry, ModelInfo } from "../contracts";

const SPATIAL_AXES = ["z", "y", "x"];

export function modelInfo(entry: ModelEntry): ModelInfo {
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
    output_class: outputClassOf(m.output),
    output_min: null,
    output_max: null,
    display_channel: null,
    engine: "browser",
  };
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
