// A model's entry in /api/huggingface-models, in the shape the dashboard's
// Models tab renders (models-tab.js renderHfModels: the key is the model id,
// the value is the repo's metadata.json, and only `description` is shown
// beside the id's last path segment; the whole value goes into the search
// string). The cellmap PyTorch repos publish metadata.json with the fields
// listed first below (e.g. cellmap/mito-aff-unet-setup-16); the ONNX exports
// publish a manifest.json instead, so this builds the same fields from it,
// then adds what the browser knows about the export.
import type { ModelEntry } from "../contracts";

import { HF_REVISION } from "./hf";
import { voxelLabel } from "./manifest";

/** Who exported and uploaded cellmap/onnx-web-models (the other cellmap repos name him in metadata.json too). */
export const MODELS_AUTHOR = "Marwan Zouinkhi";

export function metadataFor(entry: ModelEntry): Record<string, unknown> {
  const m = entry.manifest;
  const v = entry.variant;
  return {
    // The fields of a PyTorch repo's metadata.json.
    model_name: entry.name,
    model_type: m.architecture,
    framework: "onnx",
    spatial_dims: m.input_voxel_size_nm.length,
    in_channels: 1,
    out_channels: m.channels.length,
    iteration: checkpointIteration(m.checkpoint),
    input_voxel_size: m.input_voxel_size_nm,
    output_voxel_size: m.output_voxel_size_nm,
    channels_names: m.channels,
    input_shape: v.input_size,
    output_shape: v.output_size,
    // The ONNX graph has a fixed block: inference reads and writes exactly it.
    inference_input_shape: v.input_size,
    inference_output_shape: v.output_size,
    description: describe(entry),
    author: MODELS_AUTHOR,
    version: HF_REVISION,
    format_version: "1",
    // Beyond those: the export itself. The tab puts the whole value in each
    // row's search string, so "fp16" or "ResUNet" find a model too.
    repo: entry.repo,
    folder: entry.folder,
    task: m.task,
    checkpoint: m.checkpoint,
    params: m.params,
    output: m.output,
    input_norm: m.input_norm,
    variant: v.variant,
    file: v.file,
    file_mb: v.file_mb,
    onnx_url: entry.onnxUrl,
  };
}

/** One line for the tab: "mito affinities at 16 nm, ResUNet3DValid (10.7M parameters); ONNX fp32, block 132³ → 44³, 42.9 MB; raw logits (apply sigmoid / threshold at 0)". */
export function describe(entry: ModelEntry): string {
  const m = entry.manifest;
  const v = entry.variant;
  const parts = [`${m.task.replace(/_/g, " ")} at ${voxelLabel(m.input_voxel_size_nm)} nm`];
  const network = [m.architecture, m.params > 0 ? `(${(m.params / 1e6).toFixed(1)}M parameters)` : ""].filter(Boolean).join(" ");
  if (network) parts[0] += `, ${network}`;
  parts.push(`ONNX ${v.variant}, block ${blockLabel(v.input_size)} → ${blockLabel(v.output_size)}, ${v.file_mb.toFixed(1)} MB`);
  if (m.output) parts.push(m.output);
  return parts.join("; ");
}

// "132³" for a cube, "132×132×64" otherwise.
function blockLabel(size: number[]): string {
  return size.every((n) => n === size[0]) ? `${size[0]}³` : size.join("×");
}

/** The training iteration in a checkpoint name such as "model_checkpoint_260000", or null. */
export function checkpointIteration(checkpoint: string): number | null {
  const match = /(\d+)\s*$/.exec(checkpoint);
  return match ? Number(match[1]) : null;
}
