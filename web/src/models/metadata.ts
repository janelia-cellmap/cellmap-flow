// A model's entry in /api/huggingface-models, in the shape the dashboard's
// Models tab renders (models-tab.js renderHfModels: the key is the model id,
// the value is the repo's metadata.json, and only `description` is shown
// beside the id's last path segment; the whole value goes into the search
// string). The cellmap PyTorch repos publish metadata.json with the fields
// listed first below (e.g. cellmap/mito-aff-unet-setup-16); the ONNX exports
// publish a manifest.json instead, so this builds the same fields from it,
// then adds what the browser knows about the export.
//
// A model finetuned in this browser (finetuned.ts) has the same fields, its
// description saying where it came from when its record is known: the
// catalog keeps the registry's last listing and passes the record in. A
// model of a repo added beside the built-in one says its repo first, so two
// repos' folders of one name are told apart in the list, and its author is
// the repo's owner.
import { FINETUNED_REPO } from "../contracts";
import type { FinetunedRecord, ModelEntry } from "../contracts";

import { HF_REPO, HF_REVISION } from "./hf";
import { voxelLabel } from "./manifest";

/** Who exported and uploaded cellmap/onnx-web-models (the other cellmap repos name him in metadata.json too). */
export const MODELS_AUTHOR = "Marwan Zouinkhi";

/** The author of a model finetuned in this browser: not a person, and not Marwan. */
export const FINETUNED_AUTHOR = "finetuned in this browser";

export function metadataFor(entry: ModelEntry, record?: FinetunedRecord): Record<string, unknown> {
  const m = entry.manifest;
  const v = entry.variant;
  const finetuned = entry.repo === FINETUNED_REPO;
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
    description: describe(entry, record),
    author: finetuned ? FINETUNED_AUTHOR : entry.repo === HF_REPO ? MODELS_AUTHOR : entry.repo.split("/")[0],
    // The Hub revision for an export; a finetuned model's version is when it was trained.
    version: finetuned ? (record?.created_at ?? null) : HF_REVISION,
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
    // Where the export came from, when its manifest says.
    ...Object.fromEntries((["source_repo", "distilled_from_repo", "torch_version", "exporter", "opset"] as const).filter((k) => m[k] !== undefined).map((k) => [k, m[k]])),
    // And where a finetuned model came from, so the base's name or the
    // volume's id find it in the tab's search.
    ...(record ? { finetuned: { base: record.base, volume: record.volume, created_at: record.created_at, epochs: record.epochs, final_loss: record.final_loss, params: record.params, ...(record.from_link ? { from_link: record.from_link } : {}), ...(record.from_file ? { from_file: record.from_file } : {}) } } : {}),
  };
}

/** One line for the tab: "mito affinities at 16 nm, ResUNet3DValid (10.7M parameters); ONNX fp32, block 132³ → 44³, 42.9 MB; raw logits (apply sigmoid / threshold at 0)".
 *
 * For a model finetuned in this browser the first part says so instead:
 * "finetuned in this browser from mito_aff_setup54_16nm on vol-…, 2026-10-15;
 * 20 epochs, final loss 0.312" with its record, the manifest's checkpoint
 * line ("finetuned in the browser from …") without. One that came from a
 * shared link says that instead, and the name the link gave it when this
 * browser serves it under another: "from a shared link (as
 * mito_aff_setup54_16nm-ft1): mito_aff_setup54_16nm with the link's last
 * layer, 2026-10-16", with no epochs or loss, which stayed with the
 * browser that trained it; one opened from a file, "opened from
 * mito_aff_setup54_16nm-ft1.onnx in this browser, 2026-10-26".
 *
 * A model of another repo than HF_REPO says the repo first, and what its
 * manifest's published block says it was finetuned from: "alice/mito-ft1,
 * finetuned from cellmap/onnx-web-models/setup_54; mito affinities at 16 nm,
 * …". */
export function describe(entry: ModelEntry, record?: FinetunedRecord): string {
  const m = entry.manifest;
  const v = entry.variant;
  const parts: string[] = [];
  if (entry.repo === FINETUNED_REPO) {
    if (record?.from_link) {
      const as = record.from_link.name === record.name ? "" : ` (as ${record.from_link.name})`;
      parts.push(`from a shared link${as}: ${record.base.name} with the link's last layer, ${dateOf(record.created_at)}`);
    } else if (record?.from_file) {
      parts.push(`opened from ${record.from_file.name} in this browser, ${dateOf(record.created_at)}`);
    } else if (record) {
      const what = record.params.finetune_mode === "full" ? "whole network finetuned" : "finetuned";
      parts.push(`${what} in this browser from ${record.base.name} on ${record.volume}, ${dateOf(record.created_at)}`);
      parts.push(`${record.epochs} ${record.epochs === 1 ? "epoch" : "epochs"}, ${record.final_loss === null ? "final loss unknown" : `final loss ${lossLabel(record.final_loss)}`}`);
    } else {
      parts.push(m.checkpoint || FINETUNED_AUTHOR);
    }
  } else {
    if (entry.repo !== HF_REPO) parts.push(m.published?.finetuned_from ? `${entry.repo}, finetuned from ${m.published.finetuned_from}` : entry.repo);
    const network = [m.architecture, m.params > 0 ? `(${(m.params / 1e6).toFixed(1)}M parameters)` : ""].filter(Boolean).join(" ");
    parts.push(`${m.task.replace(/_/g, " ")} at ${voxelLabel(m.input_voxel_size_nm)} nm${network ? `, ${network}` : ""}`);
  }
  parts.push(`ONNX ${v.variant}, block ${blockLabel(v.input_size)} → ${blockLabel(v.output_size)}, ${v.file_mb.toFixed(1)} MB`);
  if (m.output) parts.push(m.output);
  return parts.join("; ");
}

// "132³" for a cube, "132×132×64" otherwise.
function blockLabel(size: number[]): string {
  return size.every((n) => n === size[0]) ? `${size[0]}³` : size.join("×");
}

// The day of an ISO 8601 time stamp ("2026-10-15"); anything else as it is.
function dateOf(createdAt: string): string {
  return /^\d{4}-\d{2}-\d{2}/.test(createdAt) ? createdAt.slice(0, 10) : createdAt;
}

// Three significant figures, trailing zeros dropped: 0.312, 0.0004, 12.5.
function lossLabel(loss: number): string {
  return Number.isFinite(loss) ? String(Number(loss.toPrecision(3))) : String(loss);
}

/** The training iteration in a checkpoint name such as "model_checkpoint_260000", or null. */
export function checkpointIteration(checkpoint: string): number | null {
  const match = /(\d+)\s*$/.exec(checkpoint);
  return match ? Number(match[1]) : null;
}
