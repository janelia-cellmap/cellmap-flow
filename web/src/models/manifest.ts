// A model folder's manifest.json, read into the ModelManifest the modules
// share, and the ModelEntry built from it.
//
// Marwan's exports (cellmap/onnx-web-models, 2026-10-14) write two fields in
// a shorter form than contracts.ts declares, and parseManifest widens them:
//   - variants[].input_size / output_size are one number (the blocks are
//     cubes): here they become [N, N, N], one per spatial axis;
//   - io is one sentence, "input 'input' float32 [1,1,z,y,x]; output 'output'
//     float32 [1,C,z',y',x']": here it becomes {input, output, description}
//     with the tensor names picked out of the quotes, so the inference code
//     can name its feeds and fetches without parsing prose.
// A manifest that already uses the wide forms is taken as it is, so a future
// export can switch without a change here.
//
// Since 2026-10-16 the manifests also say what the browser can train
// (`finetune`: the head, and the whole network, the 2D students' since
// 2026-10-16 and the 3D models' since 2026-10-17) and where the export came
// from (source_repo, distilled_from_repo, torch_version, exporter, opset);
// both are kept (parseFinetune). A
// finetune entry that is not what it should be is left out rather than
// failing the manifest: the model still runs, without that way to train it.
import { FINETUNED_REPO } from "../contracts";
import type { FinetuneSpec, FullFinetuneSpec, HeadFinetuneSpec, ModelEntry, ModelManifest, ModelVariant, Step, VariantName } from "../contracts";

import { HF_REPO, resolveUrl } from "./hf";

/** The model folders known on 2026-10-15, for when the Hub's file list cannot be fetched. */
export const FALLBACK_FOLDERS: readonly string[] = ["setup_53", "setup_53_2d", "setup_54", "setup_54_2d", "setup_55", "setup_56", "setup_61", "setup_67"];

/** Folders that hold no manifest of their own: conv3d/ repeats every model
 * with native Conv3d nodes, about 16x slower on WebGPU (onnxruntime-web runs
 * Conv3d with a naive shader), meant for CUDA or TensorRT runtimes. */
export const SKIPPED_FOLDERS: readonly string[] = ["conv3d"];

const VARIANT_NAMES: readonly VariantName[] = ["fp32", "fp16", "int8w", "int8w_fp16"];

/** A manifest that cannot be used: a field missing or of the wrong kind. */
export class ManifestError extends Error {
  constructor(where: string, problem: string) {
    super(`${where}: ${problem}`);
    this.name = "ManifestError";
  }
}

/** The model folders among a tree listing's top-level entries, conv3d/ left out. */
export function modelFolders(entries: { type: string; path: string }[]): string[] {
  return entries
    .filter((e) => e.type === "directory" && !e.path.includes("/") && !SKIPPED_FOLDERS.includes(e.path))
    .map((e) => e.path)
    .sort();
}

/** `raw` (a manifest.json parsed as JSON) checked and widened into a ModelManifest.
 * `where` names the source in error messages. */
export function parseManifest(raw: unknown, where = "manifest.json"): ModelManifest {
  const m = asRecord(raw, where, "the manifest");
  const inputVoxel = numbers(m, "input_voxel_size_nm", where);
  const outputVoxel = numbers(m, "output_voxel_size_nm", where);
  if (outputVoxel.length !== inputVoxel.length) {
    throw new ManifestError(where, "input_voxel_size_nm and output_voxel_size_nm have different lengths");
  }
  const dims = inputVoxel.length;
  const rawVariants = m.variants;
  if (!Array.isArray(rawVariants) || rawVariants.length === 0) throw new ManifestError(where, "variants is missing or empty");
  const variants = rawVariants.map((v, i) => parseVariant(v, dims, `${where} variants[${i}]`)).filter((v): v is ModelVariant => v !== null);
  if (variants.length === 0) throw new ManifestError(where, `no variant has a known precision (${VARIANT_NAMES.join(", ")})`);
  return {
    model: string(m, "model", where),
    task: string(m, "task", where),
    architecture: optionalString(m, "architecture") ?? "",
    checkpoint: optionalString(m, "checkpoint") ?? "",
    params: typeof m.params === "number" ? m.params : 0,
    channels: strings(m, "channels", where),
    input_voxel_size_nm: inputVoxel,
    output_voxel_size_nm: outputVoxel,
    output: optionalString(m, "output") ?? "",
    input_norm: steps(m.input_norm, where),
    io: parseIo(m.io),
    variants: sortVariants(variants),
    ...provenance(m),
    ...finetuneOf(m),
  };
}

function finetuneOf(m: Record<string, unknown>): Pick<ModelManifest, "finetune"> {
  const finetune = parseFinetune(m.finetune);
  return finetune ? { finetune } : {};
}

// The export's provenance, the fields that are there.
function provenance(m: Record<string, unknown>): Pick<ModelManifest, "source_repo" | "distilled_from_repo" | "torch_version" | "exporter" | "opset"> {
  const out: Pick<ModelManifest, "source_repo" | "distilled_from_repo" | "torch_version" | "exporter" | "opset"> = {};
  for (const key of ["source_repo", "distilled_from_repo", "torch_version", "exporter"] as const) {
    const value = optionalString(m, key);
    if (value) out[key] = value;
  }
  if (typeof m.opset === "number" && Number.isInteger(m.opset)) out.opset = m.opset;
  return out;
}

/** The manifest's `finetune` block: its `head` and `full` entries that have
 * what training needs, each as given (its notes and numbers kept); null
 * when there is none, or nothing in it the browser can use. */
export function parseFinetune(value: unknown): FinetuneSpec | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const block = value as Record<string, unknown>;
  const spec: FinetuneSpec = {};
  const head = record(block.head);
  if (head && isString(head.weight) && isString(head.features) && (head.bias === null || head.bias === undefined || isString(head.bias)) && isStrings(head.files)) {
    spec.head = { ...head, bias: head.bias ?? null } as unknown as HeadFinetuneSpec;
  }
  const full = record(block.full);
  if (full && isString(full.library) && isStrings(full.files) && isString(full.initializers) && isStrings(full.write_back)) {
    const fullSpec = {
      ...full,
      weight_map: isString(full.weight_map) ? full.weight_map : "identity",
      frozen: isStrings(full.frozen) ? full.frozen : [],
    } as unknown as FullFinetuneSpec;
    if (!isString(full.torch_params)) delete fullSpec.torch_params;
    const spike = spikeOf(full.spike);
    if (spike) fullSpec.spike = spike;
    else delete fullSpec.spike;
    spec.full = fullSpec;
  }
  return spec.head || spec.full ? spec : null;
}

// The spike's per-side numbers (`full: {"300": {step_ms, peak_mb}}`), the
// sides with both numbers; the rest of the spike (its setup, its other
// measurements) as given.
function spikeOf(value: unknown): FullFinetuneSpec["spike"] | null {
  const spike = record(value);
  const sides = record(spike?.full);
  if (!spike || !sides) return null;
  const full: Record<string, { step_ms: number; peak_mb: number }> = {};
  for (const [side, numbers] of Object.entries(sides)) {
    const n = record(numbers);
    if (n && typeof n.step_ms === "number" && typeof n.peak_mb === "number" && /^\d+$/.test(side)) full[side] = { step_ms: n.step_ms, peak_mb: n.peak_mb };
  }
  return { ...spike, full } as FullFinetuneSpec["spike"];
}

function record(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : null;
}

const isString = (value: unknown): value is string => typeof value === "string" && value.length > 0;
const isStrings = (value: unknown): value is string[] => Array.isArray(value) && value.every((v) => typeof v === "string");

// A variant row; null when its precision is not one we know how to choose
// between (the caller leaves it out rather than guess at its memory use).
function parseVariant(raw: unknown, dims: number, where: string): ModelVariant | null {
  const v = asRecord(raw, where, "the variant");
  if (typeof v.variant !== "string" || !(VARIANT_NAMES as readonly string[]).includes(v.variant)) return null;
  const variant: ModelVariant = {
    variant: v.variant as VariantName,
    input_size: sizes(v.input_size, dims, `${where}.input_size`),
    output_size: sizes(v.output_size, dims, `${where}.output_size`),
    file: string(v, "file", where),
    file_mb: typeof v.file_mb === "number" ? v.file_mb : 0,
  };
  for (const key of ["max_abs", "mean_abs", "flip_frac", "ort_cpu_s"] as const) {
    if (typeof v[key] === "number") variant[key] = v[key];
  }
  return variant;
}

// A block size: one number for a cube, or one number per spatial axis.
function sizes(value: unknown, dims: number, where: string): number[] {
  if (typeof value === "number" && Number.isInteger(value) && value > 0) return Array.from({ length: dims }, () => value);
  if (Array.isArray(value) && value.length === dims && value.every((n) => typeof n === "number" && Number.isInteger(n) && n > 0)) {
    return value.slice();
  }
  throw new ManifestError(where, `expected a positive whole number or ${dims} of them`);
}

// The io field: an object as contracts.ts declares it, or Marwan's sentence
// with the tensor names in single quotes.
function parseIo(value: unknown): ModelManifest["io"] {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    const io = value as Record<string, unknown>;
    return { ...io, input: typeof io.input === "string" ? io.input : "input", output: typeof io.output === "string" ? io.output : "output" };
  }
  const text = typeof value === "string" ? value : "";
  const input = /input\s+'([^']+)'/.exec(text)?.[1] ?? "input";
  const output = /output\s+'([^']+)'/.exec(text)?.[1] ?? "output";
  return text ? { input, output, description: text } : { input, output };
}

// The normalizer steps; a missing list means no normalization.
function steps(value: unknown, where: string): Step[] {
  if (value === undefined || value === null) return [];
  if (!Array.isArray(value)) throw new ManifestError(where, "input_norm is not a list");
  return value.map((step, i) => {
    const s = asRecord(step, `${where} input_norm[${i}]`, "the step");
    if (typeof s.name !== "string" || !s.name) throw new ManifestError(where, `input_norm[${i}] has no name`);
    return s as Step;
  });
}

// Variants in a fixed order, smallest block first, precisions in VARIANT_NAMES
// order within a block, so "the middle block" means the same on every run.
function sortVariants(variants: ModelVariant[]): ModelVariant[] {
  const volume = (v: ModelVariant) => v.input_size.reduce((a, b) => a * b, 1);
  return variants.slice().sort((a, b) => volume(a) - volume(b) || VARIANT_NAMES.indexOf(a.variant) - VARIANT_NAMES.indexOf(b.variant));
}

/** The layer and job name for a manifest: `<task>_<model>_<voxel>nm`, e.g.
 * `mito_aff_setup54_16nm`, `nuc_setup56_128nm`, `er_setup67_8nm`,
 * `mito_aff_setup53-2d_16nm`.
 *
 * The rule: the task lower-cased with runs of anything but letters and digits
 * turned into one underscore and "affinities" shortened to "aff"; the model
 * id with its underscores removed (so the name's underscores separate its
 * three parts), except that a dimensionality tag at its end (`_2d`, the 2D
 * students distilled from the 3D models) is kept behind a hyphen, since
 * `setup532d` reads as setup 532; the input voxel size as one number when
 * isotropic, else z×y×x joined with "x" (16x16x8nm). The result is
 * lower-case letters, digits, underscores, at most one hyphen and at most a
 * decimal point, which is a valid URL path segment and a Neuroglancer layer
 * name, and carries no ARGS_KEY marker. */
export function entryName(manifest: ModelManifest): string {
  const task = slug(manifest.task).replace(/affinities/g, "aff") || "model";
  const model = slug(manifest.model).replace(/_(\d+d)$/, "-$1").replace(/_/g, "");
  return `${task}_${model}_${voxelLabel(manifest.input_voxel_size_nm)}nm`;
}

/** "16" for an isotropic voxel size, "16x16x8" otherwise; whole numbers without a decimal point. */
export function voxelLabel(voxelSize: number[]): string {
  const parts = voxelSize.map((n) => String(Number(n.toFixed(4))));
  return parts.every((p) => p === parts[0]) ? (parts[0] ?? "") : parts.join("x");
}

function slug(text: string): string {
  return text.toLowerCase().replace(/[^a-z0-9.]+/g, "_").replace(/^_+|_+$/g, "");
}

/** The ModelEntry for `folder`'s manifest with `variant` chosen.
 *
 * For a Hugging Face repo the name follows entryName and the bytes come from
 * the repo's resolve URL. For FINETUNED_REPO, a model finetuned in this
 * browser (finetuned.ts), the folder is the name (`<base name>-ft<n>`, which
 * is already a layer name) and the "URL" is the byte-cache key of bytes that
 * exist only in this browser: nothing is fetched. Serving re-tunes an entry
 * through this function with the variant chosen for the GPU (backend/
 * serving.ts), and a finetuned manifest has one variant, so a finetuned
 * entry comes through unchanged. */
export function entryFor(folder: string, manifest: ModelManifest, variant: ModelVariant, repo = HF_REPO): ModelEntry {
  if (repo === FINETUNED_REPO) {
    const id = `${FINETUNED_REPO}/${folder}`;
    return { id, name: folder, repo, folder, manifest, variant, onnxUrl: finetunedBytesKey(id) };
  }
  return {
    id: `${repo}/${folder}`,
    name: entryName(manifest),
    repo,
    folder,
    manifest,
    variant,
    onnxUrl: resolveUrl(`${folder}/${variant.file}`, repo),
  };
}

/** The byte-cache key a finetuned entry's bytes are kept under, and so its
 * `onnxUrl`: `finetuned:finetuned/<name>`. Not a URL anyone fetches; the
 * scheme keeps it apart from the Hugging Face URLs in the same store. */
export function finetunedBytesKey(id: string): string {
  return `${FINETUNED_REPO}:${id}`;
}

// --- field readers --------------------------------------------------------

function asRecord(value: unknown, where: string, what: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new ManifestError(where, `${what} is not a JSON object`);
  return value as Record<string, unknown>;
}

function string(obj: Record<string, unknown>, key: string, where: string): string {
  const value = obj[key];
  if (typeof value !== "string" || !value) throw new ManifestError(where, `${key} is missing`);
  return value;
}

function optionalString(obj: Record<string, unknown>, key: string): string | undefined {
  return typeof obj[key] === "string" ? obj[key] : undefined;
}

function strings(obj: Record<string, unknown>, key: string, where: string): string[] {
  const value = obj[key];
  if (!Array.isArray(value) || value.length === 0 || !value.every((s) => typeof s === "string")) {
    throw new ManifestError(where, `${key} is not a list of names`);
  }
  return value.slice();
}

function numbers(obj: Record<string, unknown>, key: string, where: string): number[] {
  const value = obj[key];
  if (!Array.isArray(value) || value.length === 0 || !value.every((n) => typeof n === "number" && Number.isFinite(n) && n > 0)) {
    throw new ManifestError(where, `${key} is not a list of positive numbers`);
  }
  return value.slice();
}
