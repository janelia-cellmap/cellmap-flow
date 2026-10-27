// An annotation volume in the browser: planned over the open dataset for a
// model (planVolume, by cellmap_flow/portable/volume.py, the Python the
// server plans with, run in the engine's Pyodide worker), or before any model
// on the raw's grid (planPaintVolume: not trained on, its paint carried onto
// a model's grid by Use this model), written as zarr v2
// files into the paint store (volumeFiles), read back for training
// (readVolume, the painted chunks), and retired, kept but no longer the
// session's, once Use this model has carried its paint to another
// (supersedeVolume, currentVolume). This file reads what the plan is made
// from (the raw level the model reads, the model's blocks) and writes and
// reads the files; the plan, the grid and the labels' dtype, is the
// Python's. See PLAN_browser_finetune.md, "The volume's layout".
import { volumeCornerNm } from "../ai-annotate/fill";
import type { Chain, DatasetInfo, Inference, ModelEntry, PaintStore, PaintVolume } from "../contracts";
import { chooseRawLevel, geometryOf } from "../infer/geometry";
import { stamp } from "../shared/format";
import { decodeState } from "../viewer/share";
import { pointerNm } from "../viewer/state";
import type { PaintedChunk } from "./types";

/** The directory of the labels' chunks within a volume: zarr v2, one level. */
export const CHUNK_DIR = "annotation/s0/";

/** The chunk key of grid index (z, y, x): "." separated, as the server's volumes and MinIO have them. */
export function chunkKey(index: number[]): string {
  return index.join(".");
}

/** The grid index a chunk key names, or null for a name that is not one (`.zarray`, say). */
export function parseChunkKey(name: string): number[] | null {
  if (!/^\d+(\.\d+)*$/.test(name)) return null;
  return name.split(".").map(Number);
}

/** `vol-<8 hex>-<YYYYmmdd-HHMMSS>`, as the server's new_volume_id. */
export function newVolumeId(now: Date = new Date(), hex: string = randomHex(8)): string {
  return `vol-${hex}-${stamp(now)}`;
}

function randomHex(n: number): string {
  const bytes = new Uint8Array(Math.ceil(n / 2));
  crypto.getRandomValues(bytes);
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("").slice(0, n);
}

// ---------------------------------------------------------------------------
// Planning a volume
// ---------------------------------------------------------------------------

/** The portable Python's entry points for volumes and good regions (cellmap_flow/portable/volume_calls.py), by module path. */
export const VOLUME_CALLS = "cellmap_flow.portable.volume_calls";

/** What plan_volume answers when it plans: the plan's fields of a PaintVolume. */
type Plan = Pick<PaintVolume, "output_voxel_size" | "input_voxel_size" | "claimed_output_voxel_size" | "claimed_input_voxel_size" | "chunk_size" | "input_size" | "dataset_offset_nm" | "dataset_shape_voxels" | "dtype">;

/** What plan_paint_volume answers: a Plan whose claimed sizes are null, as no model claims one. */
type PaintPlan = Omit<Plan, "claimed_output_voxel_size" | "claimed_input_voxel_size"> & { claimed_output_voxel_size: null; claimed_input_voxel_size: null };

/** A new volume over `dataset` for `entry`, through `chain`: the server's
 * plan_volume, which is cellmap_flow/portable/volume.py's, called in the
 * engine's Pyodide worker (`infer.python`). What it is given is what the
 * browser read: the raw level the model reads (chooseRawLevel, the server's
 * "floor" level choice, as engine.ts's serve() picks it), its voxel-0
 * corner, and the model's voxel sizes and blocks (geometryOf). The Python
 * answers the grid, the one the model's prediction is served on (the
 * level's corner, its voxel times the model's ratio: outputLevel's
 * arithmetic, so a chunk of paint is a chunk of the prediction layer it is
 * painted over), the extent (the level's, rounded up to whole output voxels
 * and padded to whole chunks) and the labels' dtype (uint16 for a model
 * trained on instances, here an affinity model: hasAffinityChannels, as the
 * browser trains it). A refusal comes back as its sentence, thrown here.
 *
 * Refused for a dataset with more than one channel, as serve() refuses it:
 * the browser mode reads single-channel volumes. */
export async function planVolume(infer: Pick<Inference, "python">, dataset: DatasetInfo, entry: ModelEntry, chain: Chain, options: VolumeOptions = {}): Promise<PaintVolume> {
  refuseChannels(dataset);
  const model = geometryOf(entry);
  const raw = dataset.levels[chooseRawLevel(dataset.levels, model.input_voxel_size)];
  const plan = await infer.python<Plan | { error: string }>(`${VOLUME_CALLS}:plan_volume`, {
    level: levelOf(raw),
    model,
    output_type: hasAffinityChannels(entry) ? "affinities" : "binary",
  });
  if ("error" in plan) throw new Error(plan.error);
  return volumeOf(plan, dataset, entry, chain, options);
}

/** A new volume's id and creation time, by default a fresh id and now (tests pin them). */
export interface VolumeOptions { id?: string; now?: Date }

/** A new volume over `dataset` painted before any model (a "paint-first"
 * volume), through `chain`: the server's plan_paint_volume, which is
 * cellmap_flow/portable/volume.py's, in the engine's Pyodide worker, on the
 * raw's grid at `voxelSize` (nm, z, y, x). The level the plan is over is
 * the one chooseRawLevel picks for that size (the level at it, else the
 * finest not coarser than it), as the server's raw_level opens it; with no
 * voxel size, the level the viewer shows at a screen pixel of `pixelNm`
 * (shownVoxelSize; null, the finest). Its model_name and model_id are null:
 * it is not trained on (trainingRefusal), and Which model to start from's
 * Use this model carries its paint onto a model's grid. Refused as
 * planVolume is for a dataset of more than one channel, and with the
 * Python's sentence for a voxel size it cannot plan at. */
export async function planPaintVolume(infer: Pick<Inference, "python">, dataset: DatasetInfo, voxelSize: number[] | null, chain: Chain, options: VolumeOptions & { pixelNm?: number | null } = {}): Promise<PaintVolume> {
  refuseChannels(dataset);
  const size = voxelSize ?? (await shownVoxelSize(infer, dataset, options.pixelNm ?? null));
  const raw = dataset.levels[chooseRawLevel(dataset.levels, size)];
  const plan = await infer.python<PaintPlan | { error: string }>(`${VOLUME_CALLS}:plan_paint_volume`, { level: levelOf(raw), voxel_size: size });
  if ("error" in plan) throw new Error(plan.error);
  // No model claims a voxel size: the attrs leave them out, as the server's create_volume_zarr does.
  const { claimed_output_voxel_size: _out, claimed_input_voxel_size: _in, ...grid } = plan;
  return volumeOf(grid, dataset, null, chain, options);
}

/** The voxel size (nm, z, y, x) of `dataset`'s level the viewer shows at a
 * screen pixel of `pixelNm` (null: not known, the finest), by the portable
 * Python's shown_level, as the server's shown_voxel_size picks it. */
export async function shownVoxelSize(infer: Pick<Inference, "python">, dataset: DatasetInfo, pixelNm: number | null): Promise<number[]> {
  const shown = await infer.python<{ index: number } | { error: string }>(`${VOLUME_CALLS}:shown_level`, {
    voxel_sizes: dataset.levels.map((level) => level.voxel),
    pixel_nm: pixelNm,
  });
  if ("error" in shown) throw new Error(shown.error);
  return dataset.levels[shown.index].voxel.slice();
}

/** The size in nm of one screen pixel of the viewer whose own URL is
 * `viewerUrl` (ViewerController.viewerUrl: its state in the `#!` hash), as
 * the server's viewer/raw.py view_of reads it: the state's
 * crossSectionScale (voxels of its coordinate space a pixel spans) times
 * the smallest of its z, y and x scales in nm. Null when the URL holds no
 * state, or the state no zoom or lengths (a viewer that has not reported
 * them yet). */
export function viewPixelNm(viewerUrl: string | null): number | null {
  const hash = viewerUrl ? viewerUrl.replace(/^[^#]*/, "") : "";
  const state = decodeState(hash);
  const zoom = state?.crossSectionScale;
  // One voxel along every dimension, in nm: each of z, y and x's scale.
  const scales = pointerNm(Object.keys(state?.dimensions ?? {}).map(() => 1), state?.dimensions);
  if (typeof zoom !== "number" || !Number.isFinite(zoom) || zoom <= 0 || !scales) return null;
  return zoom * Math.min(...scales);
}

/** Why `volume` is not trained on, or null when it can be: the portable
 * Python's training_refusal (a volume painted before any model), the
 * sentence the server's submit refuses it with. */
export async function trainingRefusal(infer: Pick<Inference, "python">, volume: PaintVolume): Promise<string | null> {
  const answer = await infer.python<{ error?: string }>(`${VOLUME_CALLS}:training_refusal`, { volume: { model_name: volume.model_name } });
  return answer.error ?? null;
}

/** Refused for a dataset with more than one channel, as serve() refuses it:
 * the browser mode reads single-channel volumes. */
function refuseChannels(dataset: DatasetInfo): void {
  if (dataset.channels > 1) {
    throw new Error(`${dataset.url} has ${dataset.channels} channels; the browser mode paints and trains on single-channel volumes (it does not select a channel yet)`);
  }
}

/** A raw level as the plans take it: its shape, voxel size and voxel-0 lower corner (the engine's origin is the centre). */
function levelOf(raw: DatasetInfo["levels"][number]): { shape: number[]; voxel: number[]; corner: number[] } {
  return { shape: raw.shape, voxel: raw.voxel, corner: volumeCornerNm(raw.origin, raw.voxel) };
}

/** The volume of `plan` over `dataset`, for `entry` (null: for no model), with `chain` and the options' id and time. */
function volumeOf(plan: Plan, dataset: DatasetInfo, entry: ModelEntry | null, chain: Chain, options: VolumeOptions): PaintVolume {
  return {
    id: options.id ?? newVolumeId(options.now),
    type: "annotation_volume",
    model_name: entry ? entry.name : null,
    model_id: entry ? entry.id : null,
    dataset_path: dataset.url,
    ...plan,
    input_norm: chain.input_norm.map((step) => ({ ...step })),
    postprocess: chain.postprocess.map((step) => ({ ...step })),
    created_at: (options.now ?? new Date()).toISOString(),
  };
}

/** Whether a model's channels are affinities (`mito_aff_z`, `mito_aff_y`,
 * `mito_aff_x`), by the server's test: a channel name containing "_aff"
 * (common.py's autodetect_output_type). */
export function hasAffinityChannels(entry: ModelEntry): boolean {
  return entry.manifest.channels.some((channel) => channel.includes("_aff"));
}

// ---------------------------------------------------------------------------
// The volume's files
// ---------------------------------------------------------------------------

/** The labels' dtype as zarr v2's `.zarray` spells it: uint8 has no byte
 * order, uint16 is little-endian, which is how the fork's bytes codec writes
 * a chunk and labelsOf reads one. */
const ZARR_DTYPE: Record<PaintVolume["dtype"], string> = { uint8: "|u1", uint16: "<u2" };

/** The volume's zarr v2 files, path → JSON text: what create_volume_zarr
 * writes, metadata only, no chunks. The root group's `.zgroup` and
 * `.zattrs` (the volume's record), the `annotation` group with the OME
 * multiscales the viewer places the labels by (one level, `s0`, at the
 * output voxel size, translated to dataset_offset_nm), and `s0`'s `.zarray`:
 * the extent padded to whole chunks, one chunk per model output block, "."
 * between the indices of a chunk's key, 0 for a chunk that is not there.
 *
 * Uncompressed, where the server's volumes are Blosc zstd: the fork encodes
 * a painted chunk with the array's codecs (the bytes codec alone here) and
 * the trainer decodes nothing; zarr-python reads a downloaded volume as it
 * reads the server's. */
export function volumeFiles(volume: PaintVolume): Record<string, string> {
  const multiscales = [{
    version: "0.4",
    name: "annotation",
    axes: ["z", "y", "x"].map((name) => ({ name, type: "space", unit: "nanometer" })),
    datasets: [{
      path: "s0",
      coordinateTransformations: [
        { type: "scale", scale: volume.output_voxel_size },
        { type: "translation", translation: volume.dataset_offset_nm },
      ],
    }],
  }];
  const zarray = {
    zarr_format: 2,
    shape: volume.dataset_shape_voxels,
    chunks: volume.chunk_size,
    dtype: ZARR_DTYPE[volume.dtype],
    compressor: null,
    fill_value: 0,
    order: "C",
    filters: null,
    dimension_separator: ".",
  };
  return {
    ".zgroup": json({ zarr_format: 2 }),
    ".zattrs": json(rootAttrs(volume)),
    "annotation/.zgroup": json({ zarr_format: 2 }),
    "annotation/.zattrs": json({ multiscales }),
    [`${CHUNK_DIR}.zarray`]: json(zarray),
  };
}

/** The root `.zattrs`: every field of the volume, the server's attrs first
 * and in the order create_volume_zarr writes them (resample only when on,
 * as there), then the browser's own (model_id, dtype, id; read_volume
 * ignores what it does not know), so a downloaded volume reads like one
 * made on the server. */
function rootAttrs(volume: PaintVolume): Record<string, unknown> {
  const {
    type, model_name, dataset_path, chunk_size, output_voxel_size, input_size, input_voxel_size, dataset_offset_nm, dataset_shape_voxels,
    created_at, resample, claimed_output_voxel_size, claimed_input_voxel_size, input_norm, postprocess, ...browser
  } = volume;
  return {
    type, model_name, dataset_path, chunk_size, output_voxel_size, input_size, input_voxel_size, dataset_offset_nm, dataset_shape_voxels, created_at,
    ...(resample ? { resample: true } : {}),
    claimed_output_voxel_size, claimed_input_voxel_size, input_norm, postprocess,
    ...browser,
  };
}

function json(value: unknown): string {
  return JSON.stringify(value, null, 2);
}

/** Write a new volume's files into the store. */
export async function createVolume(store: PaintStore, volume: PaintVolume): Promise<void> {
  const encoder = new TextEncoder();
  for (const [path, text] of Object.entries(volumeFiles(volume))) await store.put(volume.id, path, encoder.encode(text));
}

// ---------------------------------------------------------------------------
// Reading a volume back
// ---------------------------------------------------------------------------

/** The volume `id` describes itself as, from its root `.zattrs`; null when
 * the store has no such volume or the file is not an annotation volume's. */
export async function readVolume(store: PaintStore, id: string): Promise<PaintVolume | null> {
  const bytes = await store.get(id, ".zattrs");
  if (!bytes) return null;
  let attrs: unknown;
  try {
    attrs = JSON.parse(new TextDecoder().decode(bytes));
  } catch {
    return null;
  }
  if (!attrs || typeof attrs !== "object" || (attrs as { type?: unknown }).type !== "annotation_volume") return null;
  return { ...(attrs as PaintVolume), id };
}

/** Retire volume `id` in favour of `by`, the volume Use this model carried
 * its paint to: its root `.zattrs` gains `superseded_by`, and nothing else
 * of it changes. A session has one current volume, the one it trains on, so
 * a superseded one is no longer listed as a session, and resuming it
 * resumes the volume that took its place (currentVolume): David found the
 * Finetune tab listing every copy of one paint, on different grids, and
 * counting the chunks of each. It is kept, never deleted: its paint as it
 * was painted, on its own grid, is what the carry was made from. The server
 * moves such a volume into its corrections directory's superseded/; here a
 * volume's id is its key prefix in the store (paintKey), so it stays where
 * it is, marked. Refused for a volume the store does not have. */
export async function supersedeVolume(store: PaintStore, id: string, by: string): Promise<void> {
  const volume = await readVolume(store, id);
  if (!volume) throw new Error(`No annotation volume ${id} in this browser to retire`);
  await store.put(id, ".zattrs", new TextEncoder().encode(json({ ...volume, superseded_by: by })));
}

/** The volume `id` stands for now: its own when it is current, else the one
 * that took its place (superseded_by), followed to the last; null when the
 * store does not have `id`, or the last was deleted since (Delete volume
 * deletes the one named, not those it took the place of). Each id is
 * followed once, so a store whose
 * volumes name each other in a ring answers the last one reached rather
 * than looping. */
export async function currentVolume(store: PaintStore, id: string): Promise<PaintVolume | null> {
  const followed = new Set<string>();
  let volume = await readVolume(store, id);
  while (volume?.superseded_by && !followed.has(volume.id)) {
    followed.add(volume.id);
    volume = await readVolume(store, volume.superseded_by);
  }
  return volume;
}

/** The grid indices of the chunks painted so far (the chunk files present), sorted. */
export async function paintedChunkIndices(store: PaintStore, id: string): Promise<number[][]> {
  const indices: number[][] = [];
  for (const path of await store.list(id, CHUNK_DIR)) {
    const index = parseChunkKey(path.slice(CHUNK_DIR.length));
    if (index) indices.push(index);
  }
  return indices;
}

/** A chunk's bytes as the volume's labels: the dtype `.zarray` declares, little-endian, C order. */
export function labelsOf(volume: PaintVolume, bytes: Uint8Array): Uint8Array | Uint16Array {
  const voxels = volume.chunk_size.reduce((a, b) => a * b, 1);
  if (volume.dtype === "uint16") {
    if (bytes.byteLength !== voxels * 2) throw new Error(`a chunk of ${volume.id} has ${bytes.byteLength} bytes, not the ${voxels * 2} of ${volume.chunk_size.join("×")} uint16 voxels`);
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const out = new Uint16Array(voxels);
    for (let i = 0; i < voxels; i++) out[i] = view.getUint16(i * 2, true);
    return out;
  }
  if (bytes.byteLength !== voxels) throw new Error(`a chunk of ${volume.id} has ${bytes.byteLength} bytes, not the ${voxels} of ${volume.chunk_size.join("×")} uint8 voxels`);
  return new Uint8Array(bytes);  // a copy: the store's bytes stay its own
}

/** The painted chunk at `index`, or null when it was never painted. */
export async function readPaintedChunk(store: PaintStore, volume: PaintVolume, index: number[]): Promise<PaintedChunk | null> {
  const bytes = await store.get(volume.id, CHUNK_DIR + chunkKey(index));
  if (!bytes) return null;
  return { index, shape: volume.chunk_size.slice(), labels: labelsOf(volume, bytes) };
}
