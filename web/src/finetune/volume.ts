// An annotation volume in the browser: planned over the open dataset for a
// model (planVolume), written as zarr v2 files into the paint store
// (volumeFiles), and read back for training (readVolume, the painted chunks).
// The port of cellmap_flow/finetune/session/volume.py (plan_volume,
// create_volume_zarr, read_volume) for the browser, where the files are the
// paint store's and the grid is the one the model stage serves on
// (infer/geometry.ts); the labels' dtype follows
// cellmap_flow/dashboard/routes/finetune/common.py's annotation_dtype_for.
// See PLAN_browser_finetune.md, "The volume's layout".
import type { Chain, DatasetInfo, ModelEntry, PaintStore, PaintVolume } from "../contracts";
import { chooseRawLevel, geometryOf, inputShape, outputLevel, outputShape } from "../infer/geometry";
import { isSegmentation } from "../infer/post";
import { stamp } from "../shared/format";
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

/** A new volume over `dataset` for `entry`, through `chain`: plan_volume,
 * on the grid the served prediction has.
 *
 * The server snaps each of the model's voxel sizes to the raw level closest
 * to it (closest_raw_scale) and lays the volume on the output level's grid
 * from the data's corner, padded to whole chunks. Here the grid is the one
 * engine.ts's serve() builds for the model's view, so a chunk of paint is
 * exactly a chunk of the prediction layer it is painted over: the raw level
 * chooseRawLevel picks for the model's input voxel size, read as if it were
 * at that size, and the output level outputLevel derives from it (its voxel
 * size the raw level's times the model's ratio, its shape the raw level's
 * floor-divided by the ratio, its origin the raw level's corner plus half an
 * output voxel). DatasetInfo's levels are in nm (datasetInfoFrom), so the
 * volume is in nm, where serve() gives outputLevel the engine's level in the
 * source's own units for the view. input_voxel_size is the raw level's, as
 * the server's is the closest raw level's; the model's own are kept as the
 * claimed sizes, for provenance.
 *
 * dataset_offset_nm is the output level's origin. chunkmirage reads a
 * level's OME translation as its origin (web/chunkmirage/web/src/ome.ts,
 * openImage), and a translation is voxel 0's centre, which is what the
 * server stores and writes back as the volume's own translation (volume.py's
 * module docstring, volume_corner_nm), so the paint is drawn where the
 * prediction is. The extent is the level's shape rounded up to whole chunks,
 * as the server's _grid pads it, so that the last chunk along each axis is a
 * whole sample.
 *
 * Refused for a dataset with more than one channel, as serve() refuses it:
 * the browser mode reads single-channel volumes. */
export function planVolume(dataset: DatasetInfo, entry: ModelEntry, chain: Chain, options: { id?: string; now?: Date } = {}): PaintVolume {
  if (dataset.channels > 1) {
    throw new Error(`${dataset.url} has ${dataset.channels} channels; the browser mode paints and trains on single-channel volumes (it does not select a channel yet)`);
  }
  const g = geometryOf(entry);
  const raw = dataset.levels[chooseRawLevel(dataset.levels, g.input_voxel_size)];
  const level = outputLevel(raw, g);
  const chunk = outputShape(g);
  return {
    id: options.id ?? newVolumeId(options.now),
    type: "annotation_volume",
    model_name: entry.name,
    model_id: entry.id,
    dataset_path: dataset.url,
    chunk_size: chunk,
    input_size: inputShape(g),
    output_voxel_size: level.voxel,
    input_voxel_size: [...raw.voxel],
    claimed_output_voxel_size: g.output_voxel_size,
    claimed_input_voxel_size: g.input_voxel_size,
    dataset_offset_nm: level.origin,
    dataset_shape_voxels: level.shape.map((n, a) => Math.ceil(n / chunk[a]) * chunk[a]),
    dtype: annotationDtypeFor(entry, chain),
    input_norm: chain.input_norm.map((step) => ({ ...step })),
    postprocess: chain.postprocess.map((step) => ({ ...step })),
    created_at: (options.now ?? new Date()).toISOString(),
  };
}

/** The labels' dtype of a new volume for `entry` through `chain`: common.py's
 * annotation_dtype_for. The server asks what the model trains on
 * (autodetect_output_type) and gives uint16 to a target trained on
 * instances, whose paint numbers each object (282 of a Cellpose model's
 * masks in one box did not fit a uint8 volume): affinities, which, when the
 * model type declares nothing, it tells by a channel name containing "_aff";
 * and Cellpose's flows, which only a model type declares. Everything else is
 * uint8, background and foreground. The channel-name test is followed as it
 * stands (hasAffinityChannels); the browser has no model types or scripts
 * to ask, and no flow model is on Hugging Face. One rule is the browser's
 * own: a chain that serves labels (isSegmentation: a threshold, an affinity
 * segmentation) is painted over object by object, each with its id, so its
 * volume holds ids too. */
export function annotationDtypeFor(entry: ModelEntry, chain: Chain): PaintVolume["dtype"] {
  return hasAffinityChannels(entry) || isSegmentation(chain.postprocess) === true ? "uint16" : "uint8";
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
