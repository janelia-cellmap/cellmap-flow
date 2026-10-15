// An annotation volume in the browser: planned over the open dataset for a
// model (planVolume), written as zarr v2 files into the paint store
// (volumeFiles), and read back for training (readVolume, the painted chunks).
// The port of cellmap_flow/finetune/session/volume.py (plan_volume,
// create_volume_zarr, read_volume) for the browser, where the files are the
// paint store's and the grid is the one the model stage serves on
// (infer/geometry.ts). See PLAN_browser_finetune.md, "The volume's layout".
import type { Chain, DatasetInfo, ModelEntry, PaintStore, PaintVolume } from "../contracts";
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
  const pad = (n: number) => String(n).padStart(2, "0");
  const stamp = `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}-${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
  return `vol-${hex}-${stamp}`;
}

function randomHex(n: number): string {
  const bytes = new Uint8Array(Math.ceil(n / 2));
  crypto.getRandomValues(bytes);
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("").slice(0, n);
}

/** A new volume over `dataset` for `entry`, through `chain`: see plan_volume. */
export function planVolume(_dataset: DatasetInfo, _entry: ModelEntry, _chain: Chain, _options: { id?: string; now?: Date } = {}): PaintVolume {
  throw new Error("planVolume is not implemented yet");
}

/** The volume's zarr v2 files, path → JSON text: see create_volume_zarr. */
export function volumeFiles(_volume: PaintVolume): Record<string, string> {
  throw new Error("volumeFiles is not implemented yet");
}

/** Write a new volume's files into the store. */
export async function createVolume(store: PaintStore, volume: PaintVolume): Promise<void> {
  const encoder = new TextEncoder();
  for (const [path, text] of Object.entries(volumeFiles(volume))) await store.put(volume.id, path, encoder.encode(text));
}

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
