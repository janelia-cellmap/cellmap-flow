// Good regions: boxes the user marked (Mark as Good) where the model already
// gets it right, the counterweight to the paint. Their rules are the
// server's, cellmap_flow/portable/regions.py, run in the engine's Pyodide
// worker (volume_calls.py): the region a mark makes and its size
// (good_regions.py's mark-view), which stored entries are regions (what the
// layer draws and the trainer places), and where a region anchors the model
// on the volume's grid (the server trainer's rehearsal patch,
// finetune/data/sampler.py with PatchReader.snap). This file keeps the
// regions' file and turns the anchors into the browser trainer's per-chunk
// masks (anchorChunks), which only its trainer has.
//
// Where they are kept. The server keeps a session's regions in
// good_regions.json beside its corrections directory; a browser session is
// one volume (routes/finetune.ts), so the same file sits at the volume's root
// in the paint store, beside its zarr, as the AI annotation's audit log
// does: it travels with the volume's zip, goes when the volume is deleted,
// and a volume resumed keeps its regions. The file is a JSON list of
// {id, label, offset_nm, shape_nm}, the server's shape, so a downloaded
// volume's regions read as the server's do.
//
// What a region holds. The server draws a rehearsal patch for a region:
// one model output block centred on the region's centre, snapped to whole
// voxels, and holds its unpainted voxels to the teacher (the anchor mask,
// `annotation == 0`, so a scribble inside a good region wins: the region
// stays correctable). anchorBoxes makes the same box on the volume's grid,
// whose blocks are the model's output blocks too, so a region marked at the
// default size, one output block, anchors the block it was marked over and
// no more. A region whose centre is outside the volume is skipped, as the
// server skips one (marked on another volume, it would anchor the model to
// padding).
import type { Inference, PaintStore, PaintVolume } from "../contracts";
import type { RegionBox } from "../viewer/layers";
import type { GoodRegion } from "./types";
import { VOLUME_CALLS } from "./volume";

/** The regions' file, at the volume's root (manifest.GOOD_REGIONS_FILENAME). */
export const GOOD_REGIONS_FILE = "good_regions.json";

/** The regions of volume `id`, as stored ([] when there are none, or the
 * file is not a list: manifest.load_good_regions's rule). An entry that is
 * not a region is kept as it is, so that a list rewritten after a mark or a
 * delete loses nothing; anchorBoxes skips it with a word. */
export async function loadGoodRegions(store: PaintStore, id: string): Promise<unknown[]> {
  const bytes = await store.get(id, GOOD_REGIONS_FILE);
  if (!bytes) return [];
  try {
    const regions: unknown = JSON.parse(new TextDecoder().decode(bytes));
    return Array.isArray(regions) ? regions : [];
  } catch {
    return [];
  }
}

/** Replace volume `id`'s regions with `regions` (the store's put is whole-file, so atomic as the server's os.replace is). */
export async function saveGoodRegions(store: PaintStore, id: string, regions: unknown[]): Promise<void> {
  await store.put(id, GOOD_REGIONS_FILE, new TextEncoder().encode(JSON.stringify(regions, null, 2)));
}

/** The good_regions layer's box of each stored entry, or null for one that
 * is not a region (the Python's region_boxes, by the rule the server's layer
 * draws by), so the layer's ids stay the entries' indices. */
export async function regionBoxes(infer: Pick<Inference, "python">, regions: readonly unknown[]): Promise<(RegionBox | null)[]> {
  return regions.length ? infer.python<(RegionBox | null)[]>(`${VOLUME_CALLS}:region_boxes`, { regions }) : [];
}

/** The region a mark makes (mark_current_view_good), by the Python's
 * mark_region: a box of the size `sizeNm` asks for, as the request sent it
 * (one model output block of `volume` when none is, 896 nm a side without a
 * volume; one number for every axis, or three), centred on `centreNm`,
 * labelled `label` or good-<n + 1> after the `stored` regions; with the
 * layer's boxes of the stored regions and it. A size that is not one is
 * answered as the sentence it is refused with. */
export function markRegion(
  infer: Pick<Inference, "python">,
  stored: readonly unknown[],
  centreNm: readonly number[],
  sizeNm: unknown,
  label: unknown,
  volume: Pick<PaintVolume, "chunk_size" | "output_voxel_size"> | null,
): Promise<{ region: GoodRegion; boxes: (RegionBox | null)[] } | { error: string }> {
  return infer.python(`${VOLUME_CALLS}:mark_region`, {
    centre_nm: centreNm,
    size_nm: sizeNm ?? null,
    label: label ?? null,
    regions: stored,
    volume: volume && { chunk_size: volume.chunk_size, output_voxel_size: volume.output_voxel_size },
  });
}

/** A stored entry that is a region: its box, and its id and label as stored
 * (either may be missing: neither is needed to place a region). */
export type StoredRegion = Pick<GoodRegion, "offset_nm" | "shape_nm"> & Partial<Pick<GoodRegion, "id" | "label">>;

/** A region's anchor: the box of the volume's voxels it holds to the
 * teacher, [lo, hi) per axis (z, y, x), clipped to the volume. */
export interface AnchorBox {
  region: StoredRegion;
  lo: number[];
  hi: number[];
}

/** The anchor box of each usable region of `regions` on `volume`'s grid, and
 * what was skipped and why (sampler._index_rehearsal): an entry that is not
 * a region, or a region whose centre lies outside the volume. The Python's
 * anchor_boxes: the server's rehearsal patch, one output block (the
 * volume's chunk_size) whose start is the region's centre snapped so that
 * the block is whole voxels, clipped to the volume. */
export async function anchorBoxes(
  infer: Pick<Inference, "python">,
  volume: Pick<PaintVolume, "chunk_size" | "output_voxel_size" | "dataset_offset_nm" | "dataset_shape_voxels">,
  regions: readonly unknown[],
): Promise<{ boxes: AnchorBox[]; malformed: unknown[]; outside: StoredRegion[] }> {
  if (!regions.length) return { boxes: [], malformed: [], outside: [] };
  const grid = { chunk_size: volume.chunk_size, output_voxel_size: volume.output_voxel_size, dataset_offset_nm: volume.dataset_offset_nm, dataset_shape_voxels: volume.dataset_shape_voxels };
  // Each by its index in `regions`, which travel as they are stored.
  const placed = await infer.python<{ boxes: { region: number; lo: number[]; hi: number[] }[]; malformed: number[]; outside: number[] }>(`${VOLUME_CALLS}:anchor_boxes`, { regions, volume: grid });
  return {
    boxes: placed.boxes.map(({ region, lo, hi }) => ({ region: regions[region] as StoredRegion, lo, hi })),
    malformed: placed.malformed.map((i) => regions[i]),
    outside: placed.outside.map((i) => regions[i] as StoredRegion),
  };
}

/** The chunks the boxes reach, each with its voxels inside any box (a mask
 * of the chunk's voxels in C order, 1 inside) and how many there are, in
 * the store's chunk order (sorted by index, as paintedChunkIndices lists). */
export function anchorChunks(volume: Pick<PaintVolume, "chunk_size">, boxes: readonly AnchorBox[]): { index: number[]; inRegion: Uint8Array; voxels: number }[] {
  const size = volume.chunk_size, [sz, sy, sx] = size;
  const chunks = new Map<string, { index: number[]; inRegion: Uint8Array; voxels: number }>();
  for (const { lo, hi } of boxes) {
    const first = lo.map((l, a) => Math.floor(l / size[a])), last = hi.map((h, a) => Math.floor((h - 1) / size[a]));
    for (let cz = first[0]; cz <= last[0]; cz++) for (let cy = first[1]; cy <= last[1]; cy++) for (let cx = first[2]; cx <= last[2]; cx++) {
      const index = [cz, cy, cx], key = index.join(".");
      let chunk = chunks.get(key);
      if (!chunk) chunks.set(key, (chunk = { index, inRegion: new Uint8Array(sz * sy * sx), voxels: 0 }));
      // The box within this chunk, in the chunk's own voxels.
      const from = index.map((i, a) => Math.max(lo[a] - i * size[a], 0)), to = index.map((i, a) => Math.min(hi[a] - i * size[a], size[a]));
      for (let z = from[0]; z < to[0]; z++) for (let y = from[1]; y < to[1]; y++) for (let x = from[2]; x < to[2]; x++) {
        const v = (z * sy + y) * sx + x;
        if (!chunk.inRegion[v]) { chunk.inRegion[v] = 1; chunk.voxels++; }
      }
    }
  }
  return [...chunks.values()].sort((a, b) => compareIndex(a.index, b.index));
}

/** Grid indices in order: z, then y, then x. */
export function compareIndex(a: readonly number[], b: readonly number[]): number {
  for (let i = 0; i < Math.min(a.length, b.length); i++) if (a[i] !== b[i]) return a[i] - b[i];
  return a.length - b.length;
}
