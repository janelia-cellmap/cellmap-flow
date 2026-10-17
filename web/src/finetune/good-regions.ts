// Good regions: boxes the user marked (Mark as Good) where the model already
// gets it right, the counterweight to the paint. The port of the server's
// dashboard/routes/finetune/good_regions.py (the region a mark makes),
// finetune/session/manifest.py (good_regions.json, read when a run starts)
// and finetune/data/sampler.py with dataset.py (where a region anchors the
// model) for the browser.
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
// one model output block centred exactly on the region's centre, snapped to
// whole voxels (PatchReader.snap), and holds its unpainted voxels to the
// teacher (the anchor mask, `annotation == 0`, so a scribble inside a good
// region wins: the region stays correctable). anchorBoxes makes the same
// box on the volume's grid, whose blocks are the model's output blocks too,
// so a region marked at the default size, one output block, anchors the
// block it was marked over and no more. A region whose centre is outside
// the volume is skipped, as the server skips one (marked on another volume,
// it would anchor the model to padding).
import { volumeCornerNm } from "../ai-annotate/fill";
import type { PaintStore, PaintVolume } from "../contracts";
import type { GoodRegion } from "./types";

/** The regions' file, at the volume's root (manifest.GOOD_REGIONS_FILENAME). */
export const GOOD_REGIONS_FILE = "good_regions.json";

/** Whether `value` is a region the trainer can place: an id, a label, and a
 * box of three finite numbers each, its size positive. */
export function isGoodRegion(value: unknown): value is GoodRegion {
  if (!value || typeof value !== "object") return false;
  const r = value as Record<string, unknown>;
  const triple = (v: unknown, positive: boolean) => Array.isArray(v) && v.length === 3 && v.every((n) => typeof n === "number" && Number.isFinite(n) && (!positive || n > 0));
  return typeof r.id === "string" && typeof r.label === "string" && triple(r.offset_nm, false) && triple(r.shape_nm, true);
}

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

/** A new region's id: eight hex digits, as the server's uuid4().hex[:8]. */
export function newRegionId(): string {
  const bytes = new Uint8Array(4);
  crypto.getRandomValues(bytes);
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

/** The default size of a region, in nm: one model output block of the
 * volume (good_regions._default_size_nm: the output size times the output
 * voxel size), which is what can be seen and judged on screen and what a
 * training sample computes its loss over. */
export function defaultRegionSizeNm(volume: Pick<PaintVolume, "chunk_size" | "output_voxel_size">): number[] {
  return volume.chunk_size.map((n, a) => n * volume.output_voxel_size[a]);
}

/** The region a mark makes (mark_current_view_good): a box of `sizeNm`
 * centred on `centreNm`, labelled `label` or good-<n + 1> after `count`
 * regions. */
export function regionAt(centreNm: readonly number[], sizeNm: readonly number[], count: number, label?: string, id: string = newRegionId()): GoodRegion {
  return {
    id,
    label: label || `good-${count + 1}`,
    offset_nm: centreNm.map((c, a) => c - sizeNm[a] / 2),
    shape_nm: [...sizeNm],
  };
}

/** A region's anchor: the box of the volume's voxels it holds to the
 * teacher, [lo, hi) per axis (z, y, x), clipped to the volume. */
export interface AnchorBox {
  region: GoodRegion;
  lo: number[];
  hi: number[];
}

/** Rounded to a millionth of a voxel before a floor: the float noise of
 * nm → voxels (100.5 × 8 / 8 come back as 100.49999999999999) must not move
 * a box by a voxel. */
export const steady = (x: number) => Math.round(x * 1e6) / 1e6;

/** The anchor box of each usable region of `regions` on `volume`'s grid, and
 * what was skipped and why (sampler._index_rehearsal): an entry that is not
 * a region, or a region whose centre lies outside the volume.
 *
 * The box is the server's rehearsal patch: one output block (the volume's
 * chunk_size) whose centre is the region's centre snapped so that the block
 * is whole voxels, `lo = floor(c - size/2 + 1/2)` with c the centre in
 * voxels from the volume's corner (voxel v spans [v, v + 1)), then clipped
 * to the volume. */
export function anchorBoxes(volume: Pick<PaintVolume, "chunk_size" | "output_voxel_size" | "dataset_offset_nm" | "dataset_shape_voxels">, regions: readonly unknown[]): { boxes: AnchorBox[]; malformed: unknown[]; outside: GoodRegion[] } {
  const corner = volumeCornerNm(volume.dataset_offset_nm, volume.output_voxel_size);
  const shape = volume.dataset_shape_voxels, size = volume.chunk_size;
  const boxes: AnchorBox[] = [], malformed: unknown[] = [], outside: GoodRegion[] = [];
  for (const region of regions) {
    if (!isGoodRegion(region)) {
      malformed.push(region);
      continue;
    }
    const centre = region.offset_nm.map((o, a) => (o + region.shape_nm[a] / 2 - corner[a]) / volume.output_voxel_size[a]);
    if (centre.some((c, a) => c < 0 || c >= shape[a])) {
      outside.push(region);
      continue;
    }
    const start = centre.map((c, a) => Math.floor(steady(c - size[a] / 2 + 0.5)));
    const lo = start.map((s) => Math.max(s, 0)), hi = start.map((s, a) => Math.min(s + size[a], shape[a]));
    if (hi.some((h, a) => h <= lo[a])) {
      outside.push(region);
      continue;
    }
    boxes.push({ region, lo, hi });
  }
  return { boxes, malformed, outside };
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
