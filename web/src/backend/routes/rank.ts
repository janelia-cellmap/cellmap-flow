// Which model to start from: the Finetune tab ranks the models the browser
// can run by how well each one's prediction already matches the session
// volume's paint (cellmap_flow/portable/rank.py says how a model is scored:
// the Python both modes run, called here in the engine's Pyodide worker
// through Inference.python), so the one to finetune is the one with the
// least to learn. This file reads the paint and the model's output and
// hands them to that Python; nothing of the score is decided here.
//
// GET /api/finetune/rank/candidates lists them: the catalog's models,
// finetuned ones among them, with the session's volume and the model it is
// for. POST /api/finetune/rank/score {name} scores one, as the tab asks for
// them one at a time and shows each as it comes. A model already served is
// read as it is served, its chunks from the engine's caches where the viewer
// has had them computed; any other is served for the score alone, with its
// manifest's normalizers and no layer, and released after, so a ranking
// leaves the Models tab and the GPU as they were (its download stays in the
// catalog's byte cache, and Show serves it at once). Where the raw under the
// paint is narrow, a model is scored a second time through its normalizer
// stretched to the raw's window, served for that alone (portable/rank.py
// says when that is kept). POST /api/finetune/rank/use {name, window?}
// makes the one picked the model to finetune, the window its score chose
// (if any) first made the session's input normalization:
// served with its layer (as the Models tab's Add serves it), a volume made
// for it with the session volume's paint carried onto its grid
// (cellmap_flow/portable/carry.py, the same way) and its good regions, and that volume the session's,
// its paint layer in the old one's place; the old volume stays in the paint
// store.
import { TYPED } from "../../../chunkmirage/web/src/asread";
import { inputBox } from "../../../chunkmirage/web/src/stages";
import { volumeCornerNm } from "../../ai-annotate/fill";
import { percentile } from "../../ai-annotate/mask";
import type { AppContext, Chain, DatasetInfo, ModelEntry, PaintVolume, ServedModel, Step } from "../../contracts";
import { FINETUNED_REPO } from "../../contracts";
import { loadGoodRegions, saveGoodRegions } from "../../finetune/good-regions";
import { readsOneSlice } from "../../finetune/modes";
import type { PaintedChunk } from "../../finetune/types";
import { targetTransformFor } from "../../finetune/target-call";
import { CHUNK_DIR, chunkKey, createVolume, hasAffinityChannels, paintedChunkIndices, planVolume, readPaintedChunk, readVolume } from "../../finetune/volume";
import { RANK_BLOCKS, WINDOW_MARGIN } from "../../generated/cellmap-flow";
import { chooseRawLevel, geometryOf, outputShape, ratio } from "../../infer/geometry";
import { servedIndices, servedLevel } from "../../label-it/prediction";
import { probability } from "../../label-it/segment";
import { entryFor } from "../../models/manifest";
import { PAINT_TAB, paintLayer, paintLayerName, paintLayerUrl } from "../../viewer/layers";
import type { Router } from "../router";
import { badRequest, json, message, objectBody } from "../router";
import { blobFor, pageBase, serveModels, setChain } from "../serving";

const NO_DATASET = "No dataset is open. Enter its URL in the viewer panel first; the volume is painted on it.";

/** The most raw voxels the window of the raw under the paint is taken from (every few of them). */
const RAW_SAMPLE = 1_000_000;

export function rankRoutes(router: Router): void {
  router.get("/api/finetune/rank/candidates", async (_req, _params, ctx) => {
    const volume = await sessionVolume(ctx);
    const served = new Set(ctx.infer.served().map((s) => s.entry.id));
    const entries = await ctx.catalog.list();
    return json(200, {
      success: true,
      volume: volume ? { id: volume.id, model_name: volume.model_name, painted_chunks: (await paintedChunkIndices(ctx.paint, volume.id)).length } : null,
      candidates: entries.map((e) => ({
        name: e.name, repo: e.repo, folder: e.folder, task: e.manifest.task, channels: e.manifest.channels,
        voxel_size_nm: e.manifest.output_voxel_size_nm, dims: readsOneSlice(e) ? 2 : 3,
        finetuned: e.repo === FINETUNED_REPO, served: served.has(e.id), volume_model: volume?.model_id === e.id,
      })),
    });
  });

  // POST /api/finetune/rank/score {name}: the model's score on the paint
  // (rank_score's, portable/rank.py), how long it took, and a
  // note when some painted chunks lie where the model's level does not
  // reach. Refused without a dataset, a volume or paint (400), and for a
  // model the catalog does not have (404).
  router.post("/api/finetune/rank/score", async (req, _params, ctx) => {
    const body = objectBody(req);
    if (!body) return badRequest("expected a JSON object");
    const name = typeof body.name === "string" ? body.name.trim() : "";
    if (!name) return badRequest("name is required");
    const dataset = ctx.infer.dataset();
    if (!dataset) return badRequest(NO_DATASET);
    const volume = await sessionVolume(ctx);
    if (!volume) return json(400, { success: false, error: "No annotation volume in this session: create one (New Volume) or resume one, and paint on it, first.", needs_volume: true });
    const indices = await paintedChunkIndices(ctx.paint, volume.id);
    if (!indices.length) return badRequest(`Nothing is painted in ${volume.id} yet: paint background (1) and objects (2 and up) first.`);
    const entry = (await ctx.catalog.list()).find((e) => e.name === name) ?? (await ctx.catalog.resolve(name));
    if (!entry) return json(404, { success: false, error: `${name} is not a model the browser can run` });

    const started = performance.now();
    let held: Held;
    try {
      held = await servedForRanking(ctx, entry);
    } catch (e) {
      return json(400, { success: false, error: `${name} could not be served: ${message(e)}` });
    }
    try {
      const base = await scorePaint(ctx, dataset, volume, indices, held.served);
      let score = base.score, window: [number, number] | null = null;
      // The raw stretched to its own range, when it is narrow (portable/rank.py).
      const range = base.score && (await rawWindow(ctx, dataset, held.served.entry, base.read));
      const stretched = range && (await windowed(ctx, held.chain.input_norm, range));
      if (base.score && range && stretched) {
        const alt = await servedForRanking(ctx, entry, stretched);
        try {
          const s = (await scorePaint(ctx, dataset, volume, indices, alt.served)).score;
          if (s && s.matches >= base.score.matches + WINDOW_MARGIN) [score, window] = [s, range];
        } finally {
          await alt.release();
        }
      }
      if (!score) {
        return badRequest(hasAffinityChannels(entry)
          ? `The paint gives ${name} nothing to score: an affinity is between two painted neighbours, so paint strokes, not single voxels.`
          : `The paint gives ${name} nothing to score: only the edges between object and background strokes are painted, and those are not counted.`);
      }
      const notes = [
        ...(base.blocks > RANK_BLOCKS ? [`scored under ${RANK_BLOCKS} of the ${base.blocks} blocks of its output the paint reaches, spread over it`] : []),
        ...(base.outside ? [`${base.outside} of ${indices.length} painted chunks lie past its output, and were not scored`] : []),
      ];
      const seconds = (performance.now() - started) / 1000;
      const pct = (v: number | null) => (v === null ? "–" : `${Math.round(v * 100)}%`);
      const stretchedBy = window ? `, the raw stretched from ${window.map((v) => v.toFixed(1)).join("–")} (${pct(base.score!.matches)} as normalized)` : "";
      ctx.log(`Ranked ${name} on ${score.scored_voxels} painted voxels of ${volume.id}: matches ${pct(score.matches)} at cut-off ${score.cutoff} (objects found ${pct(score.objects_found)}, background right ${pct(score.background_right)}; ${pct(score.matches_at_zero)} at 0)${stretchedBy}, loss ${score.loss.toFixed(3)}, ${seconds.toFixed(1)} s`);
      return json(200, {
        success: true, name, ...score, seconds,
        window, ...(window ? { matches_without_window: base.score!.matches } : {}),
        ...(notes.length ? { note: notes.join("; ") } : {}),
      });
    } catch (e) {
      return json(400, { success: false, error: `${name} could not be scored: ${message(e)}` });
    } finally {
      await held.release();
    }
  });

  // POST /api/finetune/rank/use {name}: see the top of the file. Answers the
  // new volume, how much paint it took and how much was dropped where the
  // old voxels disagreed on the new grid. Refused as score is, and when the
  // session's volume is the model's already.
  router.post("/api/finetune/rank/use", async (req, _params, ctx) => {
    const body = objectBody(req);
    if (!body) return badRequest("expected a JSON object");
    const name = typeof body.name === "string" ? body.name.trim() : "";
    if (!name) return badRequest("name is required");
    const dataset = ctx.infer.dataset();
    if (!dataset) return badRequest(NO_DATASET);
    const from = await sessionVolume(ctx);
    if (!from) return json(400, { success: false, error: "No annotation volume in this session to carry paint from.", needs_volume: true });
    const entry = (await ctx.catalog.list()).find((e) => e.name === name) ?? (await ctx.catalog.resolve(name));
    if (!entry) return json(404, { success: false, error: `${name} is not a model the browser can run` });
    if (from.model_id === entry.id) return badRequest(`${from.id} is ${name}'s volume already: finetune it as it is.`);
    try {
      // The raw's window the score chose (window: [lo, hi]), made the session's
      // normalizer: the raw is the dataset's, so every model reads it so.
      const w = Array.isArray(body.window) && body.window.length === 2 ? body.window.map(Number) : null;
      const norm = w && (await windowed(ctx, ctx.session.state.chain.input_norm, [w[0], w[1]]));
      if (norm) await setChain(ctx, { ...ctx.session.state.chain, input_norm: norm });
      if (!ctx.infer.served().some((s) => s.entry.id === entry.id)) {
        const entries = [...ctx.session.state.models.filter((m) => m.id !== entry.id), entry];
        await serveModels(ctx, entries, ctx.session.state.chain);
        await ctx.session.update({ models: entries });
      }
      const served = ctx.infer.served().find((s) => s.entry.id === entry.id);
      if (!served) throw new Error(`${name} did not start`);
      const to = planVolume(dataset, served.entry, ctx.session.state.chain);
      await createVolume(ctx.paint, to);
      const carried = await carryPaint(ctx, from, to);
      // Good regions are boxes in nm, the same on any grid: they go with the paint.
      const regions = await loadGoodRegions(ctx.paint, from.id);
      if (regions.length) await saveGoodRegions(ctx.paint, to.id, regions);
      await ctx.session.update({ paintVolume: to.id });
      const layer = paintLayerName(to.id);
      ctx.viewer.removeLayer(paintLayerName(from.id));
      ctx.viewer.setLayers({ [layer]: { ...paintLayer(paintLayerUrl(to.id, pageBase()), to), name: layer } });
      ctx.viewer.selectLayer(layer, PAINT_TAB);
      ctx.log(`Carried ${from.id}'s paint to ${to.id} for ${name}: ${carried.painted} voxels at ${to.output_voxel_size.join("×")} nm from ${carried.source} at ${from.output_voxel_size.join("×")} nm, in ${carried.chunks} chunk${carried.chunks === 1 ? "" : "s"}${carried.conflicts ? `; ${carried.conflicts} left unpainted where object and background met in one voxel` : ""}`);
      return json(200, {
        success: true, name, volume_id: to.id, from_volume: from.id, layer_name: layer, window: norm ? w : null,
        painted_voxels: carried.painted, source_voxels: carried.source, conflicts: carried.conflicts, chunks: carried.chunks, good_regions: regions.length,
      });
    } catch (e) {
      return json(400, { success: false, error: `Could not make ${name} the model to finetune: ${message(e)}` });
    }
  });
}

/** `served`'s score on the paint of `volume`'s chunks `indices`, with how
 * many of its output blocks the paint reaches, which of them were read (at
 * most RANK_BLOCKS, spread over the paint: [lo, hi) on its level), and how
 * many painted chunks lie past its output. */
async function scorePaint(ctx: AppContext, dataset: DatasetInfo, volume: PaintVolume, indices: number[][], served: ServedModel): Promise<{ score: RankScore | null; blocks: number; read: { lo: number[]; hi: number[] }[]; outside: number }> {
  const entry = served.entry;
  const level = servedLevel(dataset, entry), block = outputShape(geometryOf(entry));
  const transform = targetTransformFor(entry, ctx.infer), affinity = hasAffinityChannels(entry);
  // Each painted chunk with the model's voxel under each of its painted
  // voxels (servedIndices, over the painted voxels' box, which it shrinks
  // to what the level covers), and the model's blocks those lie in.
  const placed: { chunk: PaintedChunk; origin: number[]; at: NonNullable<ReturnType<typeof servedIndices>> }[] = [];
  const blocks = new Map<string, number[]>();
  let outside = 0;
  for (const index of indices) {
    const chunk = await readPaintedChunk(ctx.paint, volume, index);
    const box = chunk && paintedBox(chunk);
    if (!chunk || !box) continue;
    const origin = index.map((i, a) => i * volume.chunk_size[a]);
    const at = servedIndices(volume, box.lo.map((v, a) => v + origin[a]), box.hi.map((v, a) => v + origin[a]), level);
    if (!at) { outside++; continue; }
    placed.push({ chunk, origin, at });
    forEachPainted(chunk, origin, at, (_i, s) => {
      const b = s.map((v, a) => Math.floor(v / block[a]));
      blocks.set(b.join(","), b);
    });
  }
  // At most RANK_BLOCKS of them, spread over the paint, each read whole.
  const keys = [...blocks.keys()].sort((p, q) => compareIndex(blocks.get(p)!, blocks.get(q)!));
  const read = new Map<string, { lo: number[]; hi: number[]; shape: number[]; data: Float32Array }>();
  for (const key of spread(keys, RANK_BLOCKS)) {
    const lo = blocks.get(key)!.map((b, a) => b * block[a]), hi = lo.map((l, a) => Math.min(l + block[a], level.shape[a]));
    const out = await ctx.infer.readOutput(served.view, lo, hi);
    read.set(key, { lo, hi, shape: out.shape, data: logitsOf(out, served.info.output_class) });
  }
  // Each placed chunk added to the model's tallies by the Python (rank.tally,
  // with paint_edges for the voxels a binary model's score skips): its
  // labels, its targets and the model's logits on its voxels, NaN where
  // none was read. The tallies travel with each call: the workers take turns.
  const channels = entry.manifest.channels, C = channels.length;
  let tallies: Tallies | null = null;
  for (const { chunk, origin, at } of placed) {
    const n = chunk.labels.length, logits = new Float32Array(C * n).fill(NaN);
    forEachPainted(chunk, origin, at, (i, s) => {
      const r = read.get(s.map((v, a) => Math.floor(v / block[a])).join(","));
      if (!r) return;
      const [, bz, by, bx] = r.shape, [sz, sy, sx] = s.map((v, a) => v - r.lo[a]);
      for (let c = 0; c < C; c++) logits[c * n + i] = r.data[((c * bz + sz) * by + sy) * bx + sx];
    });
    const { target, mask } = await transform(chunk);
    tallies = await ctx.infer.python<Tallies>(`${PORTABLE}:rank_tally`,
      { tallies, shape: chunk.shape, channels: C, labels_dtype: chunk.labels instanceof Uint16Array ? "uint16" : "uint8", affinity },
      [bytesOf(chunk.labels), bytesOf(target), bytesOf(mask), bytesOf(logits)]);
  }
  const score = tallies ? await ctx.infer.python<RankScore | null>(`${PORTABLE}:rank_score`, { tallies, channels, affinity }) : null;
  return { score, blocks: keys.length, read: [...read.values()].map(({ lo, hi }) => ({ lo, hi })), outside };
}

/** A model's output block as logits, whatever its activation, as the
 * server's ranking reads it (rank_routes._score): the output as a
 * probability by the served model's output class (label-it's probability,
 * so a sigmoid model's 0.9 and a logit model's 2.2 are one and the same),
 * clipped to [1e-6, 1 - 1e-6], then its logit. A logit model's output comes
 * back as it was, within float32; a probability's is no longer read as if
 * 0.1 said object. */
function logitsOf(out: { data: Float32Array; shape: number[] }, outputClass: string | null | undefined): Float32Array {
  const p = probability(out, outputClass, null).data;
  for (let i = 0; i < p.length; i++) {
    const q = Math.min(Math.max(p[i], 1e-6), 1 - 1e-6);
    p[i] = Math.log(q / (1 - q));
  }
  return p;
}

/** The window of the raw under the output blocks `blocks` of `entry`'s
 * level (scorePaint's read): the 1st and 99th percentiles of the raw it reads
 * there (every few voxels, at most RAW_SAMPLE), or null when nothing was
 * read or the raw has more than one channel. */
async function rawWindow(ctx: AppContext, dataset: DatasetInfo, entry: ModelEntry, blocks: { lo: number[]; hi: number[] }[]): Promise<[number, number] | null> {
  if (dataset.channels !== 1) return null;
  const g = geometryOf(entry), level = chooseRawLevel(dataset.levels, g.input_voxel_size), shape = dataset.levels[level].shape;
  const r = ratio(g), Typed = TYPED[dataset.dtype] ?? Float32Array;
  const parts: ArrayLike<number | bigint>[] = [];
  for (const { lo, hi } of blocks) {
    const box = inputBox(lo, hi, r, r.map(() => 0));
    const readLo = box.lo.map((v) => Math.max(v, 0)), readHi = box.hi.map((v, a) => Math.min(v, shape[a]));
    if (readHi.some((h, a) => h <= readLo[a])) continue;
    parts.push(new Typed(await ctx.infer.readRaw(level, readLo, readHi)));
  }
  const total = parts.reduce((n, p) => n + p.length, 0);
  if (!total) return null;
  const every = Math.max(1, Math.floor(total / RAW_SAMPLE)), values: number[] = [];
  for (const p of parts) for (let i = 0; i < p.length; i += every) values.push(Number(p[i]));
  const sorted = Float64Array.from(values).sort();
  return [percentile(sorted, 1), percentile(sorted, 99)];
}

/** `visit(i, served)` for each painted voxel of `chunk` the model's level
 * covers: `i` its index in the chunk, `served` the model's voxel under it,
 * from `at` (servedIndices over a box of the volume, `origin` the chunk's
 * first voxel). */
function forEachPainted(chunk: PaintedChunk, origin: number[], at: NonNullable<ReturnType<typeof servedIndices>>, visit: (i: number, served: number[]) => void): void {
  const [, h, w] = chunk.shape;
  for (let z = at.lo[0]; z < at.hi[0]; z++) for (let y = at.lo[1]; y < at.hi[1]; y++) for (let x = at.lo[2]; x < at.hi[2]; x++) {
    const i = ((z - origin[0]) * h + y - origin[1]) * w + x - origin[2];
    if (chunk.labels[i]) visit(i, [at.indices[0][z - at.lo[0]], at.indices[1][y - at.lo[1]], at.indices[2][x - at.lo[2]]]);
  }
}

function compareIndex(p: number[], q: number[]): number {
  return p[0] - q[0] || p[1] - q[1] || p[2] - q[2];
}

/** The portable Python's entry points (cellmap_flow/portable/calls.py), by module path. */
const PORTABLE = "cellmap_flow.portable.calls";

/** A model's tallies of the paint so far, as the Python answers them and takes them back; nothing here reads them. */
type Tallies = Record<string, unknown>[];

/** A model's score on the paint, as rank_score answers it (portable/rank.py): the
 * balanced accuracy at the model's own cut-off, its parts, the score at 0, the loss,
 * the voxels counted and a binary model's channel (null for an affinity model's). */
interface RankScore {
  matches: number;
  objects_found: number | null;
  background_right: number | null;
  cutoff: number;
  matches_at_zero: number;
  loss: number;
  scored_voxels: number;
  object_voxels: number;
  background_voxels: number;
  channel: string | null;
}

/** A typed array's own bytes, copied: a call's arrays are handed to the worker. */
function bytesOf(a: ArrayBufferView): ArrayBuffer {
  return a.buffer.slice(a.byteOffset, a.byteOffset + a.byteLength) as ArrayBuffer;
}

/** `norm` with its MinMaxNormalizer set to the raw's window, or null (rank.windowed:
 * no such normalizer, no window, or one not narrower than NARROW_WINDOW of its range). */
function windowed(ctx: AppContext, norm: readonly Step[], [lo, hi]: readonly [number, number]): Promise<Step[] | null> {
  return ctx.infer.python<Step[] | null>(`${PORTABLE}:windowed`, { norm, lo, hi });
}

/** `count` of `items`, spread evenly over them in their order; all of them when there
 * are no more (rank.spread, for which blocks to read: a pick, so no call for it). */
function spread<T>(items: readonly T[], count: number): T[] {
  if (items.length <= count) return [...items];
  return Array.from({ length: count }, (_, j) => items[Math.floor(((j + 0.5) * items.length) / count)]);
}

/** The box of `chunk`'s painted voxels, in its own voxels; null when none is painted. */
function paintedBox(chunk: PaintedChunk): { lo: number[]; hi: number[] } | null {
  const [, h, w] = chunk.shape, lo = [Infinity, Infinity, Infinity], hi = [-1, -1, -1];
  chunk.labels.forEach((label, i) => {
    if (!label) return;
    const at = [Math.floor(i / (h * w)), Math.floor(i / w) % h, i % w];
    for (let a = 0; a < 3; a++) { lo[a] = Math.min(lo[a], at[a]); hi[a] = Math.max(hi[a], at[a] + 1); }
  });
  return hi[0] < 0 ? null : { lo, hi };
}

/** What carryPaint did: the new volume's voxels painted, the old volume's painted
 * voxels they came from, the new voxels left unpainted because their old voxels
 * disagreed, and the new volume's chunks written. */
interface Carried { painted: number; source: number; conflicts: number; chunks: number }

/** The paint of `from` copied into `to`, an empty volume over the same dataset on
 * another model's grid (createVolume'd): cellmap_flow.portable.carry, in the engine's
 * Python, with the paint store's reads and writes around it here. The old volume's
 * painted chunks go whole, both grids placed by their volume's dataset_offset_nm
 * (voxel 0's centre, fill.ts volumeCornerNm), and the new chunks that got paint come
 * back whole, to be written as they are. */
async function carryPaint(ctx: AppContext, from: PaintVolume, to: PaintVolume): Promise<Carried> {
  const grid = (v: PaintVolume) => ({ corner_nm: volumeCornerNm(v.dataset_offset_nm, v.output_voxel_size), voxel_nm: v.output_voxel_size, shape: v.dataset_shape_voxels, chunks: v.chunk_size });
  const indices: number[][] = [], labels: ArrayBuffer[] = [];
  for (const index of await paintedChunkIndices(ctx.paint, from.id)) {
    const chunk = await readPaintedChunk(ctx.paint, from, index);
    if (chunk) { indices.push(index); labels.push(bytesOf(chunk.labels)); }
  }
  const { value, arrays } = await ctx.infer.python<{ value: Carried & { indices: number[][] }; arrays: ArrayBuffer[] }>(
    `${PORTABLE}:carry_paint`, { old: grid(from), new: grid(to), old_dtype: from.dtype, new_dtype: to.dtype, indices }, labels);
  await Promise.all(value.indices.map((index, k) => ctx.paint.put(to.id, CHUNK_DIR + chunkKey(index), new Uint8Array(arrays[k]))));
  const { painted, source, conflicts, chunks } = value;
  return { painted, source, conflicts, chunks };
}

async function sessionVolume(ctx: AppContext): Promise<PaintVolume | null> {
  const id = ctx.session.state.paintVolume;
  return id ? readVolume(ctx.paint, id) : null;
}

/** A model served for a score, the chain it is read through, and its release. */
interface Held { served: ServedModel; chain: Chain; release: () => Promise<void> }

/** `entry` served for a score: as it is served already (through the
 * session's chain), or served now with `inputNorm` or else its manifest's
 * normalizers and no postprocessing (the model stage's own output is what
 * is read), and released when the score is done. Served
 * now, it is its smallest block at the precision this GPU takes
 * (Catalog.chooseVariant without a buffer limit): a 3D model's largest
 * block can hold most of a GPU's memory, beside the models already served,
 * and a GPU run out of memory loses its device and every model with it. */
async function servedForRanking(ctx: AppContext, entry: ModelEntry, inputNorm?: Step[]): Promise<Held> {
  const current = !inputNorm && ctx.infer.served().find((s) => s.entry.id === entry.id);
  if (current) return { served: current, chain: ctx.session.state.chain, release: async () => {} };
  const limits = { ...(await ctx.infer.gpuLimits()), maxBufferSize: null };
  const tuned = entryFor(entry.folder, entry.manifest, ctx.catalog.chooseVariant(entry.manifest, limits), entry.repo);
  const chain: Chain = { input_norm: inputNorm ?? entry.manifest.input_norm, postprocess: [] };
  const served = await ctx.infer.serve(tuned, chain, (await blobFor(chain)).blob, () => ctx.catalog.modelBytes(tuned));
  return { served, chain, release: () => served.dispose() };
}
