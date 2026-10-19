// Which model to start from: the Finetune tab ranks the models the browser
// can run by how well each one's prediction already matches the session
// volume's paint (finetune/rank.ts says how a model is scored), so the one
// to finetune is the one with the least to learn.
//
// GET /api/finetune/rank/candidates lists them: the catalog's models,
// finetuned ones among them, with the session's volume and the model it is
// for. POST /api/finetune/rank/score {name} scores one, as the tab asks for
// them one at a time and shows each as it comes. A model already served is
// read as it is served, its chunks from the engine's caches where the viewer
// has had them computed; any other is served for the score alone, with its
// manifest's normalizers and no layer, and released after, so a ranking
// leaves the Models tab and the GPU as they were (its download stays in the
// catalog's byte cache, and Show serves it at once). POST
// /api/finetune/rank/use {name} makes the one picked the model to finetune:
// served with its layer (as the Models tab's Add serves it), a volume made
// for it with the session volume's paint carried onto its grid
// (finetune/carry.ts) and its good regions, and that volume the session's,
// its paint layer in the old one's place; the old volume stays in the paint
// store.
import type { AppContext, Chain, ModelEntry, PaintVolume, ServedModel } from "../../contracts";
import { FINETUNED_REPO } from "../../contracts";
import { carryPaint, paintedBox } from "../../finetune/carry";
import { loadGoodRegions, saveGoodRegions } from "../../finetune/good-regions";
import { readsOneSlice } from "../../finetune/modes";
import type { PaintedChunk } from "../../finetune/types";
import { emptyTallies, paintEdges, RANK_BLOCKS, rankScore, spread, tallyChunk } from "../../finetune/rank";
import { targetTransformFor } from "../../finetune/targets";
import { createVolume, hasAffinityChannels, paintedChunkIndices, planVolume, readPaintedChunk, readVolume } from "../../finetune/volume";
import { geometryOf, outputShape } from "../../infer/geometry";
import { servedIndices, servedLevel } from "../../label-it/prediction";
import { entryFor } from "../../models/manifest";
import { PAINT_TAB, paintLayer, paintLayerName, paintLayerUrl } from "../../viewer/layers";
import type { Router } from "../router";
import { badRequest, json, message, objectBody } from "../router";
import { blobFor, pageBase, serveModels } from "../serving";

const NO_DATASET = "No dataset is open. Enter its URL in the viewer panel first; the volume is painted on it.";

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
  // (finetune/rank.ts RankScore, in snake case), how long it took, and a
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
    let held: { served: ServedModel; release: () => Promise<void> };
    try {
      held = await servedForRanking(ctx, entry);
    } catch (e) {
      return json(400, { success: false, error: `${name} could not be served: ${message(e)}` });
    }
    try {
      const level = servedLevel(dataset, held.served.entry), block = outputShape(geometryOf(held.served.entry));
      const transform = targetTransformFor(entry), affinity = hasAffinityChannels(entry);
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
        forEachPainted(chunk, origin, at, (_i, served) => {
          const b = served.map((s, a) => Math.floor(s / block[a]));
          blocks.set(b.join(","), b);
        });
      }
      // At most RANK_BLOCKS of them, spread over the paint, each read whole.
      const keys = [...blocks.keys()].sort((p, q) => compareIndex(blocks.get(p)!, blocks.get(q)!));
      const read = new Map<string, { lo: number[]; shape: number[]; data: Float32Array }>();
      for (const key of spread(keys, RANK_BLOCKS)) {
        const lo = blocks.get(key)!.map((b, a) => b * block[a]), hi = lo.map((l, a) => Math.min(l + block[a], level.shape[a]));
        const out = await ctx.infer.readOutput(held.served.view, lo, hi);
        read.set(key, { lo, shape: out.shape, data: out.data });
      }
      const tallies = emptyTallies(entry.manifest.channels.length);
      for (const { chunk, origin, at } of placed) {
        const logits = new Map<number, (c: number) => number>();
        forEachPainted(chunk, origin, at, (i, served) => {
          const r = read.get(served.map((s, a) => Math.floor(s / block[a])).join(","));
          if (!r) return;
          const [, bz, by, bx] = r.shape, [sz, sy, sx] = served.map((s, a) => s - r.lo[a]);
          logits.set(i, (c) => r.data[((c * bz + sz) * by + sy) * bx + sx]);
        });
        tallyChunk(tallies, chunk, transform(chunk), (c, i) => logits.get(i)?.(c) ?? null, affinity ? null : paintEdges(chunk));
      }
      const score = rankScore(tallies, entry.manifest.channels, affinity);
      if (!score) {
        return badRequest(affinity
          ? `The paint gives ${name} nothing to score: an affinity is between two painted neighbours, so paint strokes, not single voxels.`
          : `The paint gives ${name} nothing to score: only the edges between object and background strokes are painted, and those are not counted.`);
      }
      const notes = [
        ...(keys.length > RANK_BLOCKS ? [`scored under ${RANK_BLOCKS} of the ${keys.length} blocks of its output the paint reaches, spread over it`] : []),
        ...(outside ? [`${outside} of ${indices.length} painted chunks lie past its output, and were not scored`] : []),
      ];
      const seconds = (performance.now() - started) / 1000;
      const pct = (v: number | null) => (v === null ? "–" : `${Math.round(v * 100)}%`);
      ctx.log(`Ranked ${name} on ${score.scored} painted voxels of ${volume.id}: matches ${pct(score.matches)} (objects found ${pct(score.objectsFound)}, background right ${pct(score.backgroundRight)}), loss ${score.loss.toFixed(3)}, ${seconds.toFixed(1)} s`);
      return json(200, {
        success: true, name,
        matches: score.matches, objects_found: score.objectsFound, background_right: score.backgroundRight, loss: score.loss,
        scored_voxels: score.scored, object_voxels: score.objects, background_voxels: score.background, channel: score.channel, seconds,
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
      if (!ctx.infer.served().some((s) => s.entry.id === entry.id)) {
        const entries = [...ctx.session.state.models.filter((m) => m.id !== entry.id), entry];
        await serveModels(ctx, entries, ctx.session.state.chain);
        await ctx.session.update({ models: entries });
      }
      const served = ctx.infer.served().find((s) => s.entry.id === entry.id);
      if (!served) throw new Error(`${name} did not start`);
      const to = planVolume(dataset, served.entry, ctx.session.state.chain);
      await createVolume(ctx.paint, to);
      const carried = await carryPaint(ctx.paint, from, to);
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
        success: true, name, volume_id: to.id, from_volume: from.id, layer_name: layer,
        painted_voxels: carried.painted, source_voxels: carried.source, conflicts: carried.conflicts, chunks: carried.chunks, good_regions: regions.length,
      });
    } catch (e) {
      return json(400, { success: false, error: `Could not make ${name} the model to finetune: ${message(e)}` });
    }
  });
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

async function sessionVolume(ctx: AppContext): Promise<PaintVolume | null> {
  const id = ctx.session.state.paintVolume;
  return id ? readVolume(ctx.paint, id) : null;
}

/** `entry` served for a score: as it is served already, or served now with
 * its manifest's normalizers and no postprocessing (the model stage's own
 * output is what is read), and released when the score is done. Served
 * now, it is its smallest block at the precision this GPU takes
 * (Catalog.chooseVariant without a buffer limit): a 3D model's largest
 * block can hold most of a GPU's memory, beside the models already served,
 * and a GPU run out of memory loses its device and every model with it. */
async function servedForRanking(ctx: AppContext, entry: ModelEntry): Promise<{ served: ServedModel; release: () => Promise<void> }> {
  const current = ctx.infer.served().find((s) => s.entry.id === entry.id);
  if (current) return { served: current, release: async () => {} };
  const limits = { ...(await ctx.infer.gpuLimits()), maxBufferSize: null };
  const tuned = entryFor(entry.folder, entry.manifest, ctx.catalog.chooseVariant(entry.manifest, limits), entry.repo);
  const chain: Chain = { input_norm: entry.manifest.input_norm, postprocess: [] };
  const served = await ctx.infer.serve(tuned, chain, (await blobFor(chain)).blob, () => ctx.catalog.modelBytes(tuned));
  return { served, release: () => served.dispose() };
}
