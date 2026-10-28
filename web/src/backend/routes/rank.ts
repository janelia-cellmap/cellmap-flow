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
// them one at a time and shows each as it comes. A model is scored through
// each input normalization portable/rank.py's norm_tries gives it, its own
// (its manifest's declaration, or for a model that declares none what the
// server's model advice assumes of it: the raw's dtype range to [-1, 1],
// assumedInputNorm) and the session's when that is another, and the better
// score is kept (best_try): read through none, or another model's, a
// model's score is noise (fly_organelles fed raw 0 to 255 matched 51% in the
// server's ranking, which read every model through the session's input
// normalization). Through the session's, a model already served is
// read as it is served, its chunks from the engine's caches where the viewer
// has had them computed; any other read is served for the score alone, with
// those normalizers and no layer, and released after, so a ranking leaves
// the Models tab and the GPU as they were (its download stays in the
// catalog's byte cache, and Show serves it at once). Where the raw under the
// paint is narrow, a model is scored a second time through the kept
// normalization stretched to the raw's window, served for that alone
// (portable/rank.py says when that is kept). The answer is the ranking's
// row, as the Python builds it (score_row: the score kept, its threshold,
// the raw's range it read and the normalizer steps it was read through,
// input_norm, with where they came from; and why those: what each channel
// of a binary model matched, channel_matches, and what the other
// normalizations tried matched, input_norm_others), with the
// postprocessing its layer needs to show a good prediction
// (rank.needed_postprocess: the sigmoid its output needs, a second one
// taken away, and a binary model's threshold at the row's), or null.
// POST /api/finetune/rank/use {name, select_channel?, input_norm?,
// postprocess?} makes the one picked the model to finetune, the row's
// channel first made the one its layer shows (select_channel, the row's
// channel_index, as POST /api/finetune/channel picks one; an affinity
// model's row has none), so its layer opens on that channel rather than on
// Neuroglancer's middle one, and the row's steps and postprocessing made the
// session chain's (each when given and not the session's already, in one
// chain set), so it is served through what it was ranked through and shows
// as it should: served with its layer (as the Models tab's Add serves it), a
// volume made for it with the session volume's paint carried onto its grid
// (cellmap_flow/portable/carry.py, the same way) and its good regions, and
// that volume the session's, its paint layer in the old one's place. The old
// volume is retired: kept in the paint store, never deleted, but marked
// superseded (volume.ts supersedeVolume), so the session has one volume, the
// one it trains on, and the old one is neither listed nor resumed again. The
// carry is finetune.ts carryOnto, which training's submit carries through
// too, for a model whose grid is not the volume's.
// POST /api/finetune/rank/input-norm {input_norm, postprocess?} makes them
// the session chain's alone, with no model served and no volume made: Fit to
// paint's, for the model already seeding the paint.
//
// The list offers the models the Models tab does, found the same way: each
// candidate is under the tab's section it would be listed in (GROUPS:
// Running, cellmap, Added, Finetuned), the tab's Refresh (POST
// /api/huggingface-models/refresh) reads the catalog again and the next
// listing is of what it read, and POST /api/finetune/rank/add {ref} adds a
// model by hand, by any ref the tab's Add a model takes, to every listing
// after, without serving it.
import { TYPED } from "../../../chunkmirage/web/src/asread";
import { inputBox } from "../../../chunkmirage/web/src/stages";
import { percentile } from "../../ai-annotate/mask";
import type { AppContext, Chain, DatasetInfo, ModelEntry, PaintVolume, ServedModel, Step } from "../../contracts";
import { FINETUNED_REPO } from "../../contracts";
import { readsOneSlice } from "../../finetune/modes";
import type { PaintedChunk } from "../../finetune/types";
import { targetTransformFor } from "../../finetune/target-call";
import { hasAffinityChannels, paintedChunkIndices, planVolume, readPaintedChunk, readVolume } from "../../finetune/volume";
import { RANK_BLOCKS } from "../../generated/cellmap-flow";
import { chooseRawLevel, geometryOf, outputShape, ratio } from "../../infer/geometry";
import { place, type Placement, servedLevel } from "../../label-it/prediction";
import { probability } from "../../label-it/segment";
import { HF_REPO } from "../../models";
import { entryFor } from "../../models/manifest";
import { dtypeRange } from "../../viewer/display";
import type { Router } from "../router";
import { badRequest, json, message, objectBody } from "../router";
import { blobFor, declaredInputNorm, PORTABLE, serveModels, setChain } from "../serving";
import { bytesOf, carryOnto, notAChannel, pickShownChannel, readSelectChannel } from "./finetune";
import { resolveRef } from "./models";
import { readChain } from "./pipeline";

const NO_DATASET = "No dataset is open. Enter its URL in the viewer panel first; the volume is painted on it.";

/** The most raw voxels the window of the raw under the paint is taken from (every few of them). */
const RAW_SAMPLE = 1_000_000;

export function rankRoutes(router: Router): void {
  router.get("/api/finetune/rank/candidates", async (_req, _params, ctx) => {
    const volume = await sessionVolume(ctx);
    const candidates = (await ctx.catalog.list()).map(listingOf(ctx, volume));
    return json(200, {
      success: true,
      volume: volume ? { id: volume.id, model_name: volume.model_name, painted_chunks: (await paintedChunkIndices(ctx.paint, volume.id)).length } : null,
      // In the groups' order, each group's in the catalog's (a stable sort).
      candidates: candidates.sort((a, b) => GROUPS.indexOf(a.group) - GROUPS.indexOf(b.group)),
    });
  });

  // POST /api/finetune/rank/add {ref}: a model added to the list by hand,
  // by any ref the Models tab's Add a model takes (models.ts resolveRef, the
  // resolve POST /api/models/resolve answers from, so a ref it refuses is
  // refused here the same way), remembered in the session (rankAdded) and
  // answered as every listing after lists it: under Added, ticked (under
  // Running while it is served). Nothing is served: Rank scores it as it
  // scores any candidate, found by its name in the catalog, whose entry it
  // is.
  router.post("/api/finetune/rank/add", async (req, _params, ctx) => {
    const resolved = await resolveRef(req, ctx);
    if ("refused" in resolved) return resolved.refused;
    const { entry } = resolved;
    const { rankAdded } = ctx.session.state;
    if (!rankAdded.includes(entry.id)) await ctx.session.update({ rankAdded: [...rankAdded, entry.id] });
    return json(200, { success: true, candidate: listingOf(ctx, await sessionVolume(ctx))(entry) });
  });

  // POST /api/finetune/rank/score {name}: the model's row of the ranking
  // (score_row's, portable/rank.py), the postprocessing its layer needs
  // (postprocess: needed_postprocess's steps, or null), how long it took,
  // and a note when some painted chunks lie where the model's level does
  // not reach. Refused without a dataset, a volume or paint (400), and for
  // a model the catalog does not have (404).
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
    const pct = (v: number | null) => (v === null ? "–" : `${Math.round(v * 100)}%`);
    try {
      // The paint scored through each input normalization the model is
      // tried with (rank.norm_tries: its own first, the session's when that
      // is another; an assumed one after the session's), the model served
      // for each score alone and released after, but read as it is served
      // through the session's steps when it is served already (sessionsTry
      // says which try those are); and the best of the scores kept
      // (rank.best_try, its own among equals).
      const session = ctx.session.state.chain.input_norm;
      const declared = declaredInputNorm(entry);
      const own = declared.length ? { own: declared, own_from: "declared" } : { own: assumedInputNorm(dataset.dtype), own_from: "assumed" };
      const tries = await ctx.infer.python<NormTry[]>(`${PORTABLE}:norm_tries`, { ...own, session });
      const asServed = sessionsTry(tries, session);
      const stepsOf = (i: number) => (i === asServed ? session : tries[i].input_norm);
      const reads: (PaintScore & { served: ServedModel })[] = [];
      for (let i = 0; i < tries.length; i++) {
        let held: Held;
        try {
          held = await servedForRanking(ctx, entry, stepsOf(i), i === asServed);
        } catch (e) {
          return json(400, { success: false, error: `${name} could not be served: ${message(e)}` });
        }
        try {
          reads.push({ ...(await scorePaint(ctx, dataset, volume, indices, held.served)), served: held.served });
        } finally {
          await held.release();
        }
      }
      const best = await ctx.infer.python<number | null>(`${PORTABLE}:best_try`, { scores: reads.map((r) => r.score) });
      const kept = best === null ? null : reads[best];
      if (best === null || !kept?.score) {
        return badRequest(hasAffinityChannels(entry)
          ? `The paint gives ${name} nothing to score: an affinity is between two painted neighbours, so paint strokes, not single voxels.`
          : `The paint gives ${name} nothing to score: only the edges between object and background strokes are painted, and those are not counted.`);
      }
      // The raw stretched to its own range, when it is narrow, through the
      // steps kept, and the row from the two scores (portable/rank.py).
      const norm = stepsOf(best);
      const range = await rawWindow(ctx, dataset, kept.served.entry, kept.read);
      const stretched = range && (await windowed(ctx, norm, range));
      let alt: RankScore | null = null;
      if (range && stretched) {
        const again = await servedForRanking(ctx, entry, stretched, false);
        try {
          alt = (await scorePaint(ctx, dataset, volume, indices, again.served)).score;
        } finally {
          await again.release();
        }
      }
      // What the other tries matched (those that scored), beside it in the row.
      const others = reads.flatMap((r, i) => (i === best || !r.score ? [] : [{ from: tries[i].from, matches: r.score.matches }]));
      const score = (await rankRow(ctx, kept.score, alt, norm, range, tries[best].from, others))!;
      // What its layer needs to show a good prediction as the session's chain
      // stands (rank.needed_postprocess): the sigmoid its output needs and,
      // for a binary model, a threshold at the row's, fitted to the paint,
      // which Use this model, Fit to Paint and Use these settings set beside
      // the input normalization. An affinity model gets the sigmoid alone:
      // the watershed that makes its objects (AffinityPostprocessor, the
      // model advice's optional chain) joins neighbours by how far their
      // affinity is above its bias, so it needs graded affinities, not the
      // 0s and 1s of a threshold, which it reads as 0-255 and divides by
      // 255, leaving no edge at the bias and no object. Its fitted threshold
      // is still the row's, Seed from Prediction's.
      const postprocess = await ctx.infer.python<Step[] | null>(`${PORTABLE}:needed_postprocess`, {
        output_class: kept.served.info.output_class ?? null, postprocess: ctx.session.state.chain.postprocess,
        threshold: hasAffinityChannels(entry) ? null : score.threshold,
      });
      // Remembered as what the paint fitted this model to (the session's
      // rankFits), so the model advice keeps to these steps once they are
      // the chain's, and the Processing tab says where the postprocessing
      // came from.
      await ctx.session.update({ rankFits: { ...ctx.session.state.rankFits, [entry.name]: { input_norm: score.input_norm, postprocess } } });
      const notes = [
        ...(kept.blocks > RANK_BLOCKS ? [`scored under ${RANK_BLOCKS} of the ${kept.blocks} blocks of its output the paint reaches, spread over it`] : []),
        ...(kept.outside ? [`${kept.outside} of ${indices.length} painted chunks lie past its output, and were not scored`] : []),
      ];
      const seconds = (performance.now() - started) / 1000;
      const stretchedBy = score.window ? `, the raw stretched from ${score.window.map((v) => v.toFixed(1)).join("–")} (${pct(kept.score.matches)} as normalized)` : "";
      ctx.log(`Ranked ${name} on ${score.scored_voxels} painted voxels of ${volume.id}: matches ${pct(score.matches)} at cut-off ${score.cutoff} (objects found ${pct(score.objects_found)}, background right ${pct(score.background_right)}; ${pct(score.matches_at_zero)} at 0) through ${through(score.input_norm_from)}, ${stepsText(score.input_norm)}${stretchedBy}, loss ${score.loss.toFixed(3)}, ${seconds.toFixed(1)} s`);
      return json(200, {
        success: true, name, ...score, postprocess, seconds,
        ...(notes.length ? { note: notes.join("; ") } : {}),
      });
    } catch (e) {
      return json(400, { success: false, error: `${name} could not be scored: ${message(e)}` });
    }
  });

  // POST /api/finetune/rank/use {name, entry?, select_channel?,
  // input_norm?, postprocess?}: see the top of the file; `select_channel`,
  // `input_norm` and `postprocess` are the row's (`entry`, the server's model
  // config for a model it does not list, is the catalog's here). Answers the
  // new volume, the one it took the place of (from_volume, now superseded),
  // how much paint it took and how much was dropped where the old voxels
  // disagreed on the new grid, and the input normalization and
  // postprocessing it set (each null when the session's was those steps
  // already). Refused as score is, when the session's volume is the model's
  // already, for a select_channel that is not one of the model's channels,
  // and for steps that are not a list of steps the browser runs (checked as
  // PUT /api/pipeline checks them), each of which changes nothing.
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
    const pick = readSelectChannel(body.select_channel);
    if ("error" in pick) return badRequest(pick.error);
    const past = pick.channel === null ? null : notAChannel(entry, pick.channel);
    if (past) return badRequest(past);
    const read = readParts(body);
    if ("error" in read) return badRequest(read.error);
    try {
      // The row's channel first, so the layer the model is served with opens
      // on it (and a layer it has already moves to it), then the steps it was
      // ranked through and the postprocessing its layer needs, so the model
      // is served through them (and the models served already redrawn).
      if (pick.channel !== null) await pickShownChannel(ctx, entry.name, pick.channel);
      const set = await setParts(ctx, read);
      if (set.input_norm) ctx.log(`Input normalization set to the one ${name} was ranked through: ${stepsText(set.input_norm)}`);
      if (set.postprocess) ctx.log(`Postprocessing set to the one ${name}'s layer needs: ${stepsText(set.postprocess)}`);
      if (!ctx.infer.served().some((s) => s.entry.id === entry.id)) {
        const entries = [...ctx.session.state.models.filter((m) => m.id !== entry.id), entry];
        await serveModels(ctx, entries, ctx.session.state.chain);
        await ctx.session.update({ models: entries });
      }
      const served = ctx.infer.served().find((s) => s.entry.id === entry.id);
      if (!served) throw new Error(`${name} did not start`);
      // The volume made for it, the paint and good regions carried onto it,
      // and the old one superseded (finetune.ts carryOnto, which training's
      // submit carries through too, for a model on another grid).
      const to = await planVolume(ctx.infer, dataset, served.entry, ctx.session.state.chain);
      const carried = await carryOnto(ctx, from, to, name);
      return json(200, {
        success: true, name, volume_id: to.id, from_volume: from.id, layer_name: carried.layer, input_norm: set.input_norm, postprocess: set.postprocess,
        painted_voxels: carried.painted, source_voxels: carried.source, conflicts: carried.conflicts, chunks: carried.chunks, good_regions: carried.goodRegions,
      });
    } catch (e) {
      return json(400, { success: false, error: `Could not make ${name} the model to finetune: ${message(e)}` });
    }
  });

  // POST /api/finetune/rank/input-norm {input_norm, postprocess?}: Fit to
  // paint's, the steps a score was read through and the postprocessing it
  // gave (the score answer's input_norm and postprocess) made the session
  // chain's, as Use this model makes them, without serving a model or
  // making a volume. Answers the chain's two parts as they now are; refused
  // (400) for a body with no input_norm, or with steps that are not a list
  // of steps the browser runs (checked as PUT /api/pipeline checks them),
  // which changes nothing.
  router.post("/api/finetune/rank/input-norm", async (req, _params, ctx) => {
    const body = objectBody(req);
    if (!body) return badRequest("expected a JSON object");
    const read = readParts(body);
    if ("error" in read) return badRequest(read.error);
    if (!read.input_norm) return badRequest("input_norm: expected a list of steps");
    try {
      const set = await setParts(ctx, read);
      if (set.input_norm) ctx.log(`Input normalization set to fit the paint: ${stepsText(set.input_norm)}`);
      if (set.postprocess) ctx.log(`Postprocessing set to the one the model's layer needs: ${stepsText(set.postprocess)}`);
      const { chain } = ctx.session.state;
      return json(200, { success: true, input_norm: chain.input_norm, postprocess: chain.postprocess });
    } catch (e) {
      return json(400, { success: false, error: `Could not set the input normalization and postprocessing: ${message(e)}` });
    }
  });
}

/** A chain's two parts, each null where it is not given (or not changed). */
interface Parts { input_norm: Step[] | null; postprocess: Step[] | null }

/** The chain's parts a request gives, a row's `input_norm` and
 * `postprocess`, each null where the body leaves it out; or the reason one
 * is not a chain part, each checked as PUT /api/pipeline checks a chain's
 * (pipeline.ts readChain: a list, each step one the browser runs). */
function readParts(body: Record<string, unknown>): Parts | { error: string } {
  const given = (v: unknown) => v !== undefined && v !== null;
  const read = readChain({ input_norm: given(body.input_norm) ? body.input_norm : [], postprocess: given(body.postprocess) ? body.postprocess : [] });
  if ("error" in read) return read;
  return { input_norm: given(body.input_norm) ? read.chain.input_norm : null, postprocess: given(body.postprocess) ? read.chain.postprocess : null };
}

/** `parts` made the session chain's, each in place of the session's own
 * when given and other than it, in one chain set, and its layers redrawn
 * through it (serving.ts setChain, as PUT /api/pipeline sets it): the raw
 * is the dataset's, so every model reads it so. Answers the parts set, each
 * null where nothing changed: the session's steps already, as a row read
 * through them gives them back, change nothing. */
async function setParts(ctx: AppContext, parts: Parts): Promise<Parts> {
  const { chain } = ctx.session.state;
  const changed = (steps: Step[] | null, now: Step[]) => (steps && JSON.stringify(steps) !== JSON.stringify(now) ? steps : null);
  const set = { input_norm: changed(parts.input_norm, chain.input_norm), postprocess: changed(parts.postprocess, chain.postprocess) };
  if (set.input_norm || set.postprocess) await setChain(ctx, { input_norm: set.input_norm ?? chain.input_norm, postprocess: set.postprocess ?? chain.postprocess });
  return set;
}

/** What the server's model advice assumes of a model nothing is known of
 * (serving/probe.py suggest_input_norm, its low-confidence answer, which
 * the server's ranking tries as "assumed"): the raw scaled from its dtype's
 * range (dtypeRange: an integer dtype's own, 0 to 1 for a float) to [0, 1],
 * then to [-1, 1], which everything in this repo trains on. */
function assumedInputNorm(dtype: string): Step[] {
  const [lo, hi] = dtypeRange(dtype);
  return [{ name: "MinMaxNormalizer", min_value: lo, max_value: hi, invert: false }, { name: "LambdaNormalizer", expression: "x*2-1" }];
}

/** Chain steps as the log shows them: each by name with its parameters; "none" for none. */
function stepsText(steps: readonly Step[]): string {
  return steps.map(({ name, ...p }) => `${name}(${Object.entries(p).map(([k, v]) => `${k}=${v}`).join(", ")})`).join(" → ") || "none";
}

/** Which of `tries` (rank.norm_tries's, the session's input normalization
 * being `session`) reads through the session's steps, so a model served
 * already (through the session's chain) is read as it is: the try from the
 * session; or, when there is none, the first when the session has steps
 * (norm_tries lists them once, as the model's own, when they are its own
 * by value) or when neither has any (both read the raw as it is). -1 when
 * the session has none and the model its own. */
function sessionsTry(tries: readonly NormTry[], session: readonly Step[]): number {
  const i = tries.findIndex((t) => t.from === "session");
  if (i >= 0) return i;
  return session.length || tries[0].from === "none" ? 0 : -1;
}

/** Where a try's input normalization came from (norm_tries's `from`), as the log says it. */
function through(from: string): string {
  return from === "declared" ? "its declared input normalization"
    : from === "assumed" ? "the assumed [-1, 1] input normalization"
      : from === "session" ? "the session's input normalization" : "no input normalization";
}

/** A model's score on the paint (null when it counted nothing), with how
 * many of its output blocks the paint reaches, which of them were read (at
 * most RANK_BLOCKS, spread over the paint: [lo, hi) on its level), and how
 * many painted chunks lie past its output. */
interface PaintScore { score: RankScore | null; blocks: number; read: { lo: number[]; hi: number[] }[]; outside: number }

/** `served`'s score on the paint of `volume`'s chunks `indices` (PaintScore). */
async function scorePaint(ctx: AppContext, dataset: DatasetInfo, volume: PaintVolume, indices: number[][], served: ServedModel): Promise<PaintScore> {
  const entry = served.entry;
  const level = servedLevel(dataset, entry), block = outputShape(geometryOf(entry));
  const transform = targetTransformFor(entry, ctx.infer), affinity = hasAffinityChannels(entry);
  // Each painted chunk with the model's voxel under each of its painted
  // voxels (label-it's place, the Python's placement, over the painted
  // voxels' box, which it shrinks to what the level covers), and the
  // model's blocks those lie in.
  const placed: { chunk: PaintedChunk; origin: number[]; at: Placement }[] = [];
  const blocks = new Map<string, number[]>();
  let outside = 0;
  for (const index of indices) {
    const chunk = await readPaintedChunk(ctx.paint, volume, index);
    const box = chunk && paintedBox(chunk);
    if (!chunk || !box) continue;
    const origin = index.map((i, a) => i * volume.chunk_size[a]);
    const at = await place(ctx.infer, volume, box.lo.map((v, a) => v + origin[a]), box.hi.map((v, a) => v + origin[a]), level);
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
    read.set(key, { lo, hi, shape: out.shape, data: await logitsOf(ctx, out, served.info.output_class) });
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
 * the Python's, so a sigmoid model's 0.9 and a logit model's 2.2 are one
 * and the same), clipped to [1e-6, 1 - 1e-6], then its logit. A logit
 * model's output comes back as it was, within float32; a probability's is
 * no longer read as if 0.1 said object. `out` is handed to the worker. */
async function logitsOf(ctx: AppContext, out: { data: Float32Array; shape: number[] }, outputClass: string | null | undefined): Promise<Float32Array> {
  const p = (await probability(ctx.infer, out, outputClass)).data;
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
 * from `at` (place over a box of the volume, `origin` the chunk's first
 * voxel). */
function forEachPainted(chunk: PaintedChunk, origin: number[], at: Placement, visit: (i: number, served: number[]) => void): void {
  const [, h, w] = chunk.shape;
  for (let z = at.lo[0]; z < at.hi[0]; z++) for (let y = at.lo[1]; y < at.hi[1]; y++) for (let x = at.lo[2]; x < at.hi[2]; x++) {
    const i = ((z - origin[0]) * h + y - origin[1]) * w + x - origin[2];
    if (chunk.labels[i]) visit(i, [at.indices[0][z - at.lo[0]], at.indices[1][y - at.lo[1]], at.indices[2][x - at.lo[2]]]);
  }
}

function compareIndex(p: number[], q: number[]): number {
  return p[0] - q[0] || p[1] - q[1] || p[2] - q[2];
}

/** A model's tallies of the paint so far, as the Python answers them and takes them back; nothing here reads them. */
type Tallies = Record<string, unknown>[];

/** A model's score on the paint, as rank_score answers it (portable/rank.py): the
 * balanced accuracy at the model's own cut-off, its parts, the score at 0, the loss,
 * the voxels counted and a binary model's channel, by name and by index into its
 * channels, which Use this model sets the form's Channel to (both null for an
 * affinity model's, scored on its channels together), with what each of a binary
 * model's channels matched, the best first (channel_matches; none for an affinity
 * model's). */
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
  channel_index: number | null;
  channel_matches?: { channel: string; channel_index: number; matches: number }[];
}

/** A model's row of the ranking, as score_row answers it (portable/rank.py): the
 * score kept (the stretched one when it matched WINDOW_MARGIN better), the raw's
 * window when that was kept and the match without it, the cut-off as Seed from
 * Prediction's threshold (a probability), the raw's range the model read as 0
 * to 1 (null without a MinMaxNormalizer), and the normalizer steps the score kept
 * was read through, which Use this model and Fit to Paint make the session's, with
 * where they came from (norm_tries's `from`), and what the other normalizations
 * tried matched. */
interface RankRow extends RankScore {
  window: [number, number] | null;
  matches_without_window?: number;
  threshold: number;
  input_window: [number, number] | null;
  input_norm: Step[];
  input_norm_from: string;
  input_norm_others: NormMatch[];
}

/** What a try other than the one kept matched: its `from` and its score's match. */
interface NormMatch { from: string; matches: number }

/** One input normalization a model is scored through, as rank.norm_tries answers
 * it: the steps, and where they came from ("declared", the model's manifest;
 * "assumed", assumedInputNorm's for a model that declares none; "session"; or
 * "none", the raw as it is, when neither has any). */
interface NormTry { input_norm: Step[]; from: string }

/** The ranking's row from `base`, the score read through `norm` (which came from
 * `normFrom`), `stretched`, the score read through `norm` stretched to `raw` (or
 * null), and `others`, what the other tries matched: rank.score_row. */
function rankRow(ctx: AppContext, base: RankScore, stretched: RankScore | null, norm: readonly Step[], raw: readonly [number, number] | null, normFrom: string, others: NormMatch[]): Promise<RankRow | null> {
  return ctx.infer.python<RankRow | null>(`${PORTABLE}:rank_row`, { base, stretched, norm, raw, norm_from: normFrom, others });
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

/** The Models tab's sections a candidate is listed under, in the order the
 * list gives them, as the server's ranking groups its own ("Running", its
 * catalog's groups, "Hugging Face", "BioImage Zoo"): the models served now;
 * the catalog's own, cellmap/onnx-web-models' (HF_REPO); those added by
 * hand (POST /api/finetune/rank/add), and any other repo's; and the models
 * finetuned in this browser (FINETUNED_REPO). */
const GROUPS = ["Running", "cellmap", "Added", "Finetuned"] as const;
type Group = (typeof GROUPS)[number];

/** A model of the ranking's list, as rank.js shows and filters it: what it
 * predicts, at what voxel size, in 2D or 3D (dims, by the slices it reads),
 * the group it is listed under, whether it is served and whether the
 * session's volume is painted for it. Each starts ticked: the browser runs
 * every one itself, where the server starts one not running on the cluster
 * for its score and leaves it unticked. */
interface Candidate {
  name: string;
  repo: string;
  folder: string;
  task: string;
  channels: string[];
  voxel_size_nm: number[];
  dims: 2 | 3;
  group: Group;
  finetuned: boolean;
  served: boolean;
  ticked: true;
  volume_model: boolean;
}

/** Each entry's candidate, for the session as it is now (`volume` its
 * volume): the listing's and rank/add's answer alike. Its group is Running
 * while it is served, Added once added by hand, else its repo's. */
function listingOf(ctx: AppContext, volume: PaintVolume | null): (entry: ModelEntry) => Candidate {
  const served = new Set(ctx.infer.served().map((s) => s.entry.id));
  const added = new Set(ctx.session.state.rankAdded);
  const group = (e: ModelEntry): Group => {
    if (served.has(e.id)) return "Running";
    if (added.has(e.id)) return "Added";
    return e.repo === HF_REPO ? "cellmap" : e.repo === FINETUNED_REPO ? "Finetuned" : "Added";
  };
  return (e) => ({
    name: e.name, repo: e.repo, folder: e.folder, task: e.manifest.task, channels: e.manifest.channels,
    voxel_size_nm: e.manifest.output_voxel_size_nm, dims: readsOneSlice(e) ? 2 : 3, group: group(e),
    finetuned: e.repo === FINETUNED_REPO, served: served.has(e.id), ticked: true, volume_model: volume?.model_id === e.id,
  });
}

async function sessionVolume(ctx: AppContext): Promise<PaintVolume | null> {
  const id = ctx.session.state.paintVolume;
  return id ? readVolume(ctx.paint, id) : null;
}

/** A model served for a score, and its release. */
interface Held { served: ServedModel; release: () => Promise<void> }

/** `entry` served for a score read through the normalizer steps
 * `inputNorm`: as it is served already when `sessions` says they are the
 * session chain's (it is served through that chain), or else served now
 * with them and no postprocessing (the model stage's own output is what is
 * read), and released when the score is done. Served now, it is its
 * smallest block at the precision this GPU takes (Catalog.chooseVariant
 * without a buffer limit): a 3D model's largest block can hold most of a
 * GPU's memory, beside the models already served, and a GPU run out of
 * memory loses its device and every model with it. */
async function servedForRanking(ctx: AppContext, entry: ModelEntry, inputNorm: readonly Step[], sessions: boolean): Promise<Held> {
  const current = sessions && ctx.infer.served().find((s) => s.entry.id === entry.id);
  if (current) return { served: current, release: async () => {} };
  const limits = { ...(await ctx.infer.gpuLimits()), maxBufferSize: null };
  const tuned = entryFor(entry.folder, entry.manifest, ctx.catalog.chooseVariant(entry.manifest, limits), entry.repo);
  const chain: Chain = { input_norm: [...inputNorm], postprocess: [] };
  const served = await ctx.infer.serve(tuned, chain, (await blobFor(chain)).blob, () => ctx.catalog.modelBytes(tuned));
  return { served, release: () => served.dispose() };
}
