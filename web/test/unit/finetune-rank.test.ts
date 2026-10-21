// Which model to start from (src/finetune/rank.ts, carry.ts, backend/routes/rank.ts):
// the score is balanced between object and background paint, forgives a
// binary model the edges of the strokes, and the score route reads a served
// model as it is and serves any other for its score alone; Use this model
// carries the paint onto the model's grid and makes its volume the session's.
import { describe, expect, it } from "vitest";

import { readBox, writeBox } from "../../src/ai-annotate/fill";
import { Router } from "../../src/backend/router";
import { finetuneRoutes } from "../../src/backend/routes/finetune";
import { rankRoutes } from "../../src/backend/routes/rank";
import type { ApiRequest, ApiResponse, PaintVolume } from "../../src/contracts";
import { carryPaint } from "../../src/finetune/carry";
import { emptyTallies, paintEdges, rankScore, tallyChunk } from "../../src/finetune/rank";
import { binaryTargets } from "../../src/finetune/targets";
import { readVolume } from "../../src/finetune/volume";
import { entryFor } from "../../src/models";
import { MemoryPaintStore } from "../../src/state/paint-store";
import { paintLayerName } from "../../src/viewer/layers";
import { entry, type FakeContext, fakeContext, manifest, NAME_54, NAME_56, request } from "./backend-fakes";

function body(res: ApiResponse): any {
  if (!("json" in res)) throw new Error(`not a JSON answer: ${JSON.stringify(res)}`);
  return res.json;
}

/** A 1×4×10 chunk painted object (2) in x < `edge` and background (1) past
 * it, and logits that say object for x < `called`: `high` there, `low` past it. */
function stripes(edge: number, called: number, high = 3, low = -3) {
  const shape = [1, 4, 10], labels = new Uint8Array(40).map((_, i) => (i % 10 < edge ? 2 : 1));
  const logits = Float32Array.from({ length: 40 }, (_, i) => (i % 10 < called ? high : low));
  return { chunk: { index: [0, 0, 0], shape, labels }, logits };
}

function score(edge: number, called: number, forgiveEdges: boolean, high = 3, low = -3) {
  const { chunk, logits } = stripes(edge, called, high, low);
  const tallies = emptyTallies(1);
  tallyChunk(tallies, chunk, binaryTargets(1)(chunk), (_c, i) => logits[i], forgiveEdges ? paintEdges(chunk) : null);
  return rankScore(tallies, ["nuc"], false)!;
}

describe("a model's score on the paint", () => {
  it("weighs object and background paint alike, so calling everything background scores a half", () => {
    const right = score(1, 1, false), allBackground = score(1, 0, false);
    expect(right.matches).toBe(1);
    // Nine voxels in ten are background, all called right: 90% of the voxels, half the score
    // (and no cut-off does better: every voxel has the same logit).
    expect(allBackground).toMatchObject({ matches: 0.5, matchesAtZero: 0.5 });
    expect(allBackground.loss).toBeGreaterThan(right.loss);
  });

  it("reads a model that sees the objects but calls none of them at 0 at the cut-off that fits the paint", () => {
    // Objects at -1, background at -3: nothing is called object at 0.
    const shifted = score(5, 5, false, -1, -3);
    expect(shifted).toMatchObject({ matchesAtZero: 0.5, matches: 1, objectsFound: 1, backgroundRight: 1 });
    expect(shifted.cutoff).toBeGreaterThanOrEqual(-3);
    expect(shifted.cutoff).toBeLessThan(-1);
    expect(score(5, 5, false).cutoff).toBe(0);  // one that is right at 0 keeps it
  });

  it("does not hold a binary model's boundary a voxel off against it", () => {
    expect(score(5, 6, false).matches).toBeLessThan(1);
    expect(score(5, 6, true)).toMatchObject({ matches: 1, scored: 32 });
  });
});

/** The finetune routes (create-volume) beside the rank routes, on `ctx`. */
function harness(ctx: FakeContext) {
  const router = new Router();
  finetuneRoutes(router, { run: async () => { throw new Error("no finetune run here"); } });
  rankRoutes(router);
  return (req: ApiRequest) => router.handle(req, ctx);
}

/** setup_56 (nuc) served, a volume for it painted with a 10³ object in a
 * 20³ box of background, and its logits +4 inside that object, -4 outside. */
async function painted() {
  const ctx = fakeContext();
  const nuc = entry("setup_56", "nuc", ["nuc"], 128);
  await ctx.infer.serve(nuc, ctx.session.state.chain, "blob", async () => new ArrayBuffer(8));
  await ctx.session.update({ models: [nuc] });
  const handle = harness(ctx);
  const created = body(await handle(request("POST", "/api/finetune/create-volume", { model_name: NAME_56 })));
  const volume = (await readVolume(ctx.paint, created.volume_id)) as PaintVolume;
  const inside = (z: number, y: number, x: number) => [z, y, x].every((v) => v >= 5 && v < 15);
  const labels = new Uint8Array(20 ** 3).map((_, i) => (inside(Math.floor(i / 400), Math.floor(i / 20) % 20, i % 20) ? 2 : 1));
  await writeBox(ctx.paint, volume, [0, 0, 0], [20, 20, 20], labels);
  ctx.infer.outputAt = (_view, _c, z, y, x) => (inside(z, y, x) ? 4 : -4);
  return { ctx, handle };
}

describe("the rank routes", () => {
  it("score the model served as it is served, and serve another model for its score alone", async () => {
    const { ctx, handle } = await painted();
    const listed = body(await handle(request("GET", "/api/finetune/rank/candidates")));
    expect(listed.candidates.find((c: any) => c.name === NAME_56)).toMatchObject({ served: true, volume_model: true, dims: 3 });
    expect(listed.volume.painted_chunks).toBeGreaterThan(0);

    const nuc = body(await handle(request("POST", "/api/finetune/rank/score", { name: NAME_56 })));
    expect(nuc).toMatchObject({ success: true, matches: 1, objects_found: 1, background_right: 1, cutoff: 0, matches_at_zero: 1, channel: "nuc" });
    expect(nuc.loss).toBeLessThan(0.3);
    expect(ctx.infer.serves).toHaveLength(1);

    const aff = body(await handle(request("POST", "/api/finetune/rank/score", { name: NAME_54 })));
    expect(aff).toMatchObject({ success: true, channel: null });
    expect(aff.matches).toBeLessThan(nuc.matches);
    const servedForScore = ctx.infer.serves.at(-1)!;
    expect(servedForScore).toMatchObject({ name: NAME_54, chain: { input_norm: entry("setup_54").manifest.input_norm, postprocess: [] } });
    expect(ctx.infer.served().map((s) => s.entry.name)).toEqual([NAME_56]);
  });

  it("score a model again with narrow raw stretched, keep that only when it scores clearly better, and Use this model sets it", async () => {
    const { ctx, handle } = await painted();
    // Raw spanning 216 to 236 grey levels, as Jurkat's.
    ctx.infer.readRaw = async (_level, lo, hi) => Uint8Array.from({ length: hi.reduce((n, h, a) => n * (h - lo[a]), 1) }, (_, i) => 216 + (i % 21)).buffer;
    // A model that sees the object only when its normalizer stretches the raw.
    const inside = (z: number, y: number, x: number) => [z, y, x].every((v) => v >= 5 && v < 15);
    const stretched = (view: string) => ctx.infer.serves.some((s) => view.includes(s.blob) && Number(s.chain.input_norm[0].min_value) > 0);
    ctx.infer.outputAt = (view, _c, z, y, x) => (stretched(view) && inside(z, y, x) ? 4 : -4);
    const scored = body(await handle(request("POST", "/api/finetune/rank/score", { name: NAME_54 })));
    expect(scored.window[0]).toBeGreaterThanOrEqual(216);
    expect(scored.window[1]).toBeLessThanOrEqual(236);
    expect(scored.matches).toBeGreaterThan(scored.matches_without_window + 0.03);
    // Every voxel the same: no window, and nothing served for one.
    ctx.infer.readRaw = async (_level, lo, hi) => new Uint8Array(hi.reduce((n, h, a) => n * (h - lo[a]), 1)).buffer;
    const serves = ctx.infer.serves.length;
    expect(body(await handle(request("POST", "/api/finetune/rank/score", { name: NAME_54 })))).toMatchObject({ success: true, window: null });
    expect(ctx.infer.serves.length).toBe(serves + 1);

    const used = body(await handle(request("POST", "/api/finetune/rank/use", { name: NAME_54, window: scored.window })));
    expect(used).toMatchObject({ success: true, window: scored.window });
    expect(ctx.session.state.chain.input_norm.find((n) => n.name === "MinMaxNormalizer")).toMatchObject({ min_value: scored.window[0], max_value: scored.window[1] });
  });

  it("read a model's output by its activation, so a sigmoid model scores as the logit model it is equal to", async () => {
    const { ctx, handle } = await painted();
    // A nuc model at setup_56's voxel size whose manifest says it outputs probabilities.
    const m = { ...manifest("setup_60", ["nuc"], 128), task: "nuc", output: "probabilities (sigmoid applied)" };
    const sigmoid = entryFor("setup_60", m, m.variants[0]);
    ctx.catalog.entries = [...ctx.catalog.entries, sigmoid];
    const inside = (z: number, y: number, x: number) => [z, y, x].every((v) => v >= 5 && v < 15);
    // 0.9 inside the object and 0.1 outside; setup_56 (raw logits) says the same as log(9) and -log(9).
    const logit = Math.log(9);
    ctx.infer.outputAt = (view, _c, z, y, x) => (view.startsWith(sigmoid.name) ? (inside(z, y, x) ? 0.9 : 0.1) : inside(z, y, x) ? logit : -logit);

    const asProbability = body(await handle(request("POST", "/api/finetune/rank/score", { name: sigmoid.name })));
    const asLogits = body(await handle(request("POST", "/api/finetune/rank/score", { name: NAME_56 })));
    // Read as logits, 0.1 said object too: half the paint right at 0, and a cut-off between the two values.
    expect(asProbability).toMatchObject({ success: true, matches: 1, matches_at_zero: 1, cutoff: 0 });
    expect(asLogits).toMatchObject({ success: true, matches: 1, matches_at_zero: 1, cutoff: 0 });
    expect(asProbability.loss).toBeCloseTo(asLogits.loss, 5);
  });

  it("refuse without paint, and for a model the catalog does not have", async () => {
    const bare = harness(fakeContext());
    expect(body(await bare(request("POST", "/api/finetune/rank/score", { name: NAME_56 })))).toMatchObject({ success: false, needs_volume: true });
    const { handle } = await painted();
    const missing = await handle(request("POST", "/api/finetune/rank/score", { name: "nope" }));
    expect(missing).toMatchObject({ status: 404 });
  });
});

/** A volume of `shape` voxels of `voxel` nm (voxel 0's centre at voxel / 2, so both grids start at 0 nm) in chunks of `chunk`. */
function volumeOf(id: string, voxel: number, shape: number, chunk: number): PaintVolume {
  return { id, dtype: "uint8", chunk_size: [chunk, chunk, chunk], output_voxel_size: [voxel, voxel, voxel], dataset_offset_nm: [voxel / 2, voxel / 2, voxel / 2], dataset_shape_voxels: [shape, shape, shape] } as PaintVolume;
}

describe("paint carried to another model's grid", () => {
  it("repeats a voxel onto a finer grid, and drops only where object and background meet on a coarser one", async () => {
    const store = new MemoryPaintStore();
    const from = volumeOf("from", 16, 8, 4);
    // A row of x: background, background, object, object (labels 1 1 2 2) at z = y = 0.
    await writeBox(store, from, [0, 0, 0], [1, 1, 4], Uint8Array.of(1, 1, 2, 2));
    const fine = volumeOf("fine", 8, 16, 8);
    expect(await carryPaint(store, from, fine)).toMatchObject({ painted: 32, source: 4, conflicts: 0 });
    expect(Array.from(await readBox(store, fine, [1, 1, 0], [2, 2, 8]))).toEqual([1, 1, 1, 1, 2, 2, 2, 2]);
    // At 32 nm, x voxel 0 holds both backgrounds and voxel 1 both objects: no conflict.
    const coarse = volumeOf("coarse", 32, 4, 2);
    expect(await carryPaint(store, from, coarse)).toMatchObject({ painted: 2, conflicts: 0 });
    // Background then object inside 32 nm voxel 1 (x 2 and 3): left unpainted. A lone 16 nm
    // object voxel (x 5) still lands on its 32 nm voxel.
    await writeBox(store, from, [0, 0, 0], [1, 1, 6], Uint8Array.of(1, 1, 1, 2, 0, 2));
    const shifted = volumeOf("shifted", 32, 4, 2);
    expect(await carryPaint(store, from, shifted)).toMatchObject({ painted: 2, conflicts: 1 });
    expect(Array.from(await readBox(store, shifted, [0, 0, 0], [1, 1, 3]))).toEqual([1, 0, 2]);
  });

  it("makes the volume of the model used the session's, its paint layer in the old one's place", async () => {
    const { ctx, handle } = await painted();
    const before = ctx.session.state.paintVolume!;
    const used = body(await handle(request("POST", "/api/finetune/rank/use", { name: NAME_54 })));
    expect(used).toMatchObject({ success: true, name: NAME_54, from_volume: before, painted_voxels: 20 ** 3, conflicts: 0 });
    expect(ctx.session.state.paintVolume).toBe(used.volume_id);
    expect(ctx.infer.served().map((s) => s.entry.name)).toContain(NAME_54);
    expect(ctx.viewer.removed).toContain(paintLayerName(before));
    expect(Object.keys(ctx.viewer.layers)).toContain(paintLayerName(used.volume_id));
    expect(body(await handle(request("POST", "/api/finetune/rank/use", { name: NAME_54 }))).success).toBe(false);
  });
});
