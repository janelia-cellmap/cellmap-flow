// Which model to start from (backend/routes/rank.ts): the routes read the
// paint and the model's output and hand them to cellmap_flow.portable, the
// Python both modes run (here in node's Pyodide, test/unit/pyodide.ts; the
// score and the carry themselves are tested in tests/finetune/test_rank.py).
// The score route reads a served model as it is and serves any other for its
// score alone; a known paint and output rank as they always did; each row
// gives its threshold and input window; Use this model carries the paint
// onto the model's grid and makes its volume the session's, and Fit to
// Paint's window route stretches the session's normalization alone.
// Without Pyodide (no network for numpy's wheel) these skip.
import { describe, expect, it, vi } from "vitest";

import { writeBox } from "../../src/ai-annotate/fill";
import { Router } from "../../src/backend/router";
import { finetuneRoutes } from "../../src/backend/routes/finetune";
import { rankRoutes } from "../../src/backend/routes/rank";
import type { ApiRequest, ApiResponse, PaintVolume } from "../../src/contracts";
import { readVolume } from "../../src/finetune/volume";
import { entryFor } from "../../src/models";
import { paintLayerName } from "../../src/viewer/layers";
import { entry, type FakeContext, fakeContext, manifest, NAME_54, NAME_56, request } from "./backend-fakes";
import { portablePython } from "./pyodide";

// Pyodide and numpy load once for the file, a few seconds; each call after is quick.
vi.setConfig({ testTimeout: 120_000 });
const python = await portablePython();

function body(res: ApiResponse): any {
  if (!("json" in res)) throw new Error(`not a JSON answer: ${JSON.stringify(res)}`);
  return res.json;
}

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

describe.skipIf(!python)("the rank routes, on the portable Python", () => {
  it("score the model served as it is served, and serve another model for its score alone", async () => {
    const { ctx, handle } = await painted();
    const listed = body(await handle(request("GET", "/api/finetune/rank/candidates")));
    expect(listed.candidates.find((c: any) => c.name === NAME_56)).toMatchObject({ served: true, volume_model: true, dims: 3 });
    expect(listed.volume.painted_chunks).toBeGreaterThan(0);

    // A paint the model fits exactly: every object found, every background right, at 0.
    const nuc = body(await handle(request("POST", "/api/finetune/rank/score", { name: NAME_56 })));
    expect(nuc).toMatchObject({ success: true, matches: 1, objects_found: 1, background_right: 1, cutoff: 0, matches_at_zero: 1, channel: "nuc" });
    // At the model's own boundary, through the session's normalizer as it is.
    const minMax = ctx.session.state.chain.input_norm.find((n) => n.name === "MinMaxNormalizer")!;
    expect(nuc).toMatchObject({ window: null, threshold: 0.5, input_window: [Number(minMax.min_value), Number(minMax.max_value)] });
    expect(nuc.scored_voxels).toBe(nuc.object_voxels + nuc.background_voxels);
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
    expect(scored.input_window).toEqual(scored.window);
    // Every voxel the same: no window, and nothing served for one.
    ctx.infer.readRaw = async (_level, lo, hi) => new Uint8Array(hi.reduce((n, h, a) => n * (h - lo[a]), 1)).buffer;
    const serves = ctx.infer.serves.length;
    expect(body(await handle(request("POST", "/api/finetune/rank/score", { name: NAME_54 })))).toMatchObject({ success: true, window: null });
    expect(ctx.infer.serves.length).toBe(serves + 1);

    const used = body(await handle(request("POST", "/api/finetune/rank/use", { name: NAME_54, window: scored.window })));
    expect(used).toMatchObject({ success: true, window: scored.window });
    expect(ctx.session.state.chain.input_norm.find((n) => n.name === "MinMaxNormalizer")).toMatchObject({ min_value: scored.window[0], max_value: scored.window[1] });
  });

  it("fit the model seeding the paint: its threshold in its row, its window made the session's normalization, no volume made", async () => {
    const { ctx, handle } = await painted();
    ctx.infer.readRaw = async (_level, lo, hi) => Uint8Array.from({ length: hi.reduce((n, h, a) => n * (h - lo[a]), 1) }, (_, i) => 216 + (i % 21)).buffer;
    // The model served, seeing the object only with the raw stretched, at -1
    // over the rest at -4: a cut-off under its own fits the paint.
    const inside = (z: number, y: number, x: number) => [z, y, x].every((v) => v >= 5 && v < 15);
    const stretched = (view: string) => ctx.infer.serves.some((s) => view.includes(s.blob) && Number(s.chain.input_norm[0].min_value) > 0);
    ctx.infer.outputAt = (view, _c, z, y, x) => (stretched(view) && inside(z, y, x) ? -1 : -4);
    const volume = ctx.session.state.paintVolume;

    const scored = body(await handle(request("POST", "/api/finetune/rank/score", { name: NAME_56 })));
    expect(scored.cutoff).toBeLessThan(0);
    expect(scored.threshold).toBeCloseTo(1 / (1 + Math.exp(-scored.cutoff)), 3);
    expect(scored.input_window).toEqual(scored.window);
    const fitted = body(await handle(request("POST", "/api/finetune/rank/window", { window: scored.window })));
    expect(fitted).toMatchObject({ success: true, window: scored.window });
    const minMax = () => ctx.session.state.chain.input_norm.find((n) => n.name === "MinMaxNormalizer");
    expect(minMax()).toMatchObject({ min_value: scored.window[0], max_value: scored.window[1] });
    expect(ctx.session.state.paintVolume).toBe(volume);
    // Stretched already: the window is no narrower than the range, and nothing changes.
    expect(body(await handle(request("POST", "/api/finetune/rank/window", { window: scored.window }))).success).toBe(false);
    expect(minMax()).toMatchObject({ min_value: scored.window[0] });
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

  it("make the volume of the model used the session's, its paint carried and its paint layer in the old one's place", async () => {
    const { ctx, handle } = await painted();
    const before = ctx.session.state.paintVolume!;
    const used = body(await handle(request("POST", "/api/finetune/rank/use", { name: NAME_54 })));
    expect(used.error).toBeUndefined();
    expect(used).toMatchObject({ success: true, name: NAME_54, from_volume: before, painted_voxels: 20 ** 3, source_voxels: 20 ** 3, conflicts: 0 });
    expect(used.chunks).toBeGreaterThan(0);
    expect(ctx.session.state.paintVolume).toBe(used.volume_id);
    expect(ctx.infer.served().map((s) => s.entry.name)).toContain(NAME_54);
    expect(ctx.viewer.removed).toContain(paintLayerName(before));
    expect(Object.keys(ctx.viewer.layers)).toContain(paintLayerName(used.volume_id));
    expect(body(await handle(request("POST", "/api/finetune/rank/use", { name: NAME_54 }))).success).toBe(false);
  });
});

describe("the rank routes, without Python", () => {
  it("refuse without paint, and for a model the catalog does not have", async () => {
    const bare = harness(fakeContext());
    expect(body(await bare(request("POST", "/api/finetune/rank/score", { name: NAME_56 })))).toMatchObject({ success: false, needs_volume: true });
    const { handle } = await painted();
    const missing = await handle(request("POST", "/api/finetune/rank/score", { name: "nope" }));
    expect(missing).toMatchObject({ status: 404 });
  });
});
