// Which model to start from (backend/routes/rank.ts): the routes read the
// paint and the model's output and hand them to cellmap_flow.portable, the
// Python both modes run (here in node's Pyodide, test/unit/pyodide.ts; the
// score and the carry themselves are tested in tests/finetune/test_rank.py).
// The list puts each model under the Models tab's section it would be in, and
// a model added by hand (rank/add) under Added from then on, scored by its
// name. The score route reads a served model as it is and serves any other for its
// score alone; a model is read through its own input normalization and the
// session's, the better kept; a known paint and output rank as they always
// did; each row gives its threshold, input window and the steps it was read
// through; Use this model picks the row's channel before the model is
// served, sets those steps, carries the paint onto the model's grid, makes
// its volume the session's and retires the old one, and Fit to Paint's
// input-norm route sets the steps alone.
// Without Pyodide (no network for numpy's wheel) these skip.
import { describe, expect, it, vi } from "vitest";

import { writeBox } from "../../src/ai-annotate/fill";
import { Router } from "../../src/backend/router";
import { finetuneRoutes } from "../../src/backend/routes/finetune";
import { rankRoutes } from "../../src/backend/routes/rank";
import type { ApiRequest, ApiResponse, Chain, PaintVolume } from "../../src/contracts";
import { readVolume } from "../../src/finetune/volume";
import { entryFor, finetunedEntry } from "../../src/models";
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

/** setup_56 (nuc) served through the session's chain (`chain`, or a fresh
 * session's), a volume for it painted with a 10³ object in a 20³ box of
 * background, and its logits +4 inside that object, -4 outside. */
async function painted(chain?: Chain) {
  const ctx = fakeContext(chain ? { state: { chain } } : {});
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

  it("score a model added by hand by the name its candidate answers", async () => {
    const { handle } = await painted();
    const { candidate } = body(await handle(request("POST", "/api/finetune/rank/add", { ref: entry("setup_54").id })));
    expect(body(await handle(request("POST", "/api/finetune/rank/score", { name: candidate.name })))).toMatchObject({ success: true, name: NAME_54 });
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

    // Use this model makes the steps of the row the session's first, and serves the model through them.
    expect(scored.input_norm.find((n: any) => n.name === "MinMaxNormalizer")).toMatchObject({ min_value: scored.window[0], max_value: scored.window[1] });
    const used = body(await handle(request("POST", "/api/finetune/rank/use", { name: NAME_54, input_norm: scored.input_norm })));
    expect(used).toMatchObject({ success: true, input_norm: scored.input_norm });
    expect(ctx.session.state.chain.input_norm).toEqual(scored.input_norm);
    expect(ctx.infer.serves.at(-1)).toMatchObject({ name: NAME_54, chain: ctx.session.state.chain });
  });

  it("fit the model seeding the paint: its threshold in its row, its steps made the session's normalization, no volume made", async () => {
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
    const serves = ctx.infer.serves.length;
    const fitted = body(await handle(request("POST", "/api/finetune/rank/input-norm", { input_norm: scored.input_norm })));
    expect(fitted).toEqual({ success: true, input_norm: scored.input_norm, postprocess: ctx.session.state.chain.postprocess });
    expect(ctx.session.state.chain.input_norm.find((n) => n.name === "MinMaxNormalizer")).toMatchObject({ min_value: scored.window[0], max_value: scored.window[1] });
    expect(ctx.session.state.paintVolume).toBe(volume);
    // The model served redrawn through the steps, and no other served.
    expect(ctx.infer.serves.slice(serves).map((s) => s.name)).toEqual([NAME_56]);
    // Not a list of steps, or a step the browser cannot run: refused, and nothing changes.
    for (const input_norm of [undefined, "MinMaxNormalizer", [{ name: "ZScoreNormalizerThatIsNot" }]]) {
      expect(await handle(request("POST", "/api/finetune/rank/input-norm", { input_norm }))).toMatchObject({ status: 400 });
    }
    expect(ctx.session.state.chain.input_norm).toEqual(scored.input_norm);
  });

  it("read a model through its own input normalization and the session's, and keep the better", async () => {
    const inside = (z: number, y: number, x: number) => [z, y, x].every((v) => v >= 5 && v < 15);
    // The view a serve made reads through steps that `fits`.
    const through = (ctx: FakeContext, view: string, fits: (chain: Chain) => boolean) => ctx.infer.serves.some((s) => view.includes(s.blob) && fits(s.chain));
    const declared = entry("setup_56").manifest.input_norm;

    // No input normalization in the session: the served model reads the raw
    // as it is and sees nothing, its declared [-1, 1] sees the object.
    const bare = await painted({ input_norm: [], postprocess: [] });
    const scaled = (chain: Chain) => chain.input_norm.some((n) => n.name === "LambdaNormalizer");
    bare.ctx.infer.outputAt = (view, _c, z, y, x) => (through(bare.ctx, view, scaled) && inside(z, y, x) ? 4 : -4);
    const own = body(await bare.handle(request("POST", "/api/finetune/rank/score", { name: NAME_56 })));
    expect(own).toMatchObject({ success: true, matches: 1, input_norm: declared, input_norm_from: "declared" });

    // The session's another, which fits this raw better: kept, read through
    // the model served as it is; its own served for its score alone.
    const session = [{ name: "MinMaxNormalizer", min_value: 0, max_value: 100 }, declared[1]];
    const { ctx, handle } = await painted({ input_norm: session, postprocess: [] });
    const narrow = (chain: Chain) => chain.input_norm.some((n) => n.name === "MinMaxNormalizer" && Number(n.max_value) === 100);
    ctx.infer.outputAt = (view, _c, z, y, x) => (through(ctx, view, narrow) && inside(z, y, x) ? 4 : -4);
    const serves = ctx.infer.serves.length;
    const theirs = body(await handle(request("POST", "/api/finetune/rank/score", { name: NAME_56 })));
    expect(theirs).toMatchObject({ success: true, matches: 1, input_norm: session, input_norm_from: "session" });
    // What its own matched, in the row, which says why the session's.
    expect(theirs.input_norm_others).toEqual([{ from: "declared", matches: 0.5 }]);
    expect(theirs.note).toBeUndefined();
    expect(ctx.infer.serves.slice(serves).map((s) => s.chain.input_norm)).toEqual([declared]);
  });

  it("read a model that declares no input normalization through the assumed [-1, 1], and give its logits the sigmoid it needs", async () => {
    const { ctx, handle } = await painted({ input_norm: [], postprocess: [] });
    // fly_organelles' copies declare nothing: a model that sees the object only through [-1, 1].
    const m = { ...manifest("setup_61", ["mito"], 128), task: "mito", input_norm: [] };
    const bare = entryFor("setup_61", m, m.variants[0]);
    ctx.catalog.entries = [...ctx.catalog.entries, bare];
    const inside = (z: number, y: number, x: number) => [z, y, x].every((v) => v >= 5 && v < 15);
    const signed = (view: string) => ctx.infer.serves.some((s) => view.includes(s.blob) && s.chain.input_norm.some((n) => n.name === "LambdaNormalizer"));
    ctx.infer.outputAt = (view, _c, z, y, x) => (signed(view) && inside(z, y, x) ? 4 : -4);

    const row = body(await handle(request("POST", "/api/finetune/rank/score", { name: bare.name })));
    const assumed = [{ name: "MinMaxNormalizer", min_value: 0, max_value: 255, invert: false }, { name: "LambdaNormalizer", expression: "x*2-1" }];
    expect(row).toMatchObject({ success: true, matches: 1, input_norm: assumed, input_norm_from: "assumed" });
    expect(ctx.infer.serves.at(-1)).toMatchObject({ name: bare.name, chain: { input_norm: assumed } });
    // Logits shown through no postprocessing: the sigmoid that makes them probabilities.
    expect(row.postprocess).toEqual([{ name: "SigmoidPostprocessor" }]);
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
    // Under the session's sigmoid, the probabilities would be squashed again: theirs is taken away;
    // the logits' is what they need, so nothing changes.
    expect(asProbability.postprocess).toEqual([]);
    expect(asLogits.postprocess).toBeNull();
  });

  it("set the row's postprocessing beside its steps in one chain set, when Use this model and Fit to Paint are given it", async () => {
    const { ctx, handle } = await painted();
    const m = { ...manifest("setup_60", ["nuc"], 128), task: "nuc", output: "probabilities (sigmoid applied)" };
    const sigmoid = entryFor("setup_60", m, m.variants[0]);
    ctx.catalog.entries = [...ctx.catalog.entries, sigmoid];
    const row = body(await handle(request("POST", "/api/finetune/rank/score", { name: sigmoid.name })));
    const stretch = [{ ...row.input_norm[0], max_value: 200 }, row.input_norm[1]];
    // A postprocessor the browser cannot run: refused, and nothing changes.
    const before = structuredClone(ctx.session.state.chain);
    expect(await handle(request("POST", "/api/finetune/rank/use", { name: sigmoid.name, input_norm: stretch, postprocess: [{ name: "NotAPostprocessor" }] }))).toMatchObject({ status: 400 });
    expect(ctx.session.state.chain).toEqual(before);

    const chainSets = () => ctx.logged.filter((line) => line.startsWith("Chain ")).length;
    const sets = chainSets();
    const used = body(await handle(request("POST", "/api/finetune/rank/use", { name: sigmoid.name, input_norm: stretch, postprocess: row.postprocess })));
    expect(used).toMatchObject({ success: true, input_norm: stretch, postprocess: [] });
    expect(ctx.session.state.chain).toEqual({ input_norm: stretch, postprocess: [] });
    expect(chainSets()).toBe(sets + 1);
    expect(ctx.infer.serves.at(-1)).toMatchObject({ name: sigmoid.name, chain: { input_norm: stretch, postprocess: [] } });

    // Fit to Paint: the postprocessing alone changes, and the answer is the chain as it now is.
    const fitted = body(await handle(request("POST", "/api/finetune/rank/input-norm", { input_norm: stretch, postprocess: [{ name: "SigmoidPostprocessor" }] })));
    expect(fitted).toEqual({ success: true, input_norm: stretch, postprocess: [{ name: "SigmoidPostprocessor" }] });
    expect(ctx.session.state.chain.postprocess).toEqual([{ name: "SigmoidPostprocessor" }]);
  });

  it("make the volume of the model used the session's, its paint carried and its paint layer in the old one's place", async () => {
    const { ctx, handle } = await painted();
    const before = ctx.session.state.paintVolume!;
    // Steps the browser cannot run are refused before anything is served or made.
    expect(await handle(request("POST", "/api/finetune/rank/use", { name: NAME_54, input_norm: [{ name: "NotANormalizer" }] }))).toMatchObject({ status: 400 });
    expect(ctx.session.state.paintVolume).toBe(before);
    // The session's steps already: nothing to set.
    const used = body(await handle(request("POST", "/api/finetune/rank/use", { name: NAME_54, input_norm: ctx.session.state.chain.input_norm })));
    expect(used.error).toBeUndefined();
    expect(used).toMatchObject({ success: true, name: NAME_54, from_volume: before, painted_voxels: 20 ** 3, source_voxels: 20 ** 3, conflicts: 0, input_norm: null });
    expect(used.chunks).toBeGreaterThan(0);
    expect(ctx.session.state.paintVolume).toBe(used.volume_id);
    expect(ctx.infer.served().map((s) => s.entry.name)).toContain(NAME_54);
    expect(ctx.viewer.removed).toContain(paintLayerName(before));
    expect(Object.keys(ctx.viewer.layers)).toContain(paintLayerName(used.volume_id));
    expect(body(await handle(request("POST", "/api/finetune/rank/use", { name: NAME_54 }))).success).toBe(false);
  });

  it("retire the old volume when a model is used: kept, superseded, no longer listed, and resuming it resumes the new one", async () => {
    const { ctx, handle } = await painted();
    const before = ctx.session.state.paintVolume!;
    const used = body(await handle(request("POST", "/api/finetune/rank/use", { name: NAME_54 })));
    expect(used.success).toBe(true);
    // Both volumes are in the store; the old one names the one that took its place.
    expect(await ctx.paint.volumes()).toEqual([before, used.volume_id].sort());
    expect(await readVolume(ctx.paint, before)).toMatchObject({ superseded_by: used.volume_id });
    // One session listed: the new volume, its chunks alone, the model it is for.
    const { sessions } = body(await handle(request("POST", "/api/finetune/list-existing-sessions", {})));
    expect(sessions).toHaveLength(1);
    expect(sessions[0]).toMatchObject({ session_id: used.volume_id, chunk_count: used.chunks, model_name: NAME_54, volumes: [{ volume_id: used.volume_id }] });
    // The old id (a list shown before Use this model) resumes the new volume.
    await ctx.session.update({ paintVolume: null });
    expect(body(await handle(request("POST", "/api/finetune/load-existing-volume", { source_session_path: before })))).toMatchObject({ success: true, volume_id: used.volume_id });
    expect(ctx.session.state.paintVolume).toBe(used.volume_id);
  });

  it("pick the row's channel before the model is served, so its layer opens on it", async () => {
    const { ctx, handle } = await painted();
    const organelles = entry("fly_organelles_run08", "organelles", ["all_mem", "organelle", "mito", "er", "nucleus", "pm", "vs", "ld"]);
    ctx.catalog.entries = [...ctx.catalog.entries, organelles];
    // Past its channels: refused, nothing picked or served.
    expect(await handle(request("POST", "/api/finetune/rank/use", { name: organelles.name, select_channel: 8 }))).toMatchObject({ status: 400 });
    expect(ctx.session.state.shownChannels).toEqual({});
    expect(ctx.infer.served().map((s) => s.entry.name)).not.toContain(organelles.name);
    const used = body(await handle(request("POST", "/api/finetune/rank/use", { name: organelles.name, select_channel: 2 })));
    expect(used.success).toBe(true);
    expect(ctx.session.state.shownChannels).toEqual({ [organelles.name]: 2 });
    expect(ctx.viewer.layers[organelles.name].localPosition).toEqual([2]);
  });
});

describe("the rank routes, without Python", () => {
  it("list each model under its Models tab section, in the sections' order, and one added by hand under Added from then on, unserved", async () => {
    const ctx = fakeContext();
    const [e54, e56] = ctx.catalog.entries;
    const tuned = finetunedEntry(e56, `${NAME_56}-ft1`);
    const er = entry("setup_60", "er", ["er"], 32);
    ctx.catalog.entries = [tuned, e54, e56, er];
    await ctx.infer.serve(e56, ctx.session.state.chain, "blob", async () => new ArrayBuffer(8));
    const handle = harness(ctx);
    const groups = async () => body(await handle(request("GET", "/api/finetune/rank/candidates"))).candidates.map((c: any) => [c.name, c.group]);
    expect(await groups()).toEqual([[NAME_56, "Running"], [NAME_54, "cellmap"], [er.name, "cellmap"], [tuned.name, "Finetuned"]]);

    // Added by its id, a ref the Models tab's Add takes, and answered as the list will list it.
    const added = await handle(request("POST", "/api/finetune/rank/add", { ref: e54.id }));
    expect(body(added)).toMatchObject({ success: true, candidate: { name: NAME_54, group: "Added", ticked: true, served: false } });
    expect(await groups()).toEqual([[NAME_56, "Running"], [er.name, "cellmap"], [NAME_54, "Added"], [tuned.name, "Finetuned"]]);
    expect(ctx.infer.served().map((s) => s.entry.name)).toEqual([NAME_56]);

    // A ref the browser cannot run is refused as the Models tab's resolve refuses it.
    const refused = await handle(request("POST", "/api/finetune/rank/add", { ref: "/groups/models/mito.pt" }));
    expect(refused).toMatchObject({ status: 503, json: { success: false } });
    expect(ctx.session.state.rankAdded).toEqual([e54.id]);
  });

  it("refuse without paint, and for a model the catalog does not have", async () => {
    const bare = harness(fakeContext());
    expect(body(await bare(request("POST", "/api/finetune/rank/score", { name: NAME_56 })))).toMatchObject({ success: false, needs_volume: true });
    const { handle } = await painted();
    const missing = await handle(request("POST", "/api/finetune/rank/score", { name: "nope" }));
    expect(missing).toMatchObject({ status: 404 });
  });
});
