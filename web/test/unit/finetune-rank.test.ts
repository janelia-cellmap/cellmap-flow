// Which model to start from (src/finetune/rank.ts, backend/routes/rank.ts):
// the score is balanced between object and background paint, forgives a
// binary model the edges of the strokes, and the score route reads a served
// model as it is and serves any other for its score alone.
import { describe, expect, it } from "vitest";

import { writeBox } from "../../src/ai-annotate/fill";
import { Router } from "../../src/backend/router";
import { finetuneRoutes } from "../../src/backend/routes/finetune";
import { rankRoutes } from "../../src/backend/routes/rank";
import type { ApiRequest, ApiResponse, PaintVolume } from "../../src/contracts";
import { emptyTallies, paintEdges, rankScore, tallyChunk } from "../../src/finetune/rank";
import { binaryTargets } from "../../src/finetune/targets";
import { readVolume } from "../../src/finetune/volume";
import { entry, type FakeContext, fakeContext, NAME_54, NAME_56, request } from "./backend-fakes";

function body(res: ApiResponse): any {
  if (!("json" in res)) throw new Error(`not a JSON answer: ${JSON.stringify(res)}`);
  return res.json;
}

/** A 1×4×10 chunk painted object (2) in x < `edge` and background (1) past it, and logits that say object for x < `called`. */
function stripes(edge: number, called: number) {
  const shape = [1, 4, 10], labels = new Uint8Array(40).map((_, i) => (i % 10 < edge ? 2 : 1));
  const logits = Float32Array.from({ length: 40 }, (_, i) => (i % 10 < called ? 3 : -3));
  return { chunk: { index: [0, 0, 0], shape, labels }, logits };
}

function score(edge: number, called: number, forgiveEdges: boolean) {
  const { chunk, logits } = stripes(edge, called);
  const tallies = emptyTallies(1);
  tallyChunk(tallies, chunk, binaryTargets(1)(chunk), (_c, i) => logits[i], forgiveEdges ? paintEdges(chunk) : null);
  return rankScore(tallies, ["nuc"], false)!;
}

describe("a model's score on the paint", () => {
  it("weighs object and background paint alike, so calling everything background scores a half", () => {
    const right = score(1, 1, false), allBackground = score(1, 0, false);
    expect(right.matches).toBe(1);
    // Nine voxels in ten are background, all called right: 90% of the voxels, half the score.
    expect(allBackground).toMatchObject({ matches: 0.5, objectsFound: 0, backgroundRight: 1 });
    expect(allBackground.loss).toBeGreaterThan(right.loss);
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
    expect(nuc).toMatchObject({ success: true, matches: 1, objects_found: 1, background_right: 1, channel: "nuc" });
    expect(nuc.loss).toBeLessThan(0.3);
    expect(ctx.infer.serves).toHaveLength(1);

    const aff = body(await handle(request("POST", "/api/finetune/rank/score", { name: NAME_54 })));
    expect(aff).toMatchObject({ success: true, channel: null });
    expect(aff.matches).toBeLessThan(nuc.matches);
    const servedForScore = ctx.infer.serves.at(-1)!;
    expect(servedForScore).toMatchObject({ name: NAME_54, chain: { input_norm: entry("setup_54").manifest.input_norm, postprocess: [] } });
    expect(ctx.infer.served().map((s) => s.entry.name)).toEqual([NAME_56]);
  });

  it("refuse without paint, and for a model the catalog does not have", async () => {
    const bare = harness(fakeContext());
    expect(body(await bare(request("POST", "/api/finetune/rank/score", { name: NAME_56 })))).toMatchObject({ success: false, needs_volume: true });
    const { handle } = await painted();
    const missing = await handle(request("POST", "/api/finetune/rank/score", { name: "nope" }));
    expect(missing).toMatchObject({ status: 404 });
  });
});
