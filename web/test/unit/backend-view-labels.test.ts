// The Label it routes in the browser (src/backend/routes/view-labels.ts),
// driven as label-it.js drives them: a volume created through the finetune
// routes for a served model, the viewer's position set, the model's output
// a fake's (FakeInference.outputAt), and the answers, the paint store, the
// viewer's refreshes and the undo stack checked against the Python routes'
// behaviour (view_labels.py). A seed's maths and every tool's box, ids,
// writes and undo are the portable Python (label_it.py, fill.py), run here
// in node's Pyodide (test/unit/pyodide.ts) and tested by pytest; the mutex
// watershed is parity-tested against the Python in
// test/parity/label-it.test.ts; these tests are about the routes. Without
// Pyodide (no network for its wheels) every test but the sources' skips.
import { describe, expect, it, vi } from "vitest";

import { readBox, writeBox } from "../../src/ai-annotate/fill";
import { Router } from "../../src/backend/router";
import { finetuneRoutes } from "../../src/backend/routes/finetune";
import { LARGE_BOX_VOXELS, viewLabelsRoutes } from "../../src/backend/routes/view-labels";
import type { ApiRequest, ApiResponse, ModelEntry, PaintVolume } from "../../src/contracts";
import type { RunFinetune } from "../../src/finetune/types";
import { readVolume } from "../../src/finetune/volume";
import { paintLayerName } from "../../src/viewer/layers";
import { entry, type FakeContext, fakeContext, NAME_54, NAME_56, request } from "./backend-fakes";
import { portablePython } from "./pyodide";

// Pyodide and numpy load once for the file, scipy at the first seed, a few
// seconds each; each call after is quick.
vi.setConfig({ testTimeout: 120_000 });
const python = await portablePython();

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function body(res: ApiResponse): any {
  if (!("json" in res)) throw new Error(`not a JSON answer: ${JSON.stringify(res)}`);
  return res.json;
}

const BASE = "/api/finetune/view-labels";
const NOW = new Date(2026, 9, 16, 12, 0, 0);
/** The centre of the first 60^3 chunk of a volume on the fakes' dataset:
 * the models read its one 8 nm level (as if at their own voxel size), so
 * their volumes are 8 nm voxels from voxel 0's centre at 0, and voxel i's
 * centre is at 8i nm (finetune/volume.ts planVolume). */
const CENTRE = [236, 236, 236];
/** Voxel 500's centre: a box around it lies inside the volume whatever its size. */
const MIDDLE = [4000, 4000, 4000];

/** The Label it routes beside the finetune routes (create-volume and add-to-viewer). */
function harness(ctx: FakeContext) {
  const router = new Router();
  const run: RunFinetune = async () => {
    throw new Error("no finetune run in these tests");
  };
  finetuneRoutes(router, { run, now: () => NOW });
  viewLabelsRoutes(router);
  return { handle: (req: ApiRequest) => router.handle(req, ctx) };
}

async function serve(ctx: FakeContext, ...entries: ModelEntry[]): Promise<void> {
  for (const e of entries) await ctx.infer.serve(e, ctx.session.state.chain, "blob", async () => new ArrayBuffer(8));
  await ctx.session.update({ models: [...ctx.session.state.models, ...entries] });
}

/** Logits on a served grid: +4 inside two balls (radius 8, centred at
 * (20, 20, 20) and (40, 40, 40)), -2 elsewhere, every channel alike. Not
 * -4: the mutex watershed takes its edges by |affinity - 0.5|, and expit(4)
 * and expit(-4) are as far from it, a tie it may break either way. */
function twoBalls(_view: string, _c: number, z: number, y: number, x: number): number {
  const inside = (cz: number, cy: number, cx: number) => (z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2 <= 64;
  return inside(20, 20, 20) || inside(40, 40, 40) ? 4 : -2;
}

/** A context with setup_54 (affinities) and setup_56 (nuc) served, a volume
 * created for `model` with its paint layer in the viewer, the viewer at
 * CENTRE and the model's output two balls. */
async function ready(model = NAME_54) {
  const ctx = fakeContext();
  await serve(ctx, entry("setup_54"), entry("setup_56", "nuc", ["nuc"], 128));
  const h = harness(ctx);
  const created = body(await h.handle(request("POST", "/api/finetune/create-volume", { model_name: model })));
  expect(created.success).toBe(true);
  const id = created.volume_id as string;
  await h.handle(request("POST", "/api/finetune/add-to-viewer", { crop_id: id, minio_url: created.neuroglancer_url }));
  ctx.viewer.positionNm = [...CENTRE];
  ctx.infer.outputAt = twoBalls;
  const volume = (await readVolume(ctx.paint, id)) as PaintVolume;
  return { ctx, h, id, volume };
}

/** Paint one voxel `label` in the store, as a brush stroke would. */
async function paintVoxel(ctx: FakeContext, volume: PaintVolume, voxel: number[], label: number): Promise<void> {
  await writeBox(ctx.paint, volume, voxel, voxel.map((v) => v + 1), volume.dtype === "uint16" ? Uint16Array.of(label) : Uint8Array.of(label));
}

async function labelAt(ctx: FakeContext, volume: PaintVolume, voxel: number[]): Promise<number> {
  return (await readBox(ctx.paint, volume, voxel, voxel.map((v) => v + 1)))[0];
}

/** How many voxels of the volume's first chunk hold each label. */
async function histogram(ctx: FakeContext, volume: PaintVolume, lo = [0, 0, 0], hi = [60, 60, 60]): Promise<Map<number, number>> {
  const counts = new Map<number, number>();
  for (const v of await readBox(ctx.paint, volume, lo, hi)) counts.set(v, (counts.get(v) ?? 0) + 1);
  return counts;
}

// ---------------------------------------------------------------------------

describe("GET /api/finetune/view-labels/sources", () => {
  it("lists the served models with the methods each offers, and no default while two are served and no volume says", async () => {
    const ctx = fakeContext();
    await serve(ctx, entry("setup_54"), entry("setup_56", "nuc", ["nuc"], 128));
    const res = body(await harness(ctx).handle(request("GET", `${BASE}/sources`)));
    expect(res).toEqual({
      success: true,
      models: [NAME_54, NAME_56],
      default: null,
      methods: { [NAME_54]: ["mutex_watershed", "components"], [NAME_56]: ["components"] },
      can_undo: false,
      box_voxels: null,
    });
  });
});

describe.skipIf(!python)("POST /api/finetune/view-labels/seed", () => {
  it("fills the patch from the prediction: an id per object counting up on a uint16 volume, background 1, the layer re-read", async () => {
    const { ctx, h, volume, id } = await ready();
    const res = body(await h.handle(request("POST", `${BASE}/seed`, {})));
    expect(res).toMatchObject({
      success: true, offset_voxels: [0, 0, 0], shape_voxels: [60, 60, 60], reload_viewer: true, layer_refreshed: true, can_undo: true,
      model: NAME_54, threshold: null, method: "components",
    });
    // The model's own output under the patch: the served grid is the volume's.
    expect(ctx.infer.outputReads).toEqual([{ view: ctx.infer.served()[0].view, lo: [0, 0, 0], hi: [60, 60, 60] }]);
    const counts = await histogram(ctx, volume);
    expect([...counts.keys()].sort((a, b) => a - b)).toEqual([1, 2, 3]);
    expect(res.filled_foreground).toBe(counts.get(2)! + counts.get(3)!);
    expect(res.filled_background).toBe(counts.get(1));
    expect(await labelAt(ctx, volume, [20, 20, 20])).toBe(2);
    expect(await labelAt(ctx, volume, [40, 40, 40])).toBe(3);
    expect(ctx.viewer.refreshed).toEqual([paintLayerName(id)]);
  });

  it("keeps what the user painted, and gives an object the id painted on it most", async () => {
    const { ctx, h, volume } = await ready();
    await paintVoxel(ctx, volume, [20, 20, 20], 7);
    await paintVoxel(ctx, volume, [5, 5, 5], 9);  // painted where the model sees background
    const res = body(await h.handle(request("POST", `${BASE}/seed`, {})));
    expect(res.success).toBe(true);
    expect(await labelAt(ctx, volume, [20, 20, 21])).toBe(7);
    expect(await labelAt(ctx, volume, [5, 5, 5])).toBe(9);
    // The other ball counts up from the box's largest id.
    expect(await labelAt(ctx, volume, [40, 40, 40])).toBe(10);
  });

  it("reuses free ids on a uint8 volume, and seeds by the mutex watershed for an affinity model", async () => {
    const { ctx, h, volume } = await ready(NAME_56);
    expect(volume.dtype).toBe("uint8");
    const seeded = body(await h.handle(request("POST", `${BASE}/seed`, { model: NAME_54, method: "mutex_watershed" })));
    expect(seeded).toMatchObject({ success: true, model: NAME_54, method: "mutex_watershed" });
    const ids = [...(await histogram(ctx, volume, seeded.offset_voxels, seeded.offset_voxels.map((o: number, a: number) => o + seeded.shape_voxels[a]))).keys()].sort((a, b) => a - b);
    // One object per ball, the lowest free ids (no count up on uint8), background 1.
    expect(ids).toEqual([1, 2, 3]);
    expect([await labelAt(ctx, volume, [20, 20, 20]), await labelAt(ctx, volume, [40, 40, 40])]).toEqual([2, 3]);
  });

  it("refuses what the server refuses, with its statuses, and writes nothing", async () => {
    const { ctx, h, volume } = await ready(NAME_56);
    const seed = async (b: Record<string, unknown>) => h.handle(request("POST", `${BASE}/seed`, b));
    for (const [b, status] of [
      [{ threshold: 1.5 }, 400],
      [{ min_size: -2 }, 400],
      [{ method: "watershed" }, 400],
      [{ connectivity: 4 }, 400],
      [{ connectivity: null }, 400],  // as the server reads it: given, and not 1, 2 or 3
      [{ box_voxels: [1, 2] }, 400],
      [{ model: "nope" }, 409],
      [{ method: "mutex_watershed" }, 400],  // setup_56 offers components only
      [{ select_channel: 3 }, 400],
    ] as [Record<string, unknown>, number][]) {
      const res = await seed(b);
      expect([res.status, body(res).success], JSON.stringify(b)).toEqual([status, false]);
    }
    ctx.viewer.positionNm = null;
    expect((await seed({})).status).toBe(400);
    ctx.viewer.positionNm = [-5000, -5000, -5000];  // outside the volume
    expect((await seed({})).status).toBe(400);
    // Nothing was written by any of them.
    expect((await histogram(ctx, volume)).get(0)).toBe(60 ** 3);
  });

  it("asks before labelling a large box, and labels it once confirmed", async () => {
    const { ctx, h } = await ready();
    ctx.viewer.positionNm = [...MIDDLE];
    const side = Math.ceil(Math.cbrt(LARGE_BOX_VOXELS)) + 1;
    const asked = await h.handle(request("POST", `${BASE}/seed`, { box_voxels: side }));
    expect(asked.status).toBe(409);
    expect(body(asked)).toMatchObject({ success: false, needs_confirmation: true, voxels: side ** 3 });
    const confirmed = body(await h.handle(request("POST", `${BASE}/seed`, { box_voxels: side, confirm: true })));
    expect(confirmed.success).toBe(true);
  });

  it("refuses without a volume, offering a new one", async () => {
    const ctx = fakeContext();
    await serve(ctx, entry("setup_54"));
    ctx.viewer.positionNm = [...CENTRE];
    const none = await harness(ctx).handle(request("POST", `${BASE}/seed`, {}));
    expect(none.status).toBe(409);
    expect(body(none)).toMatchObject({ success: false, needs_volume: true });
  });
});

describe.skipIf(!python)("POST /api/finetune/view-labels/background", () => {
  it("labels the unpainted voxels of the patch background, keeps the painted ones, and says when nothing is left", async () => {
    const { ctx, h, volume } = await ready();
    await paintVoxel(ctx, volume, [3, 3, 3], 5);
    const res = body(await h.handle(request("POST", `${BASE}/background`, { box_voxels: [4, 6, 8] })));
    expect(res).toEqual({
      success: true, offset_voxels: [28, 27, 26], shape_voxels: [4, 6, 8], filled_foreground: 0, filled_background: 4 * 6 * 8,
      reload_viewer: true, layer_refreshed: true, can_undo: true,
    });
    const whole = body(await h.handle(request("POST", `${BASE}/background`, {})));
    expect(whole.filled_background).toBe(60 ** 3 - 4 * 6 * 8 - 1);
    expect(await labelAt(ctx, volume, [3, 3, 3])).toBe(5);
    const again = body(await h.handle(request("POST", `${BASE}/background`, {})));
    expect(again).toMatchObject({ success: true, filled_background: 0, reload_viewer: false, layer_refreshed: false });
  });
});

describe.skipIf(!python)("POST /api/finetune/view-labels/split", () => {
  it("gives the two sides of a wall their own ids, and merges a stroke's objects, per slice when asked", async () => {
    const { ctx, h, volume } = await ready();
    const object = new Uint16Array(4 * 4 * 9).fill(5);
    for (let z = 0; z < 4; z++) for (let y = 0; y < 4; y++) object[(z * 4 + y) * 9 + 4] = 1;  // a wall at x = 4 through every slice
    await writeBox(ctx.paint, volume, [10, 10, 10], [14, 14, 19], object);
    const res = body(await h.handle(request("POST", `${BASE}/split`, {})));
    expect(res).toEqual({
      success: true, offset_voxels: [0, 0, 0], shape_voxels: [60, 60, 60], changed: 4 * 4 * 4, reload_viewer: true, layer_refreshed: true,
      can_undo: true, objects: 2, split: 1, merged: 0,
    });
    expect(await labelAt(ctx, volume, [10, 10, 10])).toBe(5);
    expect(await labelAt(ctx, volume, [10, 10, 18])).toBe(2);
    // Nothing more to split: the answer says so, and nothing is pushed on the undo stack.
    const again = body(await h.handle(request("POST", `${BASE}/split`, {})));
    expect(again).toMatchObject({ changed: 0, reload_viewer: false, objects: 2, split: 0, merged: 0 });
    // Per slice: each slice's two halves are objects of their own.
    const perSlice = body(await h.handle(request("POST", `${BASE}/split`, { per_slice: true })));
    expect(perSlice).toMatchObject({ objects: 8, split: 6, merged: 0 });
  });
});

describe.skipIf(!python)("POST /api/finetune/view-labels/undo", () => {
  it("takes back the label actions last first, keeping a stroke painted since, until there is nothing left", async () => {
    const { ctx, h, volume } = await ready();
    const background = body(await h.handle(request("POST", `${BASE}/background`, { box_voxels: 4 })));
    expect([background.filled_background, background.can_undo]).toEqual([64, true]);
    // With no dataset open the Python cannot run: refused, and the entry kept.
    const closed = vi.spyOn(ctx.infer, "dataset").mockReturnValue(null);
    expect((await h.handle(request("POST", `${BASE}/undo`, {}))).status).toBe(409);
    closed.mockRestore();
    const seeded = body(await h.handle(request("POST", `${BASE}/seed`, {})));
    expect(seeded.filled_foreground + seeded.filled_background).toBe(60 ** 3 - 64);
    await paintVoxel(ctx, volume, [20, 20, 20], 9);  // a stroke over the seed
    const first = body(await h.handle(request("POST", `${BASE}/undo`, {})));
    expect(first).toEqual({ success: true, restored: 60 ** 3 - 64 - 1, reload_viewer: true, layer_refreshed: true, can_undo: true });
    expect(body(await h.handle(request("GET", `${BASE}/sources`))).can_undo).toBe(true);
    const second = body(await h.handle(request("POST", `${BASE}/undo`, {})));
    expect(second).toEqual({ success: true, restored: 64, reload_viewer: true, layer_refreshed: true, can_undo: false });
    const counts = await histogram(ctx, volume);
    expect([counts.get(0), counts.get(9)]).toEqual([60 ** 3 - 1, 1]);
    expect(body(await h.handle(request("GET", `${BASE}/sources`))).can_undo).toBe(false);
    const nothing = await h.handle(request("POST", `${BASE}/undo`, {}));
    expect(nothing.status).toBe(409);
  });
});
