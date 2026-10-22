// The good regions' routes against fake context members: the JSON each
// answers, checked against what good-region-marks.js reads, what each
// does to the volume's good_regions.json, and the good_regions layer it
// draws (good_regions.py's refresh_good_regions_layer: a local annotation
// layer of boxes in nm, the ids the regions' places in the list).
import { describe, expect, it } from "vitest";

import { Router } from "../../src/backend/router";
import { goodRegionRoutes } from "../../src/backend/routes/good-regions";
import type { ApiResponse, PaintVolume } from "../../src/contracts";
import { loadGoodRegions } from "../../src/finetune/good-regions";
import { createVolume } from "../../src/finetune/volume";
import { defaultChain } from "../../src/state";
import { GOOD_REGIONS_LAYER } from "../../src/viewer/layers";
import { DATASET, type FakeContext, fakeContext, request } from "./backend-fakes";
import { portablePython } from "./pyodide";

// A volume's plan and a region's rules are the portable Python's, here in
// node's Pyodide (pyodide.ts), loaded once for the file before the tests;
// without it (no network for numpy's wheel) they skip.
const python = await portablePython();

function body(res: ApiResponse): any {
  if (!("json" in res)) throw new Error(`not a JSON answer: ${JSON.stringify(res)}`);
  return res.json;
}

const VOLUME: PaintVolume = {
  id: "vol-00000001-20261015-120000", type: "annotation_volume", model_name: "m", model_id: "m", dataset_path: DATASET.url,
  chunk_size: [2, 4, 4], input_size: [10, 12, 12], output_voxel_size: [8, 8, 8], input_voxel_size: [8, 8, 8],
  dataset_offset_nm: [4, 4, 4], dataset_shape_voxels: [4, 8, 8], dtype: "uint8",
  input_norm: defaultChain().input_norm, postprocess: defaultChain().postprocess, created_at: "2026-10-15T12:00:00.000Z",
};

/** The routes on their own router, over a context whose session paints into VOLUME. */
async function harness() {
  const ctx: FakeContext = fakeContext();
  await createVolume(ctx.paint, VOLUME);
  await ctx.session.update({ paintVolume: VOLUME.id });
  const router = new Router();
  goodRegionRoutes(router);
  return { ctx, handle: (method: string, path: string, data?: unknown) => router.handle(request(method, path, data), ctx) };
}

/** The boxes the good_regions layer draws: [pointA, pointB, id, description]. */
function drawn(ctx: FakeContext): unknown[][] {
  const layer = ctx.viewer.layers[GOOD_REGIONS_LAYER] as { annotations: { pointA: number[]; pointB: number[]; id: string; description: string }[] };
  return layer.annotations.map((a) => [a.pointA, a.pointB, a.id, a.description]);
}

describe.skipIf(!python)("POST /api/finetune/good-regions/mark-view", () => {
  it("marks one output block centred where the viewer looks, keeps it with the volume, and draws it", async () => {
    const { ctx, handle } = await harness();
    ctx.viewer.positionNm = [16, 32, 32];
    const first = body(await handle("POST", "/api/finetune/good-regions/mark-view", {}));
    expect(first).toMatchObject({ success: true, count: 1, persisted: true, region: { label: "good-1", offset_nm: [8, 16, 16], shape_nm: [16, 32, 32] } });
    expect(first.region.id).toMatch(/^[0-9a-f]{8}$/);
    expect(await loadGoodRegions(ctx.paint, VOLUME.id)).toEqual([first.region]);
    expect(ctx.viewer.layers[GOOD_REGIONS_LAYER]).toEqual({
      type: "annotation",
      source: [{ url: "local://annotations", transform: { outputDimensions: { x: [1e-9, "m"], y: [1e-9, "m"], z: [1e-9, "m"] } } }],
      annotations: [{ type: "axis_aligned_bounding_box", pointA: [16, 16, 8], pointB: [48, 48, 24], id: "0", description: "good-1" }],
    });
    // A label and one size for every axis, as the route takes them.
    ctx.viewer.positionNm = [100, 100, 100];
    const second = body(await handle("POST", "/api/finetune/good-regions/mark-view", { label: "membranes", size_nm: 40 }));
    expect(second).toMatchObject({ count: 2, region: { label: "membranes", offset_nm: [80, 80, 80], shape_nm: [40, 40, 40] } });
    expect(drawn(ctx).map((a) => a[2])).toEqual(["0", "1"]);
  });
});

describe.skipIf(!python)("POST /api/finetune/good-regions/delete", () => {
  it("deletes one region by id, every region without one, and answers 404 for an id the session has not", async () => {
    const { ctx, handle } = await harness();
    ctx.viewer.positionNm = [16, 32, 32];
    const a = body(await handle("POST", "/api/finetune/good-regions/mark-view", {})).region;
    const b = body(await handle("POST", "/api/finetune/good-regions/mark-view", {})).region;
    expect(body(await handle("POST", "/api/finetune/good-regions/delete", { id: a.id }))).toEqual({ success: true, count: 1 });
    expect(await loadGoodRegions(ctx.paint, VOLUME.id)).toEqual([b]);
    expect(drawn(ctx)).toEqual([[[16, 16, 8], [48, 48, 24], "0", "good-2"]]);
    const missing = await handle("POST", "/api/finetune/good-regions/delete", { id: "nope" });
    expect(missing.status).toBe(404);
    expect(body(await handle("POST", "/api/finetune/good-regions/delete", {}))).toEqual({ success: true, count: 0 });
    expect(await loadGoodRegions(ctx.paint, VOLUME.id)).toEqual([]);
    expect(drawn(ctx)).toEqual([]);
  });
});
