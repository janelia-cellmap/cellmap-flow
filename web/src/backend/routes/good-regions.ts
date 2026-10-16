// The Finetune tab's good regions (dashboard/routes/finetune/good_regions.py),
// answered in the page: the session's regions listed, a region marked where
// the viewer looks (Mark as Good), one deleted by id or all of them
// (clear), and the good_regions layer drawn after each change. A region is
// the user saying "the model gets this right, here": the browser's trainer
// then holds the head to what it predicted there, through the distillation
// term, wherever nothing is painted (finetune/run.ts, good-regions.ts).
//
// The session's regions are its volume's: the server keeps them in
// good_regions.json beside the session's corrections, and a browser session
// is one volume, so the file is at the volume's root in the paint store
// (finetune/good-regions.ts). Without a volume a mark is refused with
// needs_volume, as the server refuses one it has nowhere to save (the tab
// offers New Volume on it).
//
// The answers keep the Python routes' keys and statuses, so
// static/js/dashboard/finetune/good-region-marks.js reads them unchanged.
import type { AppContext, PaintVolume } from "../../contracts";
import { defaultRegionSizeNm, isGoodRegion, loadGoodRegions, regionAt, saveGoodRegions } from "../../finetune/good-regions";
import { readVolume } from "../../finetune/volume";
import { dimensionsOf, GOOD_REGIONS_LAYER, goodRegionsLayer } from "../../viewer/layers";
import type { Router } from "../router";
import { json, objectBody } from "../router";

const NO_VOLUME = "Could not work out where to save good regions for this session, so the mark was discarded. Create or resume an annotation volume first.";

export function goodRegionRoutes(router: Router): void {
  // GET /api/finetune/good-regions: the session volume's regions and their
  // count; none without a volume.
  router.get("/api/finetune/good-regions", async (_req, _params, ctx) => {
    const volume = await sessionVolume(ctx);
    const regions = volume ? await loadGoodRegions(ctx.paint, volume.id) : [];
    return json(200, { success: true, regions, count: regions.length });
  });

  // POST /api/finetune/good-regions/mark-view {size_nm?, label?}: a box
  // centred where the viewer looks (view_box_nm), one model output block by
  // default, labelled good-<n> unless a label is sent; saved, drawn, and
  // answered with the region and the count. Refused as the Python refuses:
  // no viewer position or a size that is not positive (400), then no
  // volume to keep it with (409, needs_volume).
  router.post("/api/finetune/good-regions/mark-view", async (req, _params, ctx) => {
    const body = objectBody(req) ?? {};
    const centre = ctx.viewer.position();
    if (!centre) return json(400, { success: false, error: "Viewer has no position" });
    const volume = await sessionVolume(ctx);
    const size = sizeNm(body.size_nm, volume);
    if (!size) return json(400, { success: false, error: "size_nm must be positive" });
    if (!volume) return json(409, { success: false, error: NO_VOLUME, needs_volume: true });
    const regions = await loadGoodRegions(ctx.paint, volume.id);
    const region = regionAt(centre, size, regions.length, typeof body.label === "string" ? body.label : undefined);
    regions.push(region);
    await saveGoodRegions(ctx.paint, volume.id, regions);
    showGoodRegions(ctx, regions);
    const round = (values: number[]) => `[${values.map((v) => Math.round(v)).join(", ")}]`;
    ctx.log(`Marked good region ${region.label} at ${round(centre)} nm, size ${round(size)} nm (${regions.length} total)`);
    return json(200, { success: true, region, count: regions.length, persisted: true });
  });

  // POST /api/finetune/good-regions/delete {id?}: the region `id`, or every
  // region without one (the tab's clear); 404 for an id the session has not.
  router.post("/api/finetune/good-regions/delete", async (req, _params, ctx) => {
    const id = (objectBody(req) ?? {}).id;
    const volume = await sessionVolume(ctx);
    const regions = volume ? await loadGoodRegions(ctx.paint, volume.id) : [];
    let kept: unknown[] = [];
    if (id !== undefined && id !== null) {
      kept = regions.filter((r) => !(r && typeof r === "object" && (r as { id?: unknown }).id === id));
      if (kept.length === regions.length) return json(404, { success: false, error: `No region ${String(id)}` });
    }
    if (volume) await saveGoodRegions(ctx.paint, volume.id, kept);
    showGoodRegions(ctx, kept);
    return json(200, { success: true, count: kept.length });
  });
}

/** The volume the session paints into, if any. */
async function sessionVolume(ctx: AppContext): Promise<PaintVolume | null> {
  const id = ctx.session.state.paintVolume;
  return id ? readVolume(ctx.paint, id) : null;
}

/** The size a mark asks for, in nm, (z, y, x): one number for every axis or
 * three; the volume's output block when none is sent (good_regions.py's
 * DEFAULT_REGION_SIZE_NM, 896 nm a side, without a volume); null when a
 * size is not a positive number. */
function sizeNm(value: unknown, volume: PaintVolume | null): number[] | null {
  const size = value === undefined || value === null ? (volume ? defaultRegionSizeNm(volume) : [896, 896, 896]) : typeof value === "number" ? [value, value, value] : value;
  if (!Array.isArray(size) || size.length !== 3 || !size.every((v) => typeof v === "number" && Number.isFinite(v) && v > 0)) return null;
  return size as number[];
}

/** Draw `regions` as the good_regions layer (refresh_good_regions_layer):
 * the viewer keeps the layer hidden if the user hid it, and shows it
 * otherwise, so a mark is seen to land. Its axes are the open dataset's,
 * the names the viewer's coordinate space has. */
function showGoodRegions(ctx: AppContext, regions: readonly unknown[]): void {
  const info = ctx.infer.dataset();
  const axes = info ? Object.keys(dimensionsOf(info)) : undefined;
  ctx.viewer.setLayers({ [GOOD_REGIONS_LAYER]: goodRegionsLayer(regions.map((r) => (isGoodRegion(r) ? r : null)), axes) });
}
