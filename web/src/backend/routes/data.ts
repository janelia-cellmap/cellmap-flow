// POST /api/set-data (routes/index_page.py): open a dataset in the viewer.
//
// The body is {dataset_path}; in the browser that is a URL (https://, s3://
// or gs://, to an OME-Zarr group or array, or this page's own local/<key>/
// for a directory picked on this machine), since there is no file system to
// read a path from. The answer is {success: true, neuroglancer_url}, after
// which connect.js reloads the page, and the viewer is rebuilt from the
// session (viewer.restore). A body without a path is the 400 parse() gives;
// a URL that cannot be opened is a 500 {error}, as the Python route answers
// (connect.js shows either's error).
import type { Router } from "../router";
import { badRequest, json, message, objectBody } from "../router";
import { rawLayer } from "../../viewer/layers";

const DATASET_URL = /^(https?|s3|gs):\/\/\S+$/i;

export function dataRoutes(router: Router): void {
  router.post("/api/set-data", async (req, _params, ctx) => {
    const body = objectBody(req);
    if (!body) return badRequest("expected a JSON object");
    const url = typeof body.dataset_path === "string" ? body.dataset_path.trim() : "";
    if (!url) return badRequest("dataset_path is required");
    if (!DATASET_URL.test(url)) {
      return json(400, {
        success: false,
        error: `${url} is not a URL. In the browser the dataset is a URL to an OME-Zarr group or array (https://..., s3://bucket/path or gs://bucket/path, readable without credentials), or a local directory picked on this page.`,
      });
    }
    try {
      const info = await ctx.infer.openDataset(url);
      // The info too: the viewer takes its dimensions and starting position from the volume.
      const viewerUrl = await ctx.viewer.openDataset(url, rawLayer(url, info), info);
      await ctx.session.update({ datasetUrl: url });
      ctx.log(`Opened ${url} (${info.dtype}, ${info.levels.length} level${info.levels.length === 1 ? "" : "s"})`);
      return json(200, { success: true, neuroglancer_url: viewerUrl });
    } catch (e) {
      ctx.log(`Could not open ${url}: ${message(e)}`);
      return json(500, { error: message(e) });
    }
  });
}
