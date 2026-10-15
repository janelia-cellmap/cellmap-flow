// POST /api/set-data (routes/index_page.py): open a dataset in the viewer.
//
// The body is {dataset_path}; in the browser that is a URL (https://, s3://
// or gs://, to an OME-Zarr group or array, or this page's own local/<key>/
// for a directory picked on this machine), since there is no file system to
// read a path from. The answer is {success: true, neuroglancer_url}, after
// which connect.js reloads the page; the viewer's state for the volume is put
// in the page's URL now (viewer.prepareDataset) and shown by the reload
// (viewer.restore), not mounted before it, which showed the volume twice
// with the reload between. A body without a path is the 400 parse() gives;
// a URL that cannot be opened is a 500 {error}, as the Python route answers
// (connect.js shows either's error).
import type { Router } from "../router";
import { badRequest, json, message, objectBody } from "../router";
import { rawLayer } from "../../viewer/layers";

const DATASET_URL = /^(https?|s3|gs):\/\/\S+$/i;

/** The message of a fetch that got no answer at all, as each browser words
 * it: Chrome's "Failed to fetch", Firefox's "NetworkError when attempting to
 * fetch resource.", Safari's "Load failed". */
const NO_ANSWER = /Failed to fetch|NetworkError when attempting to fetch|Load failed/;

/** Why `url` could not be opened, for the connect panel. A fetch that got no
 * answer at all is explained, because the browser's own message says nothing
 * about why and there are three usual reasons from a page like this: the
 * host cannot be reached from here; its server does not allow this page to
 * read it (CORS); or, from the deployed site only, the host is on the user's
 * private network, which Chrome lets a public site use only with the user's
 * permission (Local Network Access): it fails, as "Failed to fetch", until
 * the user allows it. A page on localhost is not a public site, so the last
 * does not apply there. Any other error speaks for itself. `pageHost` is
 * the page's hostname (the browser's, when not given). */
export function explainOpenError(url: string, e: unknown, pageHost = typeof location === "undefined" ? "" : location.hostname): string {
  const why = message(e);
  if (!NO_ANSWER.test(why)) return why;
  let host = url;
  try {
    host = new URL(url).host || url;
  } catch {
    // Not a URL the browser parses: the text itself names the place.
  }
  const reasons = [`it cannot be reached from here`, `its server does not allow this page to read it (CORS)`];
  if (pageHost !== "localhost" && pageHost !== "127.0.0.1") {
    reasons.push(`it is on a private network (an address like 10.x), which Chrome may let a public site such as this one use only with your permission: `
      + `if Chrome asked, allow it; otherwise allow "Local network access" for this site from the icon beside the address bar, and connect again`);
  }
  return `${why}: the browser got no answer from ${host}. Either ${reasons.slice(0, -1).join(", ")}, or ${reasons[reasons.length - 1]}.`;
}

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
      const viewerUrl = ctx.viewer.prepareDataset(url, rawLayer(url, info), info);
      await ctx.session.update({ datasetUrl: url });
      ctx.log(`Opened ${url} (${info.dtype}, ${info.levels.length} level${info.levels.length === 1 ? "" : "s"})`);
      return json(200, { success: true, neuroglancer_url: viewerUrl });
    } catch (e) {
      const error = explainOpenError(url, e);
      ctx.log(`Could not open ${url}: ${error}`);
      return json(500, { error });
    }
  });
}
