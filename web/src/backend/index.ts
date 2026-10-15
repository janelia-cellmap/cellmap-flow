// The /api/* backend: the dashboard's requests, answered in the page with the
// same JSON the Flask routes answer. createBackend(ctx) is what main.ts
// hands listenForApiRequests; it routes each ApiRequest (src/backend/router.ts)
// to the handler for its method and path:
//
//   routes/data.ts      POST /api/set-data
//   routes/pipeline.ts  PUT /api/pipeline
//   routes/models.ts    the Models tab: lists, Submit, add by reference, advice
//   routes/config.ts    server config and GPU queues (none here), job logs,
//                       the exported YAML, the finetune tab's preferences
//
// Every other /api/ route answers unavailable (router.ts): finetuning, the
// Review tab, the pipeline builder's routes and the dashboard's log stream
// need a server, or are still to come (PR 3). The template hides their
// controls in browser mode, so nothing on the page polls them.
//
// restoreSession(ctx) is the boot-time counterpart of Submit: after a reload
// the session still names the dataset and the models, but the Engine views
// behind their layers are gone with the page, so they are served again.
import type { ApiRequest, ApiResponse, AppContext, Backend } from "../contracts";
import { parseViewName } from "../shared/args";
import { message, Router, unavailable } from "./router";
import { configRoutes } from "./routes/config";
import { dataRoutes } from "./routes/data";
import { modelRoutes } from "./routes/models";
import { pipelineRoutes } from "./routes/pipeline";
import { serveModels } from "./serving";

/** The router with every route of the browser mode registered. */
export function createRouter(): Router {
  const router = new Router();
  dataRoutes(router);
  pipelineRoutes(router);
  modelRoutes(router);
  configRoutes(router);
  router.otherwise((req) =>
    unavailable(`${req.method} ${req.path} is not available in the browser: it needs a server (finetuning, the Review tab and the pipeline builder come later).`),
  );
  return router;
}

export function createBackend(ctx: AppContext): Backend {
  const router = createRouter();
  // The ids a layer's merging step joins across chunks go to that layer's
  // equivalences in the viewer (what the server posts to
  // /update/equivalences). A served view is named after its model, with
  // the chain's blob after it; the layer has the model's name.
  ctx.infer.onEquivalences((view, sets) => ctx.viewer.setEquivalences(parseViewName(view).name, sets));
  return {
    handle(req: ApiRequest): Promise<ApiResponse> {
      return router.handle(req, ctx);
    },
  };
}

/** Open the session's dataset in the engine and serve its models again,
 * through its chain, so the layers the viewer restores have their views
 * behind them. Call it before viewer.restore(): a layer whose view does not
 * exist yet fails its first request. Failures are logged, not thrown: the
 * dashboard must come up even when a dataset moved. */
export async function restoreSession(ctx: AppContext): Promise<void> {
  const { datasetUrl, models, chain } = ctx.session.state;
  if (!datasetUrl) return;
  try {
    await ctx.infer.openDataset(datasetUrl);
  } catch (e) {
    ctx.log(`Could not open ${datasetUrl} again: ${message(e)}`);
    return;
  }
  if (!models.length) return;
  try {
    await serveModels(ctx, models, chain);
    ctx.log(`Serving ${models.map((m) => m.name).join(", ")} again`);
  } catch (e) {
    ctx.log(`Could not serve the session's models again: ${message(e)}`);
  }
}
