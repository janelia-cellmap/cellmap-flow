// The /api/* backend: the dashboard's requests, answered in the page with the
// same JSON the Flask routes answer. createBackend(ctx) is what main.ts
// hands listenForApiRequests; it routes each ApiRequest (src/backend/router.ts)
// to the handler for its method and path:
//
//   routes/data.ts      POST /api/set-data
//   routes/pipeline.ts  PUT /api/pipeline
//   routes/models.ts    the Models tab: lists, Submit, add by reference, the
//                       loaded models and one of them stopped, advice
//   routes/config.ts    server config and GPU queues (none here), job logs,
//                       the exported YAML, the finetune tab's preferences
//   routes/finetune.ts  the Finetune tab: the models to annotate for, a paint
//                       volume and its layer, earlier volumes resumed, the
//                       training jobs (finetune-jobs.ts: the model's head
//                       trained in the tab) with their status and log stream,
//                       the volume and the finetuned model downloaded
//   routes/good-regions.ts  the Finetune tab's good regions: marked where
//                       the viewer looks, kept with the volume, drawn as the
//                       good_regions layer, and held by the head's training
//   routes/view-labels.ts  the Finetune tab's Label it tools: the patch on
//                       screen seeded from a served model's prediction,
//                       labelled all background, or its objects split, and
//                       the Undo of those and of an accepted AI annotation
//                       (one stack per volume, which the AI routes share)
//   routes/ai-annotate.ts  the Finetune tab's AI-assisted annotation: one
//                       plane of raw EM sent from the page to a Gemini image
//                       model (or the fake) with the user's own credentials,
//                       its answer staged for review (ai-annotate-job.ts),
//                       accepted into the paint volume
//
// Every other /api/ route answers unavailable (router.ts): the Review tab,
// the pipeline builder's routes, the dashboard's log stream, and the parts
// of the Finetune tab that read files the browser has not (crops import)
// need a server. The template hides their controls in browser mode, so
// nothing on the page polls them.
//
// restoreSession(ctx) is the boot-time counterpart of Submit: after a reload
// the session still names the dataset and the models, but the Engine views
// behind their layers are gone with the page, so they are served again.
import type { ApiRequest, ApiResponse, AppContext, Backend } from "../contracts";
import { parseViewName } from "../shared/args";
import { message, Router, unavailable } from "./router";
import { type AiAnnotateDeps, aiAnnotateRoutes } from "./routes/ai-annotate";
import { configRoutes } from "./routes/config";
import { dataRoutes } from "./routes/data";
import { type FinetuneDeps, finetuneRoutes } from "./routes/finetune";
import { goodRegionRoutes } from "./routes/good-regions";
import { modelRoutes } from "./routes/models";
import { pipelineRoutes } from "./routes/pipeline";
import { rankRoutes } from "./routes/rank";
import { viewLabelsRoutes } from "./routes/view-labels";
import { serveModels, stopServing } from "./serving";

export interface BackendOptions {
  /** The finetune routes' seams (a fake run, a fixed volume plan or clock), for tests. */
  finetune?: Partial<FinetuneDeps>;
  /** The AI-annotate routes' seams (a fake provider, a plain-JavaScript codec, a memory store, a clock), for tests. */
  aiAnnotate?: Partial<AiAnnotateDeps>;
}

/** The router with every route of the browser mode registered. The finetune
 * jobs live in it (one set per router, so per page), as the server's live
 * in its job manager; so do the patch tools' undo stacks and the
 * AI-annotate state (its settings, acknowledgements and job), as the
 * server's live in the session. The undo stacks are the Label it routes',
 * and the AI annotation's Accept pushes onto them, so the one Undo takes
 * back either (the server's ai_annotate.py uses view_labels._undo_stack). */
export function createRouter(options: BackendOptions = {}): Router {
  const router = new Router();
  dataRoutes(router);
  pipelineRoutes(router);
  modelRoutes(router);
  configRoutes(router);
  finetuneRoutes(router, options.finetune);
  goodRegionRoutes(router);
  rankRoutes(router);
  const labels = viewLabelsRoutes(router);
  aiAnnotateRoutes(router, { ...options.aiAnnotate, undo: labels.undo });
  router.otherwise((req) =>
    unavailable(
      `${req.method} ${req.path} is not available in the browser: it needs a server (the Review tab, the pipeline builder, and the finetune tab's crops import).`,
    ),
  );
  return router;
}

export function createBackend(ctx: AppContext, options: BackendOptions = {}): Backend {
  const router = createRouter(options);
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
    // A viewer that restore() left waiting for the volume's geometry is
    // mounted with it now, before the models' layers come in at their own
    // voxel sizes (ViewerController.datasetOpened).
    ctx.viewer.datasetOpened(datasetUrl, await ctx.infer.openDataset(datasetUrl));
  } catch (e) {
    ctx.viewer.datasetOpened(datasetUrl, null);
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
  // A model removed from the Loaded models list while this served it (the
  // list says "starting" until then) is not left served: it has left the
  // session's models since this read them. Only those: a model a Submit
  // serves meanwhile joins the session's models after it is served.
  const wanted = new Set(ctx.session.state.models.map((m) => m.id));
  for (const served of ctx.infer.served()) {
    if (wanted.has(served.entry.id) || !models.some((m) => m.id === served.entry.id)) continue;
    await stopServing(ctx, served.entry.name, [served]);
  }
}
