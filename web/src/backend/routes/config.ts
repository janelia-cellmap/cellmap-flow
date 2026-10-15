// The dashboard's settings routes: the LSF server config and GPU queues
// (routes/models.py; there is no cluster in the browser, so these answer
// what the page needs to show nothing wrong), the inference jobs' logs
// (none: the browser's log is the console), the exported YAML config, and
// the finetune tab's user preferences (routes/finetune/annotation_core.py),
// kept in the session.
import type { AppContext, Step } from "../../contracts";
import type { Router } from "../router";
import { badRequest, json, objectBody, text } from "../router";
import { toYaml, type YamlValue } from "../yaml";

/** The settings keys of jobs/settings.py's SERVER_CONFIG_DEFAULTS that the
 * page sends; kept so the Server Config form reads back what it saved. */
const SERVER_CONFIG_KEYS = ["queue", "charge_group", "walltime", "cycle_gpu_queues", "nb_cores_master", "nb_cores_worker", "nb_workers"];

function serverConfig(ctx: AppContext): Record<string, unknown> {
  const stored = ctx.session.state.prefs.serverConfig;
  return stored && typeof stored === "object" ? { ...(stored as Record<string, unknown>) } : {};
}

function userPrefs(ctx: AppContext): Record<string, unknown> {
  const stored = ctx.session.state.prefs.user;
  return stored && typeof stored === "object" ? { ...(stored as Record<string, unknown>) } : {};
}

export function configRoutes(router: Router): void {
  router.get("/api/job-logs", () => json(200, { success: true, jobs: [] }));

  router.get("/api/gpu-queues", () =>
    json(200, { queues: [], available: false, reason: "There is no cluster in the browser: the models run on this machine's GPU." }),
  );

  router.get("/api/server-config", (_req, _params, ctx) => json(200, { ...serverConfig(ctx), cached: true }));

  router.post("/api/server-config", async (req, _params, ctx) => {
    const body = objectBody(req);
    if (!body) return badRequest("expected a JSON object");
    const config = serverConfig(ctx);
    for (const key of SERVER_CONFIG_KEYS) if (key in body) config[key] = body[key];
    await ctx.session.update({ prefs: { ...ctx.session.state.prefs, serverConfig: config } });
    return json(200, { success: true, config });
  });

  router.get("/api/export-config", (_req, _params, ctx) => {
    const stamp = new Date().toISOString().replace(/[-:]/g, "").replace(/T(\d{6}).*$/, "_$1");
    return text(200, exportYaml(ctx), "text/yaml", { "content-disposition": `attachment; filename=cellmap_flow_config_${stamp}.yaml` });
  });

  router.get("/api/finetune/user-prefs", (_req, _params, ctx) => json(200, { success: true, prefs: userPrefs(ctx) }));

  router.post("/api/finetune/user-prefs", async (req, _params, ctx) => {
    const body = objectBody(req) ?? {};
    const prefs = userPrefs(ctx);
    for (const [key, value] of Object.entries(body)) if (value !== null && value !== undefined) prefs[key] = value;
    await ctx.session.update({ prefs: { ...ctx.session.state.prefs, user: prefs } });
    return json(200, { success: true, prefs });
  });
}

/** The exported configuration, as finetuned_model_templates
 * .generate_current_config_yaml writes it where there is no cluster: the
 * data path, resample, the chain as json_data (when there is one), and the
 * models as entries `cellmap_flow yaml` can rebuild. A browser model is a
 * Hugging Face repo's ONNX folder; the entry names it so, and the header
 * says what a server needs to run the same model. */
export function exportYaml(ctx: AppContext): string {
  const { datasetUrl, chain, models, resample } = ctx.session.state;
  const hasChain = chain.input_norm.length > 0 || chain.postprocess.length > 0;
  const config: Record<string, YamlValue> = { data_path: datasetUrl ?? "", resample: Boolean(resample) };
  if (hasChain) config.json_data = { input_norm: chain.input_norm.map(stepValue), postprocess: chain.postprocess.map(stepValue) };
  config.models = models.map((m) => ({
    type: "huggingface",
    repo: m.repo,
    name: m.name,
    folder: m.folder,
    onnx: m.variant.file,
  }));
  let header =
    "# CellMap-Flow exported configuration\n" +
    "# Reload with: cellmap_flow yaml <this file>\n" +
    "#\n" +
    "# Exported from the browser mode: its models are the ONNX exports in the\n" +
    "# Hugging Face repository named by each entry (folder, onnx), run on this\n" +
    "# machine's GPU. To run the same models on the cluster, point each entry\n" +
    "# at the PyTorch model (a repo with metadata.json, or a folder_path) and\n" +
    "# add charge_group: and queue:.\n" +
    "#\n";
  if (!hasChain) {
    header +=
      "# WARNING: no normalization/postprocessing was active when this\n" +
      "# was exported -- add a json_data block manually if your model(s)\n" +
      "# need one (e.g. logit outputs need a SigmoidPostprocessor).\n" +
      "#\n";
  }
  return header + toYaml(config);
}

function stepValue(step: Step): YamlValue {
  const out: Record<string, YamlValue> = {};
  for (const [k, v] of Object.entries(step)) if (v !== undefined) out[k] = v;
  return out;
}
