// The dashboard's settings routes: the LSF server config and GPU queues
// (routes/models.py; there is no cluster in the browser, so these answer
// what the page needs to show nothing wrong), the inference jobs' logs
// (none: the browser's log is the console), the exported YAML config, and
// the finetune tab's user preferences (routes/finetune/annotation_core.py),
// kept in the session.
import type { AppContext, ModelEntry, Step } from "../../contracts";
import { isFinetuned } from "../../models";
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
 * models as entries `cellmap_flow yaml` can run (serverEntry): the server's
 * onnx type serves the same ONNX files with ONNX Runtime. The header says
 * what else a server needs, and a finetuned model's entry, on its path line,
 * where its file goes. */
export function exportYaml(ctx: AppContext): string {
  const { datasetUrl, chain, models, resample } = ctx.session.state;
  const hasChain = chain.input_norm.length > 0 || chain.postprocess.length > 0;
  const config: Record<string, YamlValue> = { data_path: datasetUrl ?? "", resample: Boolean(resample) };
  if (hasChain) config.json_data = { input_norm: chain.input_norm.map(stepValue), postprocess: chain.postprocess.map(stepValue) };
  // The session keeps each entry as the catalog gave it (its default
  // variant); the one this page runs is the served entry's, tuned for its
  // GPU (serving.ts), which is the file a server should run to see the same.
  const served = ctx.infer.served();
  config.models = models.map((m) => serverEntry(served.find((s) => s.entry.id === m.id)?.entry ?? m));
  let header =
    "# CellMap-Flow exported configuration\n" +
    "# Reload with: cellmap_flow yaml <this file>\n" +
    "#\n" +
    "# Exported from the browser mode. Each model is an ONNX export, which the\n" +
    "# server serves with ONNX Runtime (type: onnx, in pixi.toml's onnx\n" +
    "# environment): a Hugging Face model by its repo, folder and file, the\n" +
    "# variant this browser ran; a model finetuned here by the .onnx that the\n" +
    "# Finetune tab's Download finetuned model saves, which carries its\n" +
    "# manifest. A relative path is taken from this file's folder. On the\n" +
    "# cluster, add charge_group: and queue:.\n" +
    "#\n";
  if (!hasChain) {
    header +=
      "# WARNING: no normalization/postprocessing was active when this\n" +
      "# was exported -- add a json_data block manually if your model(s)\n" +
      "# need one (e.g. logit outputs need a SigmoidPostprocessor).\n" +
      "#\n";
  }
  let yaml = toYaml(config);
  // The emitter writes no comments, so the finetuned models' path lines get
  // theirs here: the line is the emitter's own rendering of the entry's path.
  for (const m of models) {
    if (!isFinetuned(m)) continue;
    const line = `  ${toYaml({ path: m.variant.file }).trimEnd()}\n`;
    yaml = yaml.replace(line, `${line.trimEnd()}  # the file Download finetuned model saves: put it beside this YAML\n`);
  }
  return header + yaml;
}

/** A served model as the server's model entry, `type: onnx`
 * (cellmap_flow/models/configs/onnx.py), with serverParams, and the voxel
 * size it is run at when that is not its own (`run_voxel_size`, which the
 * server's entries take for any model), so the server runs it there too. */
export function serverEntry(m: ModelEntry): Record<string, YamlValue> {
  return { type: "onnx", name: m.name, ...serverParams(m, true), ...(m.runVoxelSize ? { run_voxel_size: m.runVoxelSize } : {}) };
}

/** The server's onnx type's arguments for a model: a Hugging Face export by
 * repo and folder, and with `withFile` its variant's file (for a served
 * entry, the variant chosen for this GPU), so the server runs the same one;
 * a model finetuned here by `path`, its variant's file (`<name>.onnx`,
 * which its download saves, routes/finetune.ts), beside the YAML. */
export function serverParams(m: ModelEntry, withFile: boolean): Record<string, YamlValue> {
  if (isFinetuned(m)) return { path: m.variant.file };
  return { repo: m.repo, folder: m.folder, ...(withFile ? { file: m.variant.file } : {}) };
}

function stepValue(step: Step): YamlValue {
  const out: Record<string, YamlValue> = {};
  for (const [k, v] of Object.entries(step)) if (v !== undefined) out[k] = v;
  return out;
}
