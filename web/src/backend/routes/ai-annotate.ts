// AI-assisted annotation in the browser: the routes of
// dashboard/routes/finetune/ai_annotate.py, answered in the page with the
// same JSON, so the tab's ai-annotate.js drives both. Accept pushes onto
// the patch tools' undo stacks (routes/view-labels.ts, whose Undo takes it
// back, as the server's accept uses view_labels._undo_stack). The user hovers a point in the viewer and
// presses Shift+G (or clicks Annotate at View Centre); the job
// (../ai-annotate-job.ts) reads one plane of raw EM around the point, in the
// plane the viewer shows, asks the chosen provider's model to paint the
// target structure in a colour, turns the image it returns into a mask and
// holds it for review. Nothing is written to the paint volume until Accept.
//
// What differs from the server, and why (PLAN_browser_finetune.md,
// "AI-assisted annotation in the browser"):
// - There is no config file. The feature is always on; the providers are
//   the page's (ai-annotate/providers.ts: the Gemini API, Vertex AI and a
//   fake), each with the destination text the user acknowledges.
// - Credentials are the user's own, typed into the panel: POST
//   .../credentials keeps them in the page (ai-annotate/credentials.ts),
//   in localStorage only when asked, and the config answer says only
//   `configured: true/false` per provider. They go to the provider's
//   request and nowhere else: never a log line, never an answer.
// - The call goes straight from the browser to Google; the day's cap is
//   this browser's (ai-annotate/usage.ts), and the config answer's
//   `usage_note` says so where the page would name the server's config.
// - The audit log is a file of the volume in the paint store
//   (ai-annotate/audit.ts); the acknowledgements, the settings and the job
//   live as long as the router (the page), as the server's live for the
//   dashboard's life, and so do the undo stacks, which are view-labels.ts's.
//
// The refusals keep the Python's order, statuses and words: the page shows
// them and keys on `needs_volume` and `needs_acknowledgement`.
import { createCredentialStore, type CredentialStore } from "../../ai-annotate/credentials";
import { canvasCodec } from "../../ai-annotate/images";
import { ORGANELLES, resolveOrganelleProfile } from "../../ai-annotate/organelles";
import { editablePrompt } from "../../ai-annotate/prompts";
import { createProviders, DEFAULT_PROVIDER } from "../../ai-annotate/providers";
import type { AiSettings, CredentialField, Credentials, DepthAxis, ImageCodec, OrganelleProfile, Provider, ProviderInfo, UndoEntry } from "../../ai-annotate/types";
import { ACTION, AIAnnotateError, CROP_SIZE_PX, DAILY_CALL_LIMIT, KEY, KEYBINDING, MAX_PROMPT_CHARS } from "../../ai-annotate/types";
import { createUsage, limitMessage, type Usage } from "../../ai-annotate/usage";
import type { AppContext, PaintVolume } from "../../contracts";
import { readVolume } from "../../finetune/volume";
import { defaultKeyValueStore, type KeyValueStore } from "../../models/listing-cache";
import { AiAnnotateJob, GENERIC_FAILURE, type JobDeps, newAnnotateId, Refused } from "../ai-annotate-job";
import type { Router } from "../router";
import { badRequest, json, message, objectBody } from "../router";

const BASE = "/api/finetune/ai-annotate";

/** The longest provider or model id, label and destination the routes take (requests.py's LABEL_MAX_CHARS and the destination's cap). */
const LABEL_MAX_CHARS = 100;
const DESTINATION_MAX_CHARS = 1000;
/** The longest credential string taken (a Vertex access token is about 1 KB). */
const CREDENTIAL_MAX_CHARS = 4096;
/** A model id as config.py's model regex has it (it goes into the request URL's path). */
const MODEL_ID = /^[A-Za-z0-9._@/-]{1,200}$/;
/** The longest model id: MODEL_ID's cap, which the panel's typed-id field (maxlength 200) has too. */
const MODEL_MAX_CHARS = 200;

/** How the panel names each credential field in a refusal. */
const FIELD_WORDS: Record<CredentialField, string> = { api_key: "API key", project: "Google Cloud project", access_token: "access token" };

/** The seams the routes run through: the providers and the image codec
 * (the page's canvas; a fake provider and a plain-JavaScript codec in
 * tests), the clock, where credentials and the day's count are kept
 * (localStorage; a MemoryKeyValueStore in tests; null for the page's life
 * only), the plane's size in annotation voxels, the day's cap, and the
 * undo stacks per volume that Accept pushes onto: the patch tools'
 * (view-labels.ts, whose Undo takes it back), which createRouter passes;
 * a map of its own when none is given. */
export interface AiAnnotateDeps {
  providers: Map<string, Provider>;
  codec: ImageCodec;
  now: () => Date;
  storage: KeyValueStore | null;
  cropSizePx: number;
  dailyCallLimit: number;
  undo: Map<string, UndoEntry[]>;
}

/** The page's AI-annotate state (the server's Session.ai_annotate), returned
 * for tests: the settings, the acknowledgements (dataset URL → provider id →
 * the destination acknowledged), the one job, the Shift+G unbind, the undo
 * stacks per volume (the patch tools'), the model lists providers have
 * listed live, the credentials and the day's count. */
export interface AiAnnotateState {
  settings: AiSettings | null;
  acknowledged: Map<string, Map<string, string>>;
  job: AiAnnotateJob | null;
  unbind: (() => void) | null;
  undo: Map<string, UndoEntry[]>;
  liveModels: Map<string, string[]>;
  credentials: CredentialStore;
  usage: Usage;
}

/** Register the routes; returns the state, which lives as long as the router (the page). */
export function aiAnnotateRoutes(router: Router, deps: Partial<AiAnnotateDeps> = {}): AiAnnotateState {
  const now = deps.now ?? (() => new Date());
  const storage = deps.storage !== undefined ? deps.storage : defaultKeyValueStore();
  const cropSizePx = deps.cropSizePx ?? CROP_SIZE_PX;
  const dailyCallLimit = deps.dailyCallLimit ?? DAILY_CALL_LIMIT;
  // The codec and the providers are made on first use, not at registration:
  // the canvas codec is the page's, and the router is also built where there
  // is no page (tests of the other routes).
  let codec: ImageCodec | null = deps.codec ?? null;
  let providers: Map<string, Provider> | null = deps.providers ?? null;
  const getCodec = (): ImageCodec => (codec ??= canvasCodec());
  const getProviders = (): Map<string, Provider> => (providers ??= createProviders({ codec: getCodec() }));

  const ai: AiAnnotateState = {
    settings: null,
    acknowledged: new Map(),
    job: null,
    unbind: null,
    undo: deps.undo ?? new Map(),
    liveModels: new Map(),
    credentials: createCredentialStore(storage),
    usage: createUsage(storage, now),
  };

  function jobDeps(ctx: AppContext): JobDeps {
    return { ctx, codec: getCodec(), usage: ai.usage, dailyCallLimit, cropSizePx, credentials: (provider) => ai.credentials.get(provider), now };
  }

  // ------------------------------------------------------------ helpers

  /** The provider `id` names, or a 400 Refused (config.py's provider(): the id came from the browser). */
  function provider(id: string): Provider {
    const found = getProviders().get(id);
    if (!found) throw new Refused("That AI provider is not available in the browser.", 400);
    return found;
  }

  /** The models the provider offers as the config answer lists them: its
   * defaults, then whatever its credentials have listed live. */
  function modelsOf(p: Provider): string[] {
    const live = ai.liveModels.get(p.info.id) ?? [];
    return [...p.info.models, ...live.filter((m) => !p.info.models.includes(m))];
  }

  /** config.check_model: refuse a model the provider does not offer. A
   * provider with a fixed list (the fake) offers only that; the Gemini
   * providers take any id shaped like one (config.py's model regex), since
   * the page's picker has an Other… entry for a model not in their default
   * list (Vertex lists none, so a typed id is the only way to one). */
  function checkModel(providerId: string, model: string): Provider {
    const p = provider(providerId);
    if (p.info.type === "fake") {
      if (!p.info.models.includes(model)) throw new Refused("That model is not listed for this provider.", 400);
    } else if (!MODEL_ID.test(model)) {
      throw new Refused("That is not a model id.", 400);
    }
    return p;
  }

  /** The provider as the page sees it: its info, with the models it offers now and whether its credentials are entered. */
  /** A provider as the page sees it: its info with the models it offers now
   * (modelsOf), and whether it can be called (`configured`): its credentials
   * are in the page. */
  function providerInfo(p: Provider): ProviderInfo & { configured: boolean } {
    return { ...p.info, models: modelsOf(p), configured: ai.credentials.configured(p.info.id, p.info.credentials) };
  }

  function defaultProvider(): string | null {
    const all = getProviders();
    if (all.has(DEFAULT_PROVIDER)) return DEFAULT_PROVIDER;
    const first = all.keys().next();
    return first.done ? null : first.value;
  }

  /** The session's annotation volume, or null. */
  async function sessionVolume(ctx: AppContext): Promise<PaintVolume | null> {
    const id = ctx.session.state.paintVolume;
    return id ? readVolume(ctx.paint, id) : null;
  }

  /** The dataset a run reads (_dataset): the annotation volume's, else the one the viewer shows. */
  function datasetOf(ctx: AppContext, volume: PaintVolume | null): string | null {
    if (volume?.dataset_path) return volume.dataset_path;
    return ctx.infer.dataset()?.url ?? ctx.session.state.datasetUrl ?? null;
  }

  /** The provider ids whose current destination has been acknowledged for
   * `dataset` (_acknowledged): an acknowledgement is of the destination
   * string the user was shown, so a provider that moved needs it again. */
  function acknowledgedFor(dataset: string | null): string[] {
    if (!dataset) return [];
    const seen = ai.acknowledged.get(dataset);
    if (!seen) return [];
    return [...getProviders().values()].filter((p) => seen.get(p.info.id) === p.info.destination).map((p) => p.info.id);
  }

  /** _check_egress: refuse sending `dataset`'s planes to the provider and
   * model unless all is allowed. The model first, so a provider that is not
   * here is reported as that rather than as a missing acknowledgement; then
   * the acknowledgement of the provider's current destination for this
   * dataset. (The server also checks its config's dataset prefixes; the
   * browser has none.) */
  function checkEgress(dataset: string | null, providerId: string, model: string): Provider {
    const p = checkModel(providerId, model);
    if (!acknowledgedFor(dataset).includes(providerId)) {
      throw new Refused("Acknowledge where the images are sent before the first run on this dataset.", 409, {
        needs_acknowledgement: true, provider: providerId, destination: p.info.destination,
      });
    }
    return p;
  }

  /** Refuse (400) a run for a provider whose credentials are not entered, saying which. */
  function checkCredentials(p: Provider): void {
    if (ai.credentials.configured(p.info.id, p.info.credentials)) return;
    const words = p.info.credentials.map((field) => FIELD_WORDS[field]);
    throw new Refused(`Paste the ${words.length ? joinWords(words) : "credentials"} for ${p.info.id} into the AI annotation panel first (they are kept in the page only, so a reload asks for them again).`, 400);
  }

  /** _check_limit: refuse (429) a run once the day's calls are used up. The
   * job counts the call just before it is made; this is so a refusal comes
   * as the answer to the click rather than as a failed job a moment later. */
  function checkLimit(): void {
    if (ai.usage.callsToday() >= dailyCallLimit) throw new AIAnnotateError("limit", limitMessage(dailyCallLimit));
  }

  function profileOf(settings: AiSettings): OrganelleProfile {
    return resolveOrganelleProfile(settings.label_key || settings.label_name || "");
  }

  /** `prompt` as the prompt's override: null when blank or the catalog's own
   * text (the page prefills the catalog's text, so an unedited prompt is
   * kept as null and the audit log shows no override). */
  function overrideOf(prompt: string | null | undefined, profile: OrganelleProfile): string | null {
    if (prompt === null || prompt === undefined) return null;
    const text = prompt.trim();
    if (!text || text === editablePrompt(profile).trim()) return null;
    return text;
  }

  /** _ensure_binding: bind Shift+G in the viewer, once per router. The
   * viewer binds it again in every viewer it mounts after. */
  function ensureBinding(ctx: AppContext): void {
    if (ai.unbind) return;
    ai.unbind = ctx.viewer.bindKey(KEY, ACTION, () => startFromKey(ctx));
  }

  /** The Shift+G action (_start_from_key): a job at the voxel under the
   * mouse. A refusal is shown by the status route (a failed job), unless a
   * job is under way or waiting for review: that one stays what the page
   * shows. */
  function startFromKey(ctx: AppContext): void {
    void (async () => {
      try {
        const pointer = ctx.viewer.pointerPosition();
        if (!pointer) throw new Refused("Hover the mouse over the data, then press Shift+G.", 400);
        await beginRun(ctx, pointer, ctx.viewer.viewDepthAxis());
      } catch (e) {
        let text: string;
        let needsVolume = false;
        if (e instanceof Refused) {
          text = e.message;
          needsVolume = Boolean(e.extra.needs_volume);
        } else if (e instanceof AIAnnotateError) {
          text = e.userMessage;
        } else {
          ctx.log(`AI annotation could not start: ${message(e)}`);
          text = GENERIC_FAILURE;
        }
        const job = ai.job;
        if (job === null || job.status === "failed") ai.job = new AiAnnotateJob(jobDeps(ctx), null, { error: text, needsVolume });
        // A job running or waiting for review stays what the panel shows, so
        // the refusal is said in the log; the server shows it in the
        // viewer's status bar (_say), which the browser's Neuroglancer has
        // no handle for.
        else ctx.log(`AI annotate: Shift+G refused: ${text}`);
      }
    })();
  }

  /** _begin_run: check everything a run needs, then start its job; returns
   * its annotate_id. The checks, in order: the session has settings, there
   * is an annotation volume, the model is offered, the provider's
   * destination is acknowledged for the volume's dataset, nothing is
   * running, waiting for review or being written, the day's limit is not
   * reached, and the provider's credentials are entered. */
  async function beginRun(ctx: AppContext, pointNm: number[], depthAxis: DepthAxis): Promise<string> {
    const settings = ai.settings;
    if (!settings) throw new Refused("Choose a provider, model and target for AI annotation first.", 400);
    const volume = await sessionVolume(ctx);
    if (!volume) throw new Refused("No annotation volume to annotate. Create or resume one first.", 409, { needs_volume: true });
    // The plane is read from the dataset the engine has open, and written on
    // the volume's grid: they must be the same dataset (the server reads the
    // volume's own, pipeline.read_plane; here the engine holds one at a time).
    const open = ctx.infer.dataset();
    if (open && open.url !== volume.dataset_path) {
      throw new Refused(`The annotation volume was painted on ${volume.dataset_path}, but the viewer shows ${open.url}: open that dataset again, or make a new volume on this one.`, 409);
    }
    const dataset = datasetOf(ctx, volume);
    const p = checkEgress(dataset, settings.provider, settings.model);
    ensureBinding(ctx);
    const profile = profileOf(settings);
    const override = overrideOf(settings.prompt, profile);
    const labelName = settings.label_name || profile.name;
    const current = ai.job;
    if (current && (current.status === "running" || current.status === "ready" || current.status === "accepting")) {
      throw new Refused({
        running: "An AI annotation is still running.",
        ready: "Accept or reject the AI annotation waiting for review first.",
        accepting: "The accepted AI annotation is still being written.",
      }[current.status]);
    }
    checkLimit();
    checkCredentials(p);
    const job = new AiAnnotateJob(jobDeps(ctx), {
      annotateId: newAnnotateId(), provider: p, providerId: settings.provider, model: settings.model, labelName,
      depthAxis, pointNm, volume, dataset, profile, override,
    });
    ai.job = job;
    ctx.log(`AI annotation ${job.annotateId}: ${labelName} in the ${["XY", "XZ", "YZ"][depthAxis]} plane at (${pointNm.map((v) => fmt(v)).join(", ")}) nm, ${settings.provider}/${settings.model}`);
    job.start();
    return job.annotateId!;
  }

  /** _job_for: the page's job when `annotateId` is its id and its status is one of `statuses`. */
  function jobFor(annotateId: string, ...statuses: AiAnnotateJob["status"][]): AiAnnotateJob {
    const job = ai.job;
    if (!job || job.annotateId !== annotateId) throw new Refused("That AI annotation is not the current one; it may have been accepted or rejected.");
    if (!statuses.includes(job.status)) {
      throw new Refused(
        {
          running: "The AI annotation is still running.",
          failed: "The AI annotation failed; resend it or reject it.",
          ready: "The AI annotation is waiting for review.",
          accepting: "The AI annotation is already being written.",
          idle: "The AI annotation cannot do that now.",
        }[job.status],
      );
    }
    return job;
  }

  /** The volume's undo stack, as view-labels.ts keeps it. */
  function undoStack(volumeId: string): UndoEntry[] {
    let stack = ai.undo.get(volumeId);
    if (!stack) ai.undo.set(volumeId, (stack = []));
    return stack;
  }

  /** The error answer for an exception from a route (_answer_error). */
  function answerError(ctx: AppContext, e: unknown) {
    if (e instanceof Refused) return json(e.status, { success: false, error: e.message, ...e.extra });
    if (e instanceof AIAnnotateError) {
      ctx.log(`AI annotation refused (${e.category}): ${e.userMessage}`);
      return json(e.httpStatus, { success: false, error: e.userMessage, category: e.category });
    }
    ctx.log(`AI annotation request failed: ${message(e)}`);
    return json(500, { success: false, error: GENERIC_FAILURE });
  }

  // ------------------------------------------------------------- routes

  // GET .../config: what the page needs to show the panel. Always enabled
  // here; `providers` carry `configured`, and `usage_note`
  // says where the day's cap is kept, which the page shows in the brackets
  // after the count in place of the server's "the daily_call_limit in
  // ai_annotate.yaml".
  router.get(`${BASE}/config`, async (_req, _params, ctx) => {
    try {
      const volume = await sessionVolume(ctx);
      const callsToday = ai.usage.callsToday();
      return json(200, {
        success: true,
        enabled: true,
        providers: [...getProviders().values()].map(providerInfo),
        default_provider: defaultProvider(),
        daily_call_limit: dailyCallLimit,
        calls_today: callsToday,
        crop_size_px: cropSizePx,
        organelles: Object.values(ORGANELLES).map((p) => ({ key: p.key, name: p.name, prompt: editablePrompt(p) })),
        keybinding: KEYBINDING,
        settings: ai.settings,
        acknowledged: acknowledgedFor(datasetOf(ctx, volume)),
        usage_note: `the browser's own cap of ${dailyCallLimit} a day`,
      });
    } catch (e) {
      return answerError(ctx, e);
    }
  });

  // POST .../credentials {provider, api_key?, project?, access_token?,
  // forget?}: the provider's credentials into the page's memory (the panel
  // posts them as they are pasted), or out of it with `forget`; never into
  // any storage (credentials.ts says why). The answer says only whether the
  // provider is configured, and lists the models the credentials can reach
  // when the provider can list them (the Gemini API); a listing that fails is
  // not an error: `warning` says why and the picker keeps its defaults.
  router.post(`${BASE}/credentials`, async (req, _params, ctx) => {
    const body = objectBody(req);
    if (!body) return badRequest("expected a JSON object");
    try {
      const id = requiredId(body.provider, "provider");
      const p = provider(id);
      if (body.forget === true) {
        ai.credentials.forget(id);
        ai.liveModels.delete(id);
        ctx.log(`AI annotate: the credentials for ${id} were forgotten`);
        // The models the forgotten credentials listed go with them: the
        // provider's defaults again, for the page's picker.
        return json(200, { success: true, configured: ai.credentials.configured(id, p.info.credentials), models: modelsOf(p) });
      }
      const credentials: Credentials = {};
      for (const field of ["api_key", "project", "access_token"] as const) {
        const value = optionalString(body[field], field, CREDENTIAL_MAX_CHARS);
        if (value) credentials[field] = value;
      }
      if (!p.info.credentials.some((field) => credentials[field])) {
        throw new Refused(`Enter the ${joinWords(p.info.credentials.map((f) => FIELD_WORDS[f]))} for ${id}.`, 400);
      }
      ai.credentials.set(id, credentials);
      ctx.log(`AI annotate: credentials for ${id} entered (kept in this page only)`);
      const answer: Record<string, unknown> = { success: true, configured: ai.credentials.configured(id, p.info.credentials) };
      if (p.listModels) {
        try {
          const models = await p.listModels(ai.credentials.get(id) ?? {}, { signal: AbortSignal.timeout(30_000) });
          if (models) {
            ai.liveModels.set(id, models);
            answer.models = models;
          }
        } catch (e) {
          answer.warning = `Could not list the models these credentials reach: ${e instanceof AIAnnotateError ? e.userMessage : message(e)}`;
        }
      }
      return json(200, answer);
    } catch (e) {
      return answerError(ctx, e);
    }
  });

  // POST .../settings {provider, model, label_key?, label_name?, prompt?,
  // acknowledge?, destination?}: the provider, model, target and prompt to
  // use; `acknowledge: true` records that the user accepts that the
  // provider's destination receives this dataset's images, `false`
  // withdraws that. Also binds Shift+G.
  router.post(`${BASE}/settings`, async (req, _params, ctx) => {
    const body = objectBody(req);
    if (!body) return badRequest("expected a JSON object");
    try {
      const providerId = requiredId(body.provider, "provider");
      const model = requiredId(body.model, "model", MODEL_MAX_CHARS);
      const labelKey = optionalString(body.label_key, "label_key", LABEL_MAX_CHARS, true);
      const labelName = optionalString(body.label_name, "label_name", LABEL_MAX_CHARS, true);
      const prompt = optionalString(body.prompt, "prompt", MAX_PROMPT_CHARS);
      const acknowledge = body.acknowledge === undefined || body.acknowledge === null ? null : body.acknowledge;
      if (acknowledge !== null && typeof acknowledge !== "boolean") throw new Refused("acknowledge: expected true or false", 400);
      const destination = optionalString(body.destination, "destination", DESTINATION_MAX_CHARS);
      const p = checkModel(providerId, model);
      if (labelKey && !(labelKey in ORGANELLES)) throw new Refused(`Unknown target ${JSON.stringify(labelKey)}; pick one from the list.`, 400);
      if (!labelKey && !labelName) throw new Refused("Name the structure to annotate (label_key or label_name).", 400);
      const settings: AiSettings = { provider: providerId, model, label_key: labelKey, label_name: labelName, prompt: null };
      settings.prompt = overrideOf(prompt, profileOf(settings));
      const volume = await sessionVolume(ctx);
      const dataset = datasetOf(ctx, volume);
      if (acknowledge === false && dataset) {
        // Unticked: the next run asks again before anything is sent.
        if (ai.acknowledged.get(dataset)?.delete(providerId)) ctx.log(`AI annotate: sending images of ${dataset} to ${providerId} no longer agreed to`);
      }
      if (acknowledge === true) {
        if (!dataset) throw new Refused("Open a dataset before acknowledging where its images are sent.");
        const current = p.info.destination;
        // The page sends the destination it showed: an acknowledgement is of
        // that, so a provider moved since the page loaded is refused rather
        // than taken as agreed to.
        if (destination !== current) {
          throw new Refused("Where this provider sends images has changed; check the new destination and tick again.", 409, {
            needs_acknowledgement: true, provider: providerId, destination: current,
          });
        }
        let seen = ai.acknowledged.get(dataset);
        if (!seen) ai.acknowledged.set(dataset, (seen = new Map()));
        seen.set(providerId, current);
        ctx.log(`AI annotate: images of ${dataset} may be sent to ${providerId} (${current})`);
      }
      ai.settings = settings;
      ensureBinding(ctx);
      return json(200, { success: true, settings, acknowledged: acknowledgedFor(dataset) });
    } catch (e) {
      return answerError(ctx, e);
    }
  });

  // POST .../run {point_nm?, depth_axis?}: start a job at point_nm (z, y, x,
  // world nm) in the plane normal to depth_axis. Both optional: the view
  // centre, and the plane of the viewer's layout.
  router.post(`${BASE}/run`, async (req, _params, ctx) => {
    const body = objectBody(req) ?? {};
    try {
      let pointNm: number[] | null = null;
      if (body.point_nm !== undefined && body.point_nm !== null) {
        if (!Array.isArray(body.point_nm) || body.point_nm.length !== 3 || !body.point_nm.every((v) => typeof v === "number" && Number.isFinite(v))) {
          throw new Refused("point_nm must be three finite numbers, z, y, x", 400);
        }
        pointNm = body.point_nm.map(Number);
      }
      let depthAxis: DepthAxis | null = null;
      if (body.depth_axis !== undefined && body.depth_axis !== null) {
        if (body.depth_axis !== 0 && body.depth_axis !== 1 && body.depth_axis !== 2) throw new Refused("depth_axis must be 0, 1 or 2 (z, y, x)", 400);
        depthAxis = body.depth_axis;
      }
      if (pointNm === null || depthAxis === null) {
        if (!ctx.infer.dataset()) throw new Refused("Open a dataset in the viewer first.", 400);
        if (depthAxis === null) depthAxis = ctx.viewer.viewDepthAxis();
        if (pointNm === null) {
          pointNm = ctx.viewer.position();
          if (!pointNm) throw new Refused("The viewer has no position.", 400);
        }
      }
      const annotateId = await beginRun(ctx, pointNm, depthAxis);
      return json(200, { success: true, annotate_id: annotateId, status: "running" });
    } catch (e) {
      return answerError(ctx, e);
    }
  });

  // GET .../status[?preview=1]: the job's status and stage, its error, or
  // its preview once ready. The preview is three PNGs of up to 1024 pixels
  // a side, so it comes only when asked for: the page asks once when a
  // result becomes ready, not on every poll.
  router.get(`${BASE}/status`, (req) => {
    const withPreview = ["1", "true"].includes(req.query.get("preview") ?? "");
    if (!ai.job) return json(200, idleStatus());
    return json(200, ai.job.statusAnswer(withPreview));
  });

  // POST .../resend {annotate_id, prompt}: ask the model again about the
  // staged plane, with the edited prompt. The plane is not read again: the
  // same image goes with the new prompt, and the preview is replaced under
  // the same annotate_id.
  router.post(`${BASE}/resend`, async (req, _params, ctx) => {
    const body = objectBody(req);
    if (!body) return badRequest("expected a JSON object");
    try {
      const annotateId = requiredAnnotateId(body.annotate_id);
      const prompt = optionalString(body.prompt, "prompt", MAX_PROMPT_CHARS);
      const job = jobFor(annotateId, "ready", "failed");
      if (!job.hasPlane()) throw new Refused("That AI annotation never read its plane; run it again instead.");
      const args = job.args!;
      const p = checkEgress(args.dataset, args.providerId, args.model);
      checkLimit();
      checkCredentials(p);
      job.resend(overrideOf(prompt, args.profile));
      ctx.log(`AI annotation ${annotateId}: the plane sent again${job.override ? " with an edited prompt" : ""}`);
      return json(200, { success: true, annotate_id: annotateId, status: "running" });
    } catch (e) {
      return answerError(ctx, e);
    }
  });

  // POST .../accept {annotate_id, overwrite}: write the staged mask into the
  // annotation volume. Marked `accepting` while it is written, so a second
  // click, a Run or a Shift+G is refused rather than replacing it; set back
  // to ready if the write fails, for the user to try again or reject it.
  router.post(`${BASE}/accept`, async (req, _params, ctx) => {
    const body = objectBody(req);
    if (!body) return badRequest("expected a JSON object");
    let job: AiAnnotateJob;
    let overwrite: boolean;
    try {
      const annotateId = requiredAnnotateId(body.annotate_id);
      overwrite = body.overwrite === true;
      job = jobFor(annotateId, "ready");
      job.status = "accepting";
    } catch (e) {
      return answerError(ctx, e);
    }
    const volumeId = job.args!.volume.id;
    const undo = undoStack(volumeId);
    let result;
    try {
      result = await job.accept(overwrite, undo);
    } catch (e) {
      job.status = "ready";
      if (e instanceof Refused || e instanceof AIAnnotateError) return answerError(ctx, e);
      // Our own messages (labelsForBox, fill's ids running out), which say
      // what to do: the seed route answers them the same way.
      ctx.log(`Could not accept AI annotation ${job.annotateId}: ${message(e)}`);
      return json(409, { success: false, error: message(e) });
    }
    if (ai.job === job) ai.job = null;
    const { lo, hi } = job.writeBox!;
    ctx.log(
      `Accepted AI annotation ${job.annotateId} into ${lo.join(",")}..${hi.join(",")} of ${volumeId}: ` +
        `${result.filled_foreground} foreground, ${result.filled_background} background, ${result.overwritten} overwritten`,
    );
    await job.audit("accepted", {
      annotate_id: job.annotateId, volume_id: volumeId, overwrite, filled_foreground: result.filled_foreground,
      filled_background: result.filled_background, overwritten: result.overwritten, write_box: job.writeBox,
    });
    return json(200, { success: true, ...result, can_undo: undo.length > 0 });
  });

  // POST .../reject {annotate_id}: throw the staged mask away.
  router.post(`${BASE}/reject`, async (req, _params, ctx) => {
    const body = objectBody(req);
    if (!body) return badRequest("expected a JSON object");
    try {
      const annotateId = requiredAnnotateId(body.annotate_id);
      const job = jobFor(annotateId, "ready", "failed");
      ai.job = null;
      await job.audit("rejected", { annotate_id: annotateId, volume_id: job.args!.volume.id });
      ctx.log(`Rejected AI annotation ${annotateId}`);
      return json(200, { success: true });
    } catch (e) {
      return answerError(ctx, e);
    }
  });

  return ai;
}

/** What the status route answers with no job (_idle_job, with stage_label and plane). */
function idleStatus() {
  return {
    success: true, annotate_id: null, status: "idle", stage: null, stage_label: null, error: null, prompt: null, provider: null, model: null,
    label_name: null, depth_axis: null, plane: null, point_nm: null, write_box: null, mask_fraction: null, preview: null, needs_volume: false,
  };
}

// ---------------------------------------------------------------------------
// The body's fields, checked as dashboard/requests.py checks them
// ---------------------------------------------------------------------------

/** A provider or model id: required, trimmed, at most `max` characters (_Id with _required). */
function requiredId(value: unknown, name: string, max = LABEL_MAX_CHARS): string {
  if (value === undefined || value === null) throw new Refused(`${name} is required`, 400);
  if (typeof value !== "string") throw new Refused(`${name}: expected a string`, 400);
  const text = value.trim();
  if (!text) throw new Refused(`${name} is required`, 400);
  if (text.length > max) throw new Refused(`${name}: at most ${max} characters`, 400);
  return text;
}

/** An optional string of at most `max` characters; with `blankIsNone`, a blank one is null (_Label). */
function optionalString(value: unknown, name: string, max: number, blankIsNone = false): string | null {
  if (value === undefined || value === null) return null;
  if (typeof value !== "string") throw new Refused(`${name}: expected a string`, 400);
  if (value.length > max) throw new Refused(`${name}: at most ${max} characters`, 400);
  if (blankIsNone && !value.trim()) return null;
  return value;
}

/** The staged annotation's id (_ANNOTATE_ID): required, 32 lowercase hex characters. */
function requiredAnnotateId(value: unknown): string {
  if (value === undefined || value === null || value === "") throw new Refused("annotate_id is required", 400);
  if (typeof value !== "string" || !/^[0-9a-f]{32}$/.test(value)) throw new Refused("annotate_id: expected 32 hex characters", 400);
  return value;
}

function joinWords(words: string[]): string {
  if (words.length <= 1) return words.join("");
  return `${words.slice(0, -1).join(", ")} and ${words[words.length - 1]}`;
}

/** A coordinate for a log line: up to three decimals, no trailing zeros. */
function fmt(v: number): string {
  return Number.isInteger(v) ? String(v) : v.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
}
