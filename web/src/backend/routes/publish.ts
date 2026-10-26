// Publishing a model finetuned in this browser to Hugging Face, under the
// user's own account (src/publish): the Training Status card's Publish to
// Hugging Face panel (static/js/dashboard/finetune/publish.js), beside
// Download finetuned model.
//
//   GET  /api/huggingface/credentials       who is signed in, how, and the Sign in popup's URL
//   POST /api/huggingface/credentials       {token}: a pasted access token; {forget: true}: sign out
//   POST /api/finetune/model/:name/publish  {repo, private, include_dataset}: start a publish
//   GET  /api/finetune/model/:name/publish  how it is going, or where the model was published
//
// A publish is a job, started by the POST and followed by the GET, as a
// finetune is: an upload of tens of megabytes can outlast the service
// worker's wait for an answer (sw.ts API_TIMEOUT_MS), and the server, when
// it answers these routes too, will run one as a cluster job (merge the
// LoRA, export the ONNX, upload). The answers are the same in both: a
// status ("none", "running", "done", "failed"), the step while running, the
// error once failed, and once done where the model is: the repository and
// folder, its ref (`<owner>/<repo>/<folder>`, which the browser's and the
// server's *Add a model* take), its Hugging Face page, and the two links
// that open it in the app, the model alone ("Try it in the browser") and
// with the dataset open now ("Open on this dataset"). A model published
// before (FinetunedRecord.published) answers done from its record, so the
// panel shows where it is after a reload; publishing it again, to the same
// repository or another, is a new commit there.
//
// The token is the user's, kept in this page's memory only, for the reasons
// ai-annotate/credentials.ts gives for an API key: never in any storage, the
// session state, a link, a log line or an answer (the answers say who it is
// and how it came, nothing more). It comes pasted (POST .../credentials) or
// from the Sign in popup (publish/oauth.ts), whose hand-over the routes
// listen for from the first GET that offers it; the popup's answer is
// announced to the page as SIGN_IN_EVENT, on which the panel asks again.
import type { AppContext, FinetunedRecord } from "../../contracts";
import { readVolume } from "../../finetune/volume";
import type { FinetunePublish } from "../../generated/cellmap-flow";
import { type PublishProgress, publishError, publishFiles, tokenError, whoIs } from "../../publish/hub";
import { appLinks, finetunedFrom, HUB, type Publication, publishedFiles, refOf } from "../../publish/model";
import { HF_OAUTH_CLIENT_ID, listenForSignIn, type SignIn, signInUrl } from "../../publish/oauth";
import type { Router } from "../router";
import { badRequest, json, objectBody } from "../router";
import { pageBase } from "../serving";

/** What the page is told when the Sign in popup has answered (detail:
 * {signed_in, error?}); lib/events.js's HUGGINGFACE_SIGN_IN_EVENT names it too. */
export const SIGN_IN_EVENT = "cellmap-flow:huggingface-sign-in";

/** A Hugging Face repository name (huggingface_hub's rule, as near as a
 * pattern says it): letters, digits, `-`, `_` and `.`, not starting or
 * ending with `-` or `.`, no `--` or `..`, at most 96 characters. */
const REPO_NAME = /^(?!.*(--|\.\.))[A-Za-z0-9](?:[\w.-]{0,94}[A-Za-z0-9])?$/;

/** The seams: the OAuth app's client id (empty: no Sign in, a token
 * only), the clock, and where SIGN_IN_EVENT goes (the page's document; null
 * where there is none). */
export interface PublishDeps {
  clientId: string;
  now: () => Date;
  page: () => EventTarget | null;
}

/** The token in use and who it is. `expiresAt`: a sign-in's, after which it is forgotten. */
interface SignedIn {
  token: string;
  user: string;
  orgs: string[];
  via: "oauth" | "token";
  expiresAt: Date | null;
}

const READ_ONLY = "This token can only read. Make one with write access at https://huggingface.co/settings/tokens (a Write token) and paste that instead.";

/** One model's publish: where it is going, and how far it has got. */
interface Job {
  repo: string;
  folder: string;
  status: "running" | "done" | "failed";
  progress: PublishProgress | null;
  error?: string;
  created?: boolean;
  /** What Hugging Face says the repository is, read back after the commit. */
  private?: boolean;
  at?: string;
}

/** Register the routes. */
export function publishRoutes(router: Router, deps: Partial<PublishDeps> = {}): void {
  const d: PublishDeps = {
    clientId: HF_OAUTH_CLIENT_ID,
    now: () => new Date(),
    page: () => (typeof document === "undefined" ? null : document),
    ...deps,
  };
  let signedIn: SignedIn | null = null;
  // The page's channel for the popup's answer, listened on from the first
  // GET that offers Sign in until an answer comes, and again from the next.
  const channel = crypto.randomUUID();
  let listening = false;
  const jobs = new Map<string, Job>();

  /** The token in use, or null; a sign-in past its expiry is forgotten. */
  function current(ctx: AppContext): SignedIn | null {
    if (signedIn?.expiresAt && signedIn.expiresAt <= d.now()) {
      ctx.log(`Hugging Face: the sign-in as ${signedIn.user} has expired`);
      signedIn = null;
    }
    return signedIn;
  }

  function account(ctx: AppContext): Record<string, unknown> {
    const s = current(ctx);
    if (d.clientId && !listening) {
      listening = true;
      listenForSignIn(channel, (answer) => void signInAnswered(ctx, answer));
    }
    return {
      success: true,
      signed_in: Boolean(s),
      user: s?.user ?? null,
      via: s?.via ?? null,
      expires_at: s?.expiresAt?.toISOString() ?? null,
      oauth_url: d.clientId ? signInUrl(pageBase(), channel) : null,
    };
  }

  /** The popup's answer: its token checked and kept, or why there is none,
   * and the page told either way. */
  async function signInAnswered(ctx: AppContext, answer: SignIn): Promise<void> {
    listening = false; // the listener closes itself on an answer
    let error: string | null = null;
    if ("error" in answer) {
      error = `Hugging Face did not sign you in: ${answer.error}`;
    } else {
      try {
        const who = await whoIs(answer.token);
        if (who.readOnly) {
          error = READ_ONLY;
        } else {
          signedIn = { token: answer.token, user: who.user, orgs: who.orgs, via: "oauth", expiresAt: answer.expires_at ? new Date(answer.expires_at) : null };
          ctx.log(`Hugging Face: signed in as ${who.user} (kept in this page only)`);
        }
      } catch (e) {
        error = tokenError(e);
      }
    }
    if (error) ctx.log(error);
    d.page()?.dispatchEvent(new CustomEvent(SIGN_IN_EVENT, { detail: { signed_in: !error, ...(error ? { error } : {}) } }));
  }

  // GET .../credentials: who is signed in, how, and the Sign in popup's URL
  // (null without an OAuth app: paste a token instead).
  router.get("/api/huggingface/credentials", (_req, _params, ctx) => json(200, account(ctx)));

  // POST .../credentials {token}: a pasted access token, checked with
  // whoAmI and kept in place of any (refused when it can only read);
  // {forget: true}: signed out.
  router.post("/api/huggingface/credentials", async (req, _params, ctx) => {
    const body = objectBody(req);
    if (!body) return badRequest("expected a JSON object");
    if (body.forget === true) {
      if (signedIn) ctx.log(`Hugging Face: signed out (${signedIn.user}'s token forgotten)`);
      signedIn = null;
      return json(200, account(ctx));
    }
    const token = typeof body.token === "string" ? body.token.trim() : "";
    if (!token.startsWith("hf_") || token.length > 4096) {
      return badRequest("Paste a Hugging Face access token: it starts with hf_ (make one at https://huggingface.co/settings/tokens).");
    }
    try {
      const who = await whoIs(token);
      if (who.readOnly) return json(403, { success: false, error: READ_ONLY });
      signedIn = { token, user: who.user, orgs: who.orgs, via: "token", expiresAt: null };
      ctx.log(`Hugging Face: signed in as ${who.user} with an access token (kept in this page only)`);
      return json(200, account(ctx));
    } catch (e) {
      return json(401, { success: false, error: tokenError(e) });
    }
  });

  // POST /api/finetune/model/:name/publish {repo?, private?, include_dataset?}:
  // the finetuned model (by its name, the layer's, or its id) published to
  // `repo` (`owner/name`; blank: the user's own, named after the model).
  // What can be refused at once is: no model, no token, a repository name
  // Hugging Face would refuse, an owner that is neither the user nor one of
  // their organizations, a publish of it under way.
  router.post("/api/finetune/model/:name/publish", async (req, params, ctx) => {
    const body = (req.body === null ? {} : objectBody(req)) as Partial<FinetunePublish> | null;
    if (!body) return badRequest("expected a JSON object");
    for (const key of ["private", "include_dataset"] as const) {
      if (body[key] !== undefined && typeof body[key] !== "boolean") return badRequest(`${key} must be true or false`);
    }
    const record = await recordOf(ctx, params.name);
    if (!record) return json(404, { success: false, error: `No model finetuned in this browser is named ${params.name}` });
    const who = current(ctx);
    if (!who) {
      return json(401, { success: false, error: "Sign in with Hugging Face, or paste an access token with write access, first: the model is published under your own account." });
    }
    const repo = typeof body.repo === "string" && body.repo.trim() ? body.repo.trim() : `${who.user}/${record.name}`;
    const [owner, name, ...rest] = repo.split("/");
    if (!owner || !name || rest.length || !REPO_NAME.test(name)) {
      return badRequest(`${repo} is not a repository name Hugging Face takes: owner/name, the name of letters, digits, - _ and . (not at its ends), at most 96 characters.`);
    }
    if (owner !== who.user && !who.orgs.includes(owner)) {
      const yours = [who.user, ...who.orgs].join(", ");
      return json(403, { success: false, error: `${owner} is not your account or one of your organizations on Hugging Face (${yours}): publish under one of those, ${who.user}/${name} for instance.` });
    }
    if (jobs.get(record.name)?.status === "running") {
      return json(409, { success: false, error: `${record.name} is being published already: wait for it to finish.` });
    }
    const job: Job = { repo, folder: record.name, status: "running", progress: null };
    jobs.set(record.name, job);
    void run(ctx, job, record, who, { private: body.private === true, includeDataset: body.include_dataset === true });
    return json(200, answerOf(ctx, job));
  });

  // GET /api/finetune/model/:name/publish: this page's publish of the model
  // (running, done or failed), else where it was published before, else
  // status "none".
  router.get("/api/finetune/model/:name/publish", async (_req, params, ctx) => {
    const record = await recordOf(ctx, params.name);
    if (!record) return json(404, { success: false, error: `No model finetuned in this browser is named ${params.name}` });
    const job = jobs.get(record.name);
    if (job) return json(200, answerOf(ctx, job));
    if (record.published) return json(200, answerOf(ctx, { ...record.published, status: "done", progress: null }));
    return json(200, { success: true, status: "none" });
  });

  /** The publish itself: the files (publish/model.ts), the repository and
   * the commit (publish/hub.ts), then the record's `published`, kept
   * through the catalog. */
  async function run(ctx: AppContext, job: Job, record: FinetunedRecord, who: SignedIn, options: { private: boolean; includeDataset: boolean }): Promise<void> {
    try {
      const bytes = await ctx.catalog.modelBytes(record.entry);
      const p: Publication = {
        record,
        repo: job.repo,
        folder: job.folder,
        publishedAt: d.now().toISOString(),
        finetunedFrom: finetunedFrom(record, await ctx.catalog.finetuned()),
        dataset: options.includeDataset ? await trainedOn(ctx, record) : null,
      };
      const { created, private: isPrivate } = await publishFiles({
        token: who.token,
        repo: job.repo,
        private: options.private,
        files: publishedFiles(p, bytes),
        title: `Publish ${job.folder} from cellmap-flow`,
        onProgress: (progress) => {
          job.progress = progress;
        },
      });
      // The record as it is now, which a model deleted meanwhile no longer
      // has: it is not kept again. addFinetuned is the catalog's one way to
      // keep a record, and writes the bytes again with it.
      const latest = (await ctx.catalog.finetuned()).find((r) => r.id === record.id);
      if (latest) await ctx.catalog.addFinetuned({ ...latest, published: { repo: job.repo, folder: job.folder, at: p.publishedAt } }, bytes);
      Object.assign(job, { status: "done", created, private: isPrivate, at: p.publishedAt });
      ctx.log(`Published ${job.folder} to ${HUB}/${job.repo} as ${refOf(job.repo, job.folder)}${created ? " (a new repository)" : ""}`);
    } catch (e) {
      job.status = "failed";
      job.error = publishError(e, { repo: job.repo, via: who.via, file: job.progress?.file ?? null });
      ctx.log(job.error);
    }
  }
}

/** The finetuned model `name` names: its name (the layer's) or its id, as the download route takes it. */
async function recordOf(ctx: AppContext, name: string): Promise<FinetunedRecord | undefined> {
  return (await ctx.catalog.finetuned()).find((r) => r.name === name || r.id === name);
}

/** The dataset the model was trained on, as the model card names it: its
 * annotation volume's, or, with the volume no longer in this browser (a
 * model from a link has none), the one open now. */
async function trainedOn(ctx: AppContext, record: FinetunedRecord): Promise<string | null> {
  const volume = record.volume ? await readVolume(ctx.paint, record.volume) : null;
  return volume?.dataset_path ?? ctx.session.state.datasetUrl;
}

/** A publish as the routes answer it. Done, it says where the model is,
 * with the app's links on this page's own address and the dataset open now. */
function answerOf(ctx: AppContext, job: Job): Record<string, unknown> {
  const answer: Record<string, unknown> = { success: true, status: job.status, repo: job.repo, folder: job.folder };
  if (job.status === "running") Object.assign(answer, job.progress ?? { step: "creating", file: null, fraction: null });
  if (job.status === "failed") answer.error = job.error;
  if (job.status === "done") {
    const ref = refOf(job.repo, job.folder);
    const links = appLinks(pageBase(), ref, ctx.session.state.datasetUrl);
    Object.assign(answer, { ref, url: `${HUB}/${job.repo}`, try_link: links.tryLink, dataset_link: links.datasetLink, at: job.at ?? null });
    if (job.created !== undefined) answer.created = job.created;
    if (job.private !== undefined) answer.private = job.private;
  }
  return answer;
}
