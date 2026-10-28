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
// repository or another, is a new commit there. The model card, the
// manifest and those links are cellmap_flow/portable/publish.py's, the
// Python the server's routes call too, run here through
// portable/publish_calls.py (Inference.python), so a model published from
// either mode is described the same way.
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
import { PYTHON } from "../../generated/python-data";
import { hubStatus, type PublishProgress, publishError, publishFiles, tokenError, whoIs } from "../../publish/hub";
import { publicationOf } from "../../publish/model";
import { listenForSignIn, type SignIn, signInUrl } from "../../publish/oauth";
import type { Router } from "../router";
import { badRequest, json, objectBody } from "../router";
import { pageBase } from "../serving";

/** What the page is told when the Sign in popup has answered (detail:
 * {signed_in, error?}); lib/events.js's HUGGINGFACE_SIGN_IN_EVENT names it too. */
export const SIGN_IN_EVENT = "cellmap-flow:huggingface-sign-in";

/** The Python of a publish: its files, where a published model is, and
 * what refuses one (the server's route calls the same). */
const PUBLISH_CALLS = "cellmap_flow.portable.publish_calls";

/** Where a published model is (portable/publish.py published_links). */
interface PublishedLinks {
  ref: string;
  url: string;
  try_link: string;
  dataset_link: string | null;
}

/** The seams: the sign-in's client id (portable/publish.py's; empty: no
 * Sign in, a token only), the clock, and where SIGN_IN_EVENT goes (the
 * page's document; null where there is none). */
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
    clientId: PYTHON.huggingface.oauth_client_id,
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
          error = PYTHON.huggingface.read_only;
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
  // (null without a client id: paste a token instead).
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
      if (who.readOnly) return json(403, { success: false, error: PYTHON.huggingface.read_only });
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
      return json(401, { success: false, error: PYTHON.huggingface.sign_in_first });
    }
    const repo = typeof body.repo === "string" && body.repo.trim() ? body.repo.trim() : `${who.user}/${record.name}`;
    const refused = await ctx.infer.python<{ status: number; error: string } | null>(`${PUBLISH_CALLS}:repo_refusal`, { repo, user: who.user, orgs: who.orgs });
    if (refused) return json(refused.status, { success: false, error: refused.error });
    if (jobs.get(record.name)?.status === "running") {
      return json(409, { success: false, error: `${record.name} is being published already: wait for it to finish.` });
    }
    const job: Job = { repo, folder: record.name, status: "running", progress: null };
    jobs.set(record.name, job);
    void run(ctx, job, record, who, { private: body.private === true, includeDataset: body.include_dataset === true });
    return json(200, await answerOf(ctx, job));
  });

  // GET /api/finetune/model/:name/publish: this page's publish of the model
  // (running, done or failed), else where it was published before, else
  // status "none".
  router.get("/api/finetune/model/:name/publish", async (_req, params, ctx) => {
    const record = await recordOf(ctx, params.name);
    if (!record) return json(404, { success: false, error: `No model finetuned in this browser is named ${params.name}` });
    const job = jobs.get(record.name);
    if (job) return json(200, await answerOf(ctx, job));
    if (record.published) return json(200, await answerOf(ctx, { ...record.published, status: "done", progress: null }));
    return json(200, { success: true, status: "none" });
  });

  /** The publish itself: the card and the manifest (portable/publish.py,
   * from the record's Publication, publish/model.ts) and the model's bytes,
   * the repository and the commit (publish/hub.ts), then the record's
   * `published`, kept through the catalog. */
  async function run(ctx: AppContext, job: Job, record: FinetunedRecord, who: SignedIn, options: { private: boolean; includeDataset: boolean }): Promise<void> {
    try {
      const bytes = await ctx.catalog.modelBytes(record.entry);
      const publishedAt = d.now().toISOString();
      const pub = publicationOf(record, await ctx.catalog.finetuned(), {
        repo: job.repo,
        folder: job.folder,
        publishedAt,
        dataset: options.includeDataset ? await trainedOn(ctx, record) : null,
      });
      // The Python's answers before the upload, so that once the commit is
      // made nothing is left that could fail the publish.
      const texts = await ctx.infer.python<{ path: string; text: string }[]>(`${PUBLISH_CALLS}:publication`, { pub });
      const { ref, url } = await linksOf(ctx, job);
      const files = [
        ...texts.map(({ path, text }) => ({ path, content: new Blob([text]) })),
        { path: `${job.folder}/${record.entry.variant.file}`, content: new Blob([bytes]) },
      ];
      const { created, private: isPrivate } = await publishFiles({
        token: who.token,
        repo: job.repo,
        private: options.private,
        files,
        title: `Publish ${job.folder} from cellmap-flow`,
        onProgress: (progress) => {
          job.progress = progress;
        },
      });
      // The record as it is now, which a model deleted meanwhile no longer
      // has: it is not kept again. addFinetuned is the catalog's one way to
      // keep a record, and writes the bytes again with it.
      const latest = (await ctx.catalog.finetuned()).find((r) => r.id === record.id);
      if (latest) await ctx.catalog.addFinetuned({ ...latest, published: { repo: job.repo, folder: job.folder, at: publishedAt } }, bytes);
      Object.assign(job, { status: "done", created, private: isPrivate, at: publishedAt });
      ctx.log(`Published ${job.folder} to ${url} as ${ref}${created ? " (a new repository)" : ""}`);
    } catch (e) {
      // Hugging Face refusing the token or the repository is worded as the
      // server words it; anything else (out of reach, the Hub's own reason) here.
      const status = hubStatus(e);
      const refused = status === null ? null : await ctx.infer.python<string | null>(`${PUBLISH_CALLS}:hub_refusal`, { status, repo: job.repo, via: who.via });
      job.status = "failed";
      job.error = refused ?? publishError(e, { repo: job.repo, file: job.progress?.file ?? null });
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

/** Where the published model `job` is, with the app's links on this page's
 * own address and the dataset open now. */
function linksOf(ctx: AppContext, job: Pick<Job, "repo" | "folder">): Promise<PublishedLinks> {
  return ctx.infer.python<PublishedLinks>(`${PUBLISH_CALLS}:published_links`, { repo: job.repo, folder: job.folder, app: pageBase(), dataset: ctx.session.state.datasetUrl });
}

/** A publish as the routes answer it. Done, it says where the model is (linksOf). */
async function answerOf(ctx: AppContext, job: Job): Promise<Record<string, unknown>> {
  const answer: Record<string, unknown> = { success: true, status: job.status, repo: job.repo, folder: job.folder };
  if (job.status === "running") Object.assign(answer, job.progress ?? { step: "creating", file: null, fraction: null });
  if (job.status === "failed") answer.error = job.error;
  if (job.status === "done") {
    Object.assign(answer, await linksOf(ctx, job), { at: job.at ?? null });
    if (job.created !== undefined) answer.created = job.created;
    if (job.private !== undefined) answer.private = job.private;
  }
  return answer;
}
