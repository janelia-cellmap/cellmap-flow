// The publish's calls to Hugging Face, through @huggingface/hub, with the
// user's token: who the token is (whoIs), and the repository made if need be
// and the model's files uploaded in one commit, with progress, then read back
// (publishFiles).
// The library is loaded when first needed, in a chunk of its own, as TF.js
// is (finetune/full/tfjs.ts): most visits never publish. Its errors are put
// in words that say what to do (publishError), never with its URLs, which
// for an upload are presigned storage URLs.
//
// The page calls Hugging Face itself, from the app's origin (github.io, or
// localhost): the Hub's API and its storage answer cross-origin requests,
// which is how the library works in any web page. The service worker leaves
// these requests alone (sw.ts: a request with an Authorization header, and
// anything but a GET).
import type { HubApiError } from "@huggingface/hub";

import { hubMessage } from "./oauth";

/** Who a token is: the user's name, their organizations' (where they may
 * publish too), and whether the token can only read (a fine-grained or an
 * OAuth token says nothing of that until it is used). */
export interface Account {
  user: string;
  orgs: string[];
  readOnly: boolean;
}

/** Where a publish has got: making the repository, then the library's
 * phases (preparing the upload, hashing and uploading a file, the
 * commit), the file and its share done where there is one, then reading
 * the repository back. */
export interface PublishProgress {
  step: "creating" | "preparing" | "hashing" | "uploading" | "committing" | "checking";
  file: string | null;
  fraction: number | null;
}

type Hub = typeof import("@huggingface/hub");
let hub: Promise<Hub> | null = null;
const loadHub = (): Promise<Hub> => (hub ??= import("@huggingface/hub"));

/** Who `token` is (Hugging Face's whoami-v2). Throws the library's error
 * (publishError words it), or one saying the token is not a user's. */
export async function whoIs(token: string): Promise<Account> {
  const { whoAmI } = await loadHub();
  const me = await whoAmI({ accessToken: token });
  if (me.type !== "user") throw new Error(`this is an ${me.type}'s token, not a user's: publish with your own`);
  return { user: me.name, orgs: me.orgs.map((org) => org.name), readOnly: me.auth.accessToken?.role === "read" };
}

/** Make `repo` (a model repository, `owner/name`; private when asked) unless
 * it is there already, and upload `files` to it in one commit titled
 * `title`, telling `onProgress` how far it has got. A repository that is
 * there already takes the files as a new commit, its visibility as it was;
 * `created` says which. Then the repository is read back, so that done means
 * Hugging Face shows each file at the size sent (else this throws), and
 * `private` is what the Hub says the repository is, whatever was asked. */
export async function publishFiles(opts: {
  token: string;
  repo: string;
  private: boolean;
  files: { path: string; content: Blob }[];
  title: string;
  onProgress: (progress: PublishProgress) => void;
}): Promise<{ created: boolean; private: boolean }> {
  const { createRepo, listFiles, modelInfo, uploadFilesWithProgress } = await loadHub();
  const repo = { type: "model" as const, name: opts.repo };
  opts.onProgress({ step: "creating", file: null, fraction: null });
  let created = true;
  try {
    await createRepo({ repo, visibility: opts.private ? "private" : "public", accessToken: opts.token });
  } catch (e) {
    // 409: there already. 403: there already, perhaps, for a token that may
    // write to it but not make repositories (a fine-grained one); the
    // upload says whether it may write.
    if (!isHubError(e) || (e.statusCode !== 409 && e.statusCode !== 403)) throw e;
    created = false;
  }
  for await (const event of uploadFilesWithProgress({ repo, accessToken: opts.token, files: opts.files, commitTitle: opts.title })) {
    if (event.event === "phase") {
      opts.onProgress({ step: event.phase === "preuploading" ? "preparing" : event.phase === "committing" ? "committing" : "uploading", file: null, fraction: null });
    } else if (event.state !== "error") {
      opts.onProgress({ step: event.state, file: event.path, fraction: event.progress });
    }
  }
  opts.onProgress({ step: "checking", file: null, fraction: null });
  const sizes = new Map<string, number>();
  for await (const entry of listFiles({ repo, recursive: true, accessToken: opts.token })) sizes.set(entry.path, entry.lfs?.size ?? entry.size);
  const absent = opts.files.filter((f) => sizes.get(f.path) !== f.content.size).map((f) => f.path);
  if (absent.length) throw new Error(`the commit went through, but Hugging Face does not show ${absent.join(", ")} at the size uploaded`);
  const info = await modelInfo({ name: opts.repo, accessToken: opts.token });
  return { created, private: info.private };
}

function isHubError(e: unknown): e is HubApiError {
  return e instanceof Error && typeof (e as Partial<HubApiError>).statusCode === "number";
}

/** A network failure rather than an answer: fetch's TypeError (Chrome's
 * "Failed to fetch", Firefox's NetworkError, Safari's "Load failed", Node's
 * "fetch failed"), the library's upload request failing with no status, or
 * any failure while the browser says it is offline. */
function isNetworkError(e: unknown): boolean {
  if (typeof navigator !== "undefined" && navigator.onLine === false) return true;
  if (e instanceof TypeError) return /fetch|network|load failed/i.test(e.message);
  return e instanceof Error && !isHubError(e) && e.message === "";
}

const UNREACHABLE = "Hugging Face could not be reached: check the connection (a firewall or a blocker may stop huggingface.co) and try again.";

/** Why a token could not be taken (whoIs failed), in words that say what to do. */
export function tokenError(e: unknown): string {
  if (isNetworkError(e)) return UNREACHABLE;
  if (isHubError(e) && e.statusCode === 401) {
    return "Hugging Face does not accept this token: check that it was copied whole and has not expired or been revoked (https://huggingface.co/settings/tokens).";
  }
  return `Hugging Face could not say whose token this is: ${hubMessage(e)}`;
}

/** What went wrong with a publish to `repo`, in words that say what to do:
 * the Hub out of reach; the token refused (401) or not allowed to write
 * (403), each said for a sign-in (`via` "oauth", whose scopes reach only the
 * repositories made through the app) or a pasted token; else the Hub's own
 * reason, after the file being uploaded when there was one. */
export function publishError(e: unknown, context: { repo: string; via: "oauth" | "token"; file: string | null }): string {
  const { repo, via, file } = context;
  if (isNetworkError(e)) return file ? `Uploading ${file}: ${UNREACHABLE}` : UNREACHABLE;
  if (isHubError(e) && e.statusCode === 401) {
    return "Hugging Face did not accept the token: it may have expired or been revoked. Sign in again, or paste a new access token.";
  }
  if (isHubError(e) && e.statusCode === 403) {
    return via === "oauth"
      ? `Hugging Face does not let this sign-in write to ${repo}: it may make repositories and write to those made through cellmap-flow, and no others. Publish to a new repository, or paste an access token that may write to this one.`
      : `This token may not write to ${repo}: make one with write access at https://huggingface.co/settings/tokens (a Write token, or a fine-grained one that may write to your repositories), or publish to a repository it may write to.`;
  }
  const reason = hubMessage(e);
  return file ? `Uploading ${file} to ${repo} failed: ${reason}` : `Publishing to ${repo} failed: ${reason}`;
}
