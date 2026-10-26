// Sign in with Hugging Face: OAuth 2.0 with PKCE, in a popup, so that the
// app's page keeps its state (the open dataset, the served models, the paint)
// while the user signs in. What it gives is an access token for the user's
// own account, which the publish routes keep in the page's memory
// (backend/routes/publish.ts) as a pasted token is kept.
//
// The flow, all of it on this origin but the middle:
// 1. The Publish panel's Sign in button opens oauth.html?channel=<id> in a
//    popup, synchronously in the click, so no popup blocker stops it.
//    <id> is the page's own (signInUrl), so only the tab that asked hears
//    the answer.
// 2. oauth.html (runSignInPopup) asks @huggingface/hub's oauthLoginUrl for
//    Hugging Face's authorize URL, keeps the PKCE code verifier, the nonce and
//    the channel id in the popup's sessionStorage, and goes there.
// 3. The user signs in and allows the app; Hugging Face sends the popup back
//    to oauth.html with a code, which the popup exchanges for the token with
//    the verifier (exchange).
// 4. The popup posts the token (or why there is none) on a BroadcastChannel
//    named after the id (handOver), and closes. The page heard it
//    (listenForSignIn), checks the token with whoAmI and keeps it.
//
// A BroadcastChannel rather than window.opener: Hugging Face's pages may cut
// the popup off from its opener (Cross-Origin-Opener-Policy), and then the
// opener is null by the time the popup is back on this origin.
//
// What is stored where: oauthLoginUrl would put the code verifier and the
// nonce in localStorage (huggingface.co:oauth:code_verifier and :nonce).
// Here both are handed to it in an object instead (its `localStorage`
// option) and kept in the popup's sessionStorage, which is the popup's alone
// and goes when it closes; they are removed as soon as the popup is back.
// Neither is the token, which is in the popup's memory, then the channel's
// message, then the page's memory.
//
// The app is a public one (no secret, PKCE) that nobody registers: its
// client id is the URL of the client metadata document the app serves
// (CIMD), which lists the redirect URLs oauth.html is reached at. The id and
// the scopes are cellmap_flow/portable/publish.py's (HF_OAUTH_CLIENT_ID,
// HF_OAUTH_SCOPES, which say why those scopes), the server's sign-in's too.
import { PYTHON } from "../generated/python-data";
import { message } from "../shared/message";

/** The popup's page, beside the app's (vite.config.ts builds it). Its URL
 * without a query is the redirect URL the client metadata document lists. */
export const SIGN_IN_PAGE = "oauth.html";

/** Where Hugging Face exchanges a code for a token (its
 * openid-configuration's token_endpoint). */
const TOKEN_ENDPOINT = "https://huggingface.co/oauth/token";

/** What the popup hands the page: the token and when it expires, or why
 * there is none. */
export type SignIn = { token: string; expires_at: string | null } | { error: string };

/** Where the popup keeps the flow's state between leaving for Hugging Face
 * and coming back (its sessionStorage). */
const KEPT_KEY = "cellmap-flow:huggingface-sign-in";

/** That state: the page's channel, the PKCE code verifier and the nonce. */
interface Kept {
  channel: string;
  codeVerifier: string;
  nonce: string;
}

function channelName(id: string): string {
  return `cellmap-flow:huggingface-sign-in:${id}`;
}

/** The popup's URL for the page at `base` (the page's directory), handing
 * its answer to channel `id`. */
export function signInUrl(base: string, id: string): string {
  const url = new URL(SIGN_IN_PAGE, base);
  url.searchParams.set("channel", id);
  return url.href;
}

/** The page's half: `onSignIn` with the first answer posted on channel `id`,
 * which is then closed. */
export function listenForSignIn(id: string, onSignIn: (signIn: SignIn) => void): void {
  const channel = new BroadcastChannel(channelName(id));
  channel.onmessage = (event: MessageEvent<unknown>) => {
    const signIn = signInOf(event.data);
    if (!signIn) return;
    channel.close();
    onSignIn(signIn);
  };
}

/** The popup's half: `signIn` posted on channel `id`. */
export function handOver(id: string, signIn: SignIn): void {
  const channel = new BroadcastChannel(channelName(id));
  channel.postMessage(signIn);
  channel.close();
}

// A message is only taken as a sign-in when it is one: anything else on the
// channel (another build's message, say) is not acted on.
function signInOf(data: unknown): SignIn | null {
  if (!data || typeof data !== "object") return null;
  const d = data as Record<string, unknown>;
  if (typeof d.error === "string") return { error: d.error };
  if (typeof d.token !== "string" || !d.token) return null;
  return { token: d.token, expires_at: typeof d.expires_at === "string" ? d.expires_at : null };
}

/** oauth.html's script: on the way out, to Hugging Face's authorize URL; on
 * the way back (a code, or an error, in the query), the exchange, the
 * hand-over and the close. `say` shows a line in the popup, for what the
 * user should know when it cannot close itself or something went wrong. */
export async function runSignInPopup(say: (text: string) => void): Promise<void> {
  const here = new URL(location.href);
  const redirectUrl = here.origin + here.pathname;
  if (here.searchParams.has("code") || here.searchParams.has("error")) {
    const kept = takeKept();
    // The code is single-use and the address bar would show it: out of the history.
    history.replaceState(null, "", redirectUrl);
    if (!kept) {
      say("This sign-in has already been used. Start again from the Publish panel in cellmap-flow.");
      return;
    }
    let signIn: SignIn;
    try {
      signIn = await exchange(here.searchParams, kept, redirectUrl);
    } catch (e) {
      signIn = { error: hubMessage(e) };
    }
    handOver(kept.channel, signIn);
    if ("error" in signIn) {
      say(`Hugging Face did not sign you in: ${signIn.error}. Close this window and try again.`);
      return;
    }
    say("Signed in. You can close this window.");
    window.close();
    return;
  }
  const channel = here.searchParams.get("channel");
  if (!channel) {
    say("Open this page from the Publish panel in cellmap-flow.");
    return;
  }
  // oauthLoginUrl fills these in instead of writing them to localStorage.
  const flow: { codeVerifier?: string; nonce?: string } = {};
  const { oauthLoginUrl } = await import("@huggingface/hub");
  const url = await oauthLoginUrl({
    clientId: PYTHON.huggingface.oauth_client_id,
    scopes: PYTHON.huggingface.oauth_scopes.join(" "),
    redirectUrl,
    localStorage: flow,
  });
  sessionStorage.setItem(KEPT_KEY, JSON.stringify({ channel, codeVerifier: flow.codeVerifier, nonce: flow.nonce }));
  location.replace(url);
}

/** The way back (`query`, the popup's): the code exchanged for the token,
 * as @huggingface/hub's oauthHandleRedirect would, but with the client id
 * in the request, which a public client must send (RFC 6749 §4.1.3) and the
 * library leaves out. First the error Hugging Face sent back, if any, and
 * the state's nonce against the one kept, so that only the code this popup
 * asked for is taken. Throws why there is no token. */
export async function exchange(query: URLSearchParams, kept: Kept, redirectUrl: string): Promise<SignIn> {
  const error = query.get("error");
  if (error) throw new Error(`${error}: ${query.get("error_description") ?? "no reason given"}`);
  const code = query.get("code");
  let nonce: unknown = null;
  try {
    nonce = (JSON.parse(query.get("state") ?? "null") as { nonce?: unknown } | null)?.nonce;
  } catch {
    // not JSON: not this sign-in's state
  }
  if (!code || nonce !== kept.nonce) throw new Error("the answer that came back is not to this sign-in");
  const response = await fetch(TOKEN_ENDPOINT, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({
      grant_type: "authorization_code",
      code,
      // The authorize request's, as the token endpoint checks it.
      redirect_uri: redirectUrl,
      code_verifier: kept.codeVerifier,
      client_id: PYTHON.huggingface.oauth_client_id,
    }),
  });
  const answer = (await response.json().catch(() => ({}))) as { access_token?: string; expires_in?: number; error?: string; error_description?: string };
  if (!response.ok || !answer.access_token) {
    throw new Error(answer.error_description ?? answer.error ?? `the token request answered HTTP ${response.status}`);
  }
  const expiresAt = answer.expires_in ? new Date(Date.now() + answer.expires_in * 1000).toISOString() : null;
  return { token: answer.access_token, expires_at: expiresAt };
}

/** The flow's state the popup kept, removed as it is read; null when there is none. */
function takeKept(): Kept | null {
  const raw = sessionStorage.getItem(KEPT_KEY);
  sessionStorage.removeItem(KEPT_KEY);
  try {
    const kept = JSON.parse(raw ?? "null") as Record<string, unknown> | null;
    if (kept && typeof kept.channel === "string" && typeof kept.codeVerifier === "string" && typeof kept.nonce === "string") {
      return { channel: kept.channel, codeVerifier: kept.codeVerifier, nonce: kept.nonce };
    }
  } catch {
    // not JSON: nothing kept
  }
  return null;
}

/** An error of @huggingface/hub's in a sentence, without the URL and request
 * id it appends (". URL: ..."), which say nothing to the user. */
export function hubMessage(e: unknown): string {
  return message(e).replace(/\. URL: .*$/s, "");
}
