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
//    to oauth.html with a code, which oauthHandleRedirect exchanges for the
//    token with the verifier.
// 4. The popup posts the token (or why there is none) on a BroadcastChannel
//    named after the id (handOver), and closes. The page heard it
//    (listenForSignIn), checks the token with whoAmI and keeps it.
//
// A BroadcastChannel rather than window.opener: Hugging Face's pages may cut
// the popup off from its opener (Cross-Origin-Opener-Policy), and then the
// opener is null by the time the popup is back on this origin.
//
// What is stored where: oauthLoginUrl would put the code verifier and the
// nonce in localStorage (huggingface.co:oauth:code_verifier and :nonce, read
// back and removed by oauthHandleRedirect, but left behind when the exchange
// fails before it gets that far). Here both are handed to it in an object
// instead (its `localStorage` option) and kept in the popup's sessionStorage,
// which is the popup's alone and goes when it closes; they are removed as
// soon as the popup is back. Neither is the token, and the library writes
// the token nowhere: it is in the popup's memory, then the channel's
// message, then the page's memory. (So the way back calls
// oauthHandleRedirect itself: oauthHandleRedirectIfPresent looks for the
// nonce in localStorage whatever it is given, and would find none.)
//
// Without a registered OAuth app there is no client id (HF_OAUTH_CLIENT_ID
// is empty) and no Sign in button: a pasted access token is the only way.
// web/README.md says how the app is registered.
import { message } from "../shared/message";

/** The client id of the Hugging Face OAuth app (a public app: no secret,
 * PKCE). Empty until one is registered (web/README.md, "Publishing a model
 * to Hugging Face"); while it is, the panel offers no Sign in. */
export const HF_OAUTH_CLIENT_ID = "";

/** What the sign-in asks for: who the user is (openid, profile), and to
 * create repositories and write to those made through this app
 * (contribute-repos), which is all a publish does. Not write-repos or
 * manage-repos: with those a page could change or delete any of the user's
 * repositories. The app must be registered with these scopes. */
export const HF_OAUTH_SCOPES = "openid profile contribute-repos";

/** The popup's page, beside the app's (vite.config.ts builds it). Its URL
 * without a query is the redirect URL registered for the app. */
export const SIGN_IN_PAGE = "oauth.html";

/** What the popup hands the page: the token and when it expires, or why
 * there is none. */
export type SignIn = { token: string; expires_at: string | null } | { error: string };

/** Where the popup keeps the flow's state between leaving for Hugging Face
 * and coming back (its sessionStorage). */
const KEPT_KEY = "cellmap-flow:huggingface-sign-in";

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
  const hub = await import("@huggingface/hub");
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
      const result = await hub.oauthHandleRedirect({ redirectedUrl: here.href, nonce: kept.nonce, codeVerifier: kept.codeVerifier });
      signIn = { token: result.accessToken, expires_at: result.accessTokenExpiresAt.toISOString() };
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
  if (!channel || !HF_OAUTH_CLIENT_ID) {
    say("Open this page from the Publish panel in cellmap-flow.");
    return;
  }
  // oauthLoginUrl fills these in instead of writing them to localStorage.
  const flow: { codeVerifier?: string; nonce?: string } = {};
  const url = await hub.oauthLoginUrl({ clientId: HF_OAUTH_CLIENT_ID, scopes: HF_OAUTH_SCOPES, redirectUrl, localStorage: flow });
  sessionStorage.setItem(KEPT_KEY, JSON.stringify({ channel, codeVerifier: flow.codeVerifier, nonce: flow.nonce }));
  location.replace(url);
}

/** The flow's state the popup kept, removed as it is read; null when there is none. */
function takeKept(): { channel: string; codeVerifier: string; nonce: string } | null {
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
