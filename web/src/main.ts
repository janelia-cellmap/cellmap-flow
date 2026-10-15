// The browser mode's entry: what web/build/render.py names as the page's
// script in place of the dashboard's main.js.
//
// Boot order matters. The dashboard's JavaScript starts asking /api/... the
// moment it loads, so everything that answers must be ready first:
//   1. the service worker (src/sw.ts) controls this page: it relays /api/*
//      requests, EventSource streams and the Export Config download to the
//      backend here, and /virtual/* chunk requests to chunkmirage's Engine;
//   2. the session (IndexedDB), the inference engine, the model catalog and
//      the viewer controller exist;
//   3. the backend listens for the relayed requests;
//   4. the dashboard's own main.js loads, unchanged, so its controls work at
//      once; its first requests are answered by the backend;
//   5. meanwhile the session is restored: the dataset reopened in the engine
//      and its models served again (the page reloads after /api/set-data, and
//      a shared link may name another dataset than the session's), and then
//      the viewer having been restored first (the iframe, when a dataset is
//      open; the models' layers are set on it as they are served again).
//      A dataset that is a folder on this machine (src/local-folder) comes before
//      both: after a reload the page must be allowed to read it again.
// <html data-dashboard="loaded"> marks step 4 done, and data-session="restored"
// step 5, for tests and for anyone waiting on the page.
// Before any of it, the dashboard's root-relative /api/ URLs are rebased onto
// the page's directory (src/rebase.ts): on GitHub Pages the app is not at the
// origin's root.
import "./browser.css";
import { createBackend, restoreSession } from "./backend";
import { listenForApiRequests, registerServiceWorker } from "./backend/relay";
import type { AppContext, Chain } from "./contracts";
import { createInference } from "./infer";
import { chainFromHash, parseLink, resolveLinkedModels, withoutLink } from "./link";
import { installLocalFolderUi, restoreLocalFolder } from "./local-folder";
import { createCatalog } from "./models";
import { installApiRebase } from "./rebase";
import { installShareUi } from "./share-link";
import { defaultChain, openSession } from "./state";
import { createViewer, datasetUrlFromPage } from "./viewer";

/** The dashboard's script, served verbatim beside the page (vite.config.ts).
 * Resolved against the page, not this module: a relative specifier in
 * import() resolves against the importing module's URL, which is /src/main.ts
 * in development and /assets/index-<hash>.js in the build. */
const DASHBOARD_MAIN = new URL("static/js/dashboard/main.js", document.baseURI).href;

const logLines: string[] = [];
function log(line: string) {
  logLines.push(line);
  if (logLines.length > 500) logLines.shift();
  console.log(`[cellmap-flow] ${line}`);
  document.dispatchEvent(new CustomEvent("cellmap-flow:log", { detail: line }));
}

function showBootError(error: unknown) {
  const message = error instanceof Error ? error.message : String(error);
  console.error("[cellmap-flow] boot failed:", error);
  const el = document.createElement("div");
  el.setAttribute("role", "alert");
  el.style.cssText = "position:fixed;left:16px;right:16px;bottom:16px;padding:12px 16px;background:#3b1111;color:#fecaca;border:1px solid #7f1d1d;border-radius:8px;font:14px/1.5 system-ui;z-index:9999;";
  el.textContent = `The browser mode could not start: ${message}`;
  document.body.appendChild(el);
}

async function boot() {
  document.documentElement.dataset.mode = "browser";
  installApiRebase();
  await registerServiceWorker();
  const session = await openSession();
  // A link to the page names its volume and models in the query (src/link.ts),
  // or a shared `#!` state carries its own dataset; either wins over what this
  // tab's session remembers, before the engine opens anything.
  const link = parseLink(location.search);
  const linked = link.datasetUrl ?? datasetUrlFromPage();
  if (linked && linked !== session.state.datasetUrl) await session.update({ datasetUrl: linked });
  const events = new EventTarget();
  const viewer = createViewer({ session, log });
  const infer = await createInference({ session, log });
  const catalog = createCatalog({ log });
  if (link.models.length) {
    // In place of the session's, as Submit would put them: served on the
    // volume when the session is restored below.
    await session.update({ models: await resolveLinkedModels(catalog, link.models, log) });
    log(`models from the link: ${link.models.join(", ")}`);
  }
  if (link.datasetUrl || link.models.length) history.replaceState(null, "", withoutLink(location.href));
  // A shared view's model layers name the chain they were drawn through:
  // served through it here too, so the link shows what its sender saw.
  const linkedChain = chainFromHash(location.hash);
  if (linkedChain) await session.update({ chain: linkedChain });
  const ctx: AppContext = { session, catalog, infer, viewer, events, log };
  const backend = createBackend(ctx);
  listenForApiRequests(backend);
  // A dataset that is a folder on this machine (src/local-folder) before anything
  // reads it: its read permission is gone after a reload and only a click
  // gets it back, so this may show a prompt on the connect panel and wait.
  await restoreLocalFolder({ session, log });
  // The viewer first: with a dataset in the session it mounts the iframe at
  // once (layers set before it exists are queued), and the connect panel the
  // page hid before its first paint (see index.html's inline script and
  // browser.css) can show again where there is no dataset. Then the engine:
  // restoring the session reads the volume's metadata from its bucket and
  // can take seconds; neither the viewer nor the dashboard need wait for it.
  await viewer.restore();
  delete document.documentElement.dataset.viewer;
  const restored = restoreSession(ctx).then(() => {
    document.documentElement.dataset.session = "restored";
  });
  // The dashboard, exactly as the Flask app serves it. A runtime import so
  // Vite does not bundle it (vite.config.ts serves it verbatim).
  const dashboard = (await import(/* @vite-ignore */ DASHBOARD_MAIN)) as { showChain?: (chain: Chain) => void };
  document.documentElement.dataset.dashboard = "loaded";
  log("dashboard loaded");
  // The page was rendered with the default chain ticked; when the session's
  // is another (submitted before this reload, or a shared link's), the Input
  // and Postprocess lists show that one, which is what the models run through.
  const { chain } = session.state;
  if (JSON.stringify(chain) !== JSON.stringify(defaultChain())) dashboard.showChain?.(chain);
  // The connect panel's Browse button and drop target press Connect, which
  // connect.js (in the dashboard's script) now listens on.
  installLocalFolderUi({ log });
  installShareUi({ session, viewer, log });
  await restored;
}

boot().catch(showBootError);
