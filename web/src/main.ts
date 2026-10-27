// The browser mode's entry: what web/build/render.py names as the page's
// script in place of the dashboard's main.js.
//
// Boot order matters. The dashboard's JavaScript starts asking /api/... the
// moment it loads, so everything that answers must be ready first:
//   1. the service worker (src/sw.ts) controls this page: it relays /api/*
//      requests, EventSource streams and the downloads' fetches to the
//      backend here, and /virtual/* chunk requests to chunkmirage's Engine;
//   2. the session (IndexedDB), the inference engine, the model catalog and
//      the viewer controller exist; a link's models are in the session (a
//      finetuned one it carries rebuilt into the catalog: src/link.ts), and
//      its `#!` view has lost the layers this browser cannot show;
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
// origin's root. Its download links are saved through fetch (same file):
// Chrome sends a download link's request past the service worker.
import "./browser.css";
import { createBackend, restoreSession } from "./backend";
import { listenForApiRequests, registerServiceWorker } from "./backend/relay";
import type { AppContext, Chain } from "./contracts";
import { createInference } from "./infer";
import { chainFromHash, linkedHash, type LinkedModels, linkNotes, parseLink, resolveLinkedModels, withoutLink } from "./link";
import { installLocalFolderUi, restoreLocalFolder } from "./local-folder";
import { createCatalog } from "./models";
import { installApiDownloads, installApiRebase } from "./rebase";
import { installRunRegionUi } from "./run-region/panel";
import { installShareUi, showLinkNotice } from "./share-link";
import { defaultChain, openSession } from "./state";
import { openPaintStore } from "./state/paint-store";
import { createViewer, datasetUrlFromPage } from "./viewer";
import { withHash } from "./viewer/share";

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
  installApiDownloads();
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
  const paint = openPaintStore();
  let linkedModels: LinkedModels = { entries: [], renamed: {}, leftOut: [] };
  if (link.models.length) {
    // In place of the session's, as Submit would put them: served on the
    // volume when the session is restored below. A finetuned model the link
    // carries is rebuilt here first (its base downloaded, unless the byte
    // cache has it), so that it is in the catalog when it is served.
    linkedModels = await resolveLinkedModels(catalog, link.models, log);
    await session.update({ models: linkedModels.entries });
    log(`models from the link: ${linkedModels.entries.map((m) => m.name).join(", ") || "none this browser can run"}`);
  }
  if (link.datasetUrl || link.models.length) history.replaceState(null, "", withoutLink(location.href));
  // A shared view's model layers name the chain they were drawn through:
  // served through it here too, so the link shows what its sender saw.
  const linkedChain = chainFromHash(location.hash);
  if (linkedChain) await session.update({ chain: linkedChain });
  // A shared view's layers that this browser cannot show are taken out of it
  // before the viewer opens it (viewer.restore() reads the page's hash): the
  // paint of a volume painted in another browser, and the layers of models
  // not served here; a finetuned model renamed here has its layer renamed.
  // The page says what it left out or renamed, beside the Share button.
  const served = new Set(session.state.models.map((m) => m.name));
  const view = await linkedHash(location.hash, { served, renamed: linkedModels.renamed, paint, log });
  if (view.hash !== location.hash) history.replaceState(null, "", withHash(location.href, view.hash));
  const missing = linkedModels.leftOut.length + view.dropped.length > 0;
  showLinkNotice(missing ? "Some of the link is not shown here." : "The link opened with a model renamed.", linkNotes(linkedModels, view.dropped), missing ? "warning" : "info");
  const ctx: AppContext = { session, catalog, infer, viewer, paint, events, log };
  const backend = createBackend(ctx);
  listenForApiRequests(backend);
  // A dataset that is a folder on this machine (src/local-folder) before anything
  // reads it: its read permission is gone after a reload and only a click
  // gets it back, so this may show a prompt on the connect panel and wait.
  await restoreLocalFolder({ session, log });
  // The viewer first: with a view of the dataset in the tab (or a link's
  // `#!` state) it mounts the iframe at once, and with only the session's
  // dataset it waits for the engine to read the volume's geometry (restoring
  // the session, below, which can take seconds; datasetOpened); layers set
  // before it exists are queued either way. The connect panel the page hid
  // before its first paint (see index.html's inline script and browser.css)
  // shows again where there is no dataset, and stays hidden while the viewer
  // waits. The dashboard does not wait for either.
  await viewer.restore();
  const restored = restoreSession(ctx).then(() => {
    document.documentElement.dataset.session = "restored";
  });
  // The dashboard, exactly as the Flask app serves it. A runtime import so
  // Vite does not bundle it (vite.config.ts serves it verbatim).
  const dashboard = (await import(/* @vite-ignore */ DASHBOARD_MAIN)) as { showChain?: (chain: Chain) => void };
  document.documentElement.dataset.dashboard = "loaded";
  log("dashboard loaded");
  // The page was rendered with the default chain ticked; when the session's
  // is another (submitted before this reload, or a shared link's), the
  // Processing tab's lists show that one, which is what the models run through.
  const { chain } = session.state;
  if (JSON.stringify(chain) !== JSON.stringify(defaultChain())) dashboard.showChain?.(chain);
  // The connect panel's Browse button and drop target press Connect, which
  // connect.js (in the dashboard's script) now listens on.
  installLocalFolderUi({ log });
  installShareUi({ session, viewer, catalog, log });
  // The Run tab (src/run-region): its own DOM, the folder picker in its
  // clicks and the engine's materialize; nothing of the dashboard's script.
  installRunRegionUi(ctx);
  await restored;
}

boot().catch(showBootError);
