# cellmap-flow in the browser: how `web/` is put together

The browser mode runs the cellmap-flow dashboard as a static page, with model
inference done in the page by onnxruntime-web (WebGPU) on chunkmirage's browser
engine, and the result served to Neuroglancer through a service worker. No
server. It is deployed to GitHub Pages beside the docs (`/app/`).

The one rule: **the dashboard is the same dashboard.** Its templates
(`cellmap_flow/dashboard/templates`) and its JavaScript and CSS
(`cellmap_flow/dashboard/static`) are used as they are, byte for byte. What
changes is who answers `fetch("/api/...")`: the Flask routes in server mode, the
TypeScript backend here in browser mode, with the same request and response
JSON. A change to the dashboard (a new finetune control, say) reaches the
browser mode by itself; only a new route needs a browser implementation, or an
honest "not available in the browser" answer.

## The core: a chunk, on demand

```
Neuroglancer (same-origin iframe, ./ng/)
   │ GET virtual/<page>/<model__CFLOW_ARGS__blob__CFLOW_ARGS__>/0/z/y/x   (zarr3://)
   ▼
service worker (src/sw.ts)  ── relays to the page over a MessageChannel; cancels
   │                           when the viewer gives up
   ▼
chunkmirage Engine (web/chunkmirage/web/src/engine.ts)
   │ demand queue: finest level first, abandoned chunks dropped unrun;
   │ 256 MB cache of computed chunks
   ▼
produce(level, index)  (src/infer/)  ─ our code, from Marwan's compute.js:
   read the input block through the engine's reader (output block + context,
   clipped to the volume, zero-padded)  →  normalizers  →  one onnxruntime-web
   session.run on WebGPU (one at a time; WASM fallback)  →  postprocessors  →
   channels-first → zarr bytes
   ▼
bytes stream back through the service worker; the viewer draws them
```

The model's view is registered with `Engine.serve(view, info, chunk, produce)`:
a single level at the model's output voxel size, so Neuroglancer sees one
scale, as the Python server serves it. The raw volume is opened as the engine
view `"raw"` (so `Engine.read` can read it) and shown in the viewer straight
from its own URL (`zarr://https://...`).

## Boot (src/main.ts)

0. `installApiRebase()` – the dashboard's root-relative `/api/...` URLs resolve
   against the page's directory (src/rebase.ts), since on GitHub Pages the app
   is under `/cellmap-flow/app/`, not at the origin's root.
1. `registerServiceWorker()` – `sw.js` beside the page, scope `./`, classic (the
   same registration chunkmirage's Engine makes); waits until it controls the
   page (first visit: one reload).
2. `openSession()` – the session state from IndexedDB (dataset URL, chains,
   served models), which must survive the reload `connect.js` does after
   `/api/set-data`. A shared link's dataset (`datasetUrlFromPage()`) wins over it.
3. `createViewer`, `createInference`, `createCatalog`, then `createBackend(ctx)`
   and `listenForApiRequests(backend)`.
4. `restoreSession(ctx)` – reopen the dataset in the engine and serve the
   session's models again (their views are gone after a reload; the same chain
   gives the same blob, so the restored layers' URLs still work).
5. `viewer.restore()` – the iframe when a dataset is open (from the shared
   link's state when there is one), else the connect panel with a URL field.
6. `import("./static/js/dashboard/main.js")` – the dashboard, unchanged.

## Request path for `/api/*`

The dashboard's JavaScript calls `fetch("/api/...")` (lib/api.js,
finetune/requests.js, raw fetch), opens `EventSource("/api/.../stream")`
(lib/sse.js and two more), and has one `<a download href="/api/export-config">`.
All of these are HTTP requests from the page, so the service worker sees them:
it relays each to the page's backend (`src/backend/`), which routes by method
and path to a handler with the same JSON contract as the Python route. A
streamed response (server-sent events) is relayed chunk by chunk. Nothing in
the dashboard's JavaScript is patched.

Routes that only a server can serve (LSF, the cluster's file systems, the
Review tab's SQLite index, Vertex credentials) answer
`{success: false, available: false, error}` and their controls are hidden by
`[data-mode="browser"] .server-only { display: none }` (the template marks them
with the `server-only` class; render.py sets `data-mode`).

The service worker is one classic script, `sw.js`, bundled on its own from
`src/sw.ts` (build/vite-plugins.ts `serviceWorker`): bundled as one of the page's
entries it would import the chunk it shares with the page (the IndexedDB
helpers), which a classic worker cannot do. It carries chunkmirage's `/virtual/`
relay verbatim, so `Engine.listen()` finds the protocol it expects; re-check it
when `chunkmirage.lock` is bumped.

## State

`src/state/` keeps `SessionState` (see `contracts.ts`) in IndexedDB. Model bytes
(.onnx, 11–43 MB each) are cached in IndexedDB too: Hugging Face rate-limits
anonymous downloads.

## Layout

```
web/
  ARCHITECTURE.md            this file; README.md says how to run and deploy it
  package.json, vite.config.ts, vitest.config.ts, playwright.config.ts, tsconfig*.json
  chunkmirage.lock           the chunkmirage commit; scripts/fetch-chunkmirage.sh clones it
  neuroglancer.lock          the Neuroglancer fork's commit; scripts/fetch-neuroglancer.sh builds it
  build/render.py            renders cellmap_flow/dashboard/templates/index.html → web/index.html
  build/vite-plugins.ts      the service worker bundle; serves/copies the dashboard's static files and a Neuroglancer build
  build/vite-ort.ts          serves/copies onnxruntime-web's wasm files under ./ort/
  chunkmirage/               (gitignored) the clone; we import web/chunkmirage/web/src/engine.ts
  ng/                        (gitignored) the Neuroglancer build, served at ./ng/
  index.html                 (gitignored) rendered
  src/
    main.ts                  boot
    contracts.ts             the types the modules share
    rebase.ts                root-relative /api/ URLs → the page's directory
    browser.css              hides .server-only under data-mode="browser"
    sw.ts                    the service worker: /api/* and /virtual/* relays, /local/ files
    backend/                 the /api/* router and routes, the SW relay (page side)
      index.ts  createBackend(ctx), restoreSession(ctx), the catch-all for server-only routes
      relay.ts  registerServiceWorker(), listenForApiRequests(backend); protocol.ts its messages
      router.ts, sse.ts, yaml.ts (the export), serving.ts (serve and re-serve the session's models)
      routes/{data,pipeline,models,config}.ts
    state/                   openSession(): SessionStore (IndexedDB, db.ts); local-store.ts for picked directories
    infer/                   createInference(): Inference — the Engine, serve() views, produce()
      engine.ts, session.ts (ORT), produce.ts, norms.ts, post.ts, expression.ts (Lambda steps),
      geometry.ts, encode.ts, semaphore.ts
    shared/                  args.ts (__CFLOW_ARGS__ codec, digest), model-info.ts
    models/                  createCatalog(): Catalog — Hugging Face manifests, variant choice, byte cache
    viewer/                  createViewer(): ViewerController — iframe (index.ts), layers.ts (port of
                             viewer/layers.py), display.ts (shaders, ranges), state.ts, share.ts (the URL)
  test/
    unit/                    vitest, one file per module concern
    parity/                  the TypeScript ports against test/fixtures/*.json, which
                             test/fixtures/make_fixtures.py writes from the Python classes
    e2e/                     Playwright smoke test (E2E=1 npm run e2e)
```

## Running it

```
cd web
npm install
npm run fetch:chunkmirage                 # once, and after bumping chunkmirage.lock
bash scripts/fetch-neuroglancer.sh        # once (a minute or two): the fork that paints, into web/ng/
PYTHON=../.pixi/envs/default/bin/python npm run render   # after a template change
npm run dev                               # http://localhost:5173/  (WebGPU needs localhost or https)
```

Without `web/ng/`, development relays `/ng/` from neuroglancer-demo.appspot.com
(upstream Neuroglancer, no painting). With it, the build there is served
instead; restart `npm run dev` after building it, since the choice is made when
Vite loads its config. `web/neuroglancer.lock` pins the fork's commit, the same
one `pixi.toml` installs for the Python viewer: bump both together. The
deployed site always carries the fork at `/app/ng/`.

## Credits

The inference loop, the normalizer/postprocessor ports, the zarr reading, the
local-directory store and the shareable URL come from Marwan Zouinkhi's
serverless prototype (his commits on this branch, under `web/` before this
restructuring), and the models are his exports on Hugging Face. The engine is
chunkmirage's.
