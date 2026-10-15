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
service worker (src/sw.ts, on chunkmirage's relay.ts)  ── relays to the page over
   │                           a MessageChannel; cancels when the viewer gives up
   ▼
chunkmirage Engine (web/chunkmirage/web/src/engine.ts, serveStages)
   │ demand queue: finest level first, abandoned chunks dropped unrun;
   │ reads the chunk's input box from the raw level (grown by the model's halo,
   │ clipped to the volume, zero past it); 512 MB cache of each stage's output
   │ under the stage's key; 256 MB cache of served chunks; whole-chunk layout
   ▼
stage 1, the model (src/infer/stages.ts modelStage)  ─ our code:
   normalizers on the data (the padding stays zero, as the Python server pads
   after the chain) → one forward pass through chunkmirage's Onnx (onnx.ts:
   onnxruntime-web on WebGPU, one pass at a time per session; WASM fallback)
   → the block [C, z, y, x], cropped to a chunk the volume ends in
   ▼
stage 2, the postprocessors (postprocessStage)  ─ our code:
   the chain's steps, built once per layer → the chunk in the chain's dtype
   ▼
bytes stream back through the service worker; the viewer draws them
```

The model's view is registered with `Engine.serveStages(view, info, chunk,
"raw", stages, "zero")`: a single level at the model's output voxel size, so
Neuroglancer sees one scale, as the Python server serves it, read from the raw
level at the model's input voxel size (the stage's `inputVoxelSize`). Each
stage has a key (`stages.ts`): the model's names the model file and the
normalizers, the postprocessors' adds the steps, so a postprocess change finds
the model's output in the engine's staged cache and a normalizer change
recomputes. The two map onto the Python server's two ops
(`serving/chunkmirage_ops.py`: InferenceOp with its input_norm, PostprocessOp),
and normalization is in the model stage for the reasons the file gives. The raw
volume is opened as the engine view `"raw"` and shown in the viewer straight
from its own URL (`zarr://https://...`).

## Who provides what

chunkmirage (`web/chunkmirage/`, the clone at `chunkmirage.lock`'s commit) is
the engine: the service worker's relay and store cache (`relay.ts`), the page's
route table (`routes.ts`), the reader and the opened source's levels, the
input box of a chunk with its clipping and padding, the per-stage cache, the
demand queue, the chunk cache, the zarr v3 and OME metadata (`engine.ts`,
`stages.ts`, `asread.ts`, `kept.ts`). Nothing of that is reimplemented here.

cellmap-flow (`src/`) is the app: the model catalog and `model_info`
(`models/`, `shared/model-info.ts`), the `__CFLOW_ARGS__` codec
(`shared/args.ts`), the model's block geometry from its manifest
(`infer/geometry.ts`: the halo, the ratio, the level choice, the served level),
the normalizer and postprocessor bodies (`infer/norms.ts`, `post.ts` and the
segmentation steps), the two stages and their keys (`infer/stages.ts`), the
viewer's layer JSON (`viewer/`), the backend
behind the dashboard's routes (`backend/`), the session (`state/`), a folder
picked on this machine (`local-folder/`) and the dashboard build (`build/`,
`vite.config.ts`).

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
   and `listenForApiRequests(backend)`: a `route("/api/", …)` in chunkmirage's
   route table (routes.ts), the one listener the page has for the worker's
   asks; the Engine adds its `/virtual/<page>/` route there when it starts.
4. `restoreLocalFolder()` (src/local-folder) – when the session's dataset is this
   page's own `local/<key>/`, a folder picked on this machine, its handle must
   still be in IndexedDB and readable before anything opens it. Chrome drops
   a stored handle's read permission on reload (unless the user chose "allow
   on every visit"), and asking again needs a click, so the connect panel
   shows a one-click prompt and boot waits; a handle that is gone (a link
   shared from another machine) clears the dataset, and the panel says so.
5. `restoreSession(ctx)` – reopen the dataset in the engine and serve the
   session's models again (their views are gone after a reload; the same chain
   gives the same blob, so the restored layers' URLs still work).
6. `viewer.restore()` – the iframe when a dataset is open (from the shared
   link's state when there is one), else the connect panel with a URL field.
7. `import("./static/js/dashboard/main.js")` – the dashboard, unchanged; then
   `installLocalFolderUi()`, the panel's "Browse local folder…" button and
   drop target, which fill the URL field and press Connect, so connect.js
   must be listening first.

## Request path for `/api/*`

The dashboard's JavaScript calls `fetch("/api/...")` (lib/api.js,
finetune/requests.js, raw fetch), opens `EventSource("/api/.../stream")`
(lib/sse.js and two more), and has one `<a download href="/api/export-config">`.
All of these are HTTP requests from the page, so the service worker sees them:
chunkmirage's `relay(request, {pick, timeout})` posts each as an Ask (path,
range, method, headers, body) to the app window (not the viewer's iframe; the
requesting tab when it is one), `route("/api/", …)` on the page hands it to
`src/backend/relay.ts`, which builds an ApiRequest and routes by method and
path to a handler with the same JSON contract as the Python route
(`src/backend/`), and answers with a Reply, or a `stream: true` head and
Laters for a streamed response (server-sent events), chunk by chunk, which
the worker turns into the Response. Nothing in the dashboard's JavaScript is
patched. (The Ask carries the path without its query string for now; the one
route that reads a query, the log stream's `offset`, has Last-Event-ID, a
header, first.)

Routes that only a server can serve (LSF, the cluster's file systems, the
Review tab's SQLite index, Vertex credentials) answer
`{success: false, available: false, error}` and their controls are hidden by
`[data-mode="browser"] .server-only { display: none }` (the template marks them
with the `server-only` class; render.py sets `data-mode`).

The service worker is one classic script, `sw.js`, bundled on its own from
`src/sw.ts` (build/vite-plugins.ts `serviceWorker`): bundled as one of the page's
entries it would import the chunk it shares with the page (the IndexedDB
helpers), which a classic worker cannot do. It is chunkmirage's `relay.ts`
(the relay for `/virtual/` and `/api/` alike, and the store cache) plus our
`/local/` route, which serves a picked folder's files from the handle
`src/local-folder` stored (and asks permission for again after a reload; the worker
cannot ask); the Engine registers the same `sw.js` by name, so both get one
registration.

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
  chunkmirage/               (gitignored) the clone; we import web/chunkmirage/web/src/{engine,stages,onnx,relay,routes,asread}.ts
                             and its Vite plugin web/vite-ort.ts (onnxruntime-web's wasm files served/copied under ./ort/)
  ng/                        (gitignored) the Neuroglancer build, served at ./ng/
  index.html                 (gitignored) rendered
  src/
    main.ts                  boot
    contracts.ts             the types the modules share
    rebase.ts                root-relative /api/ URLs → the page's directory
    browser.css              hides .server-only under data-mode="browser"
    sw.ts                    the service worker: chunkmirage's relay for /api/* and /virtual/*, its store cache, our /local/ files
    backend/                 the /api/* router and routes, the SW relay (page side)
      index.ts  createBackend(ctx), restoreSession(ctx), the catch-all for server-only routes
      relay.ts  registerServiceWorker(), listenForApiRequests(backend) on chunkmirage's route(); Ask/Reply/Later are chunkmirage's types
      router.ts, sse.ts, yaml.ts (the export), serving.ts (serve and re-serve the session's models)
      routes/{data,pipeline,models,config}.ts
    state/                   openSession(): SessionStore (IndexedDB, db.ts); local-store.ts for picked directories
    local-folder/            a folder on this machine: index.ts (the local URL and its key, the pick/drop flow,
                             the boot-time permission check), panel.ts (the connect panel's DOM: button, drop target, prompt)
    infer/                   createInference(): Inference — the Engine, one serveStages() view per model
      engine.ts (the views and the stats), stages.ts (the model and postprocess stages, their keys),
      norms.ts, post.ts,
      expression.ts (Lambda steps), geometry.ts (the manifest's blocks, halo, ratio, levels), dtype.ts (astype)
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

The inference loop (now the two stages and the ONNX sessions), the
normalizer/postprocessor ports, the local-directory store and the shareable
URL come from Marwan Zouinkhi's serverless prototype (his commits on this
branch, under `web/` before this restructuring), and the models are his
exports on Hugging Face. The engine, the reading and the relay are
chunkmirage's.
