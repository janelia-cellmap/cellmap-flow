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

1. `registerServiceWorker()` – `sw.js` beside the page, scope `./`; waits until
   it controls the page (first visit: one reload).
2. `openSession()` – the session state from IndexedDB (dataset URL, chains,
   served models), which must survive the reload `connect.js` does after
   `/api/set-data`.
3. `createViewer`, `createInference`, `createCatalog`, then `createBackend(ctx)`
   and `listenForApiRequests(backend)`.
4. `viewer.restore()` – the iframe when a dataset is open, else the connect
   panel with a URL field.
5. `import("./static/js/dashboard/main.js")` – the dashboard, unchanged.

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

## State

`src/state/` keeps `SessionState` (see `contracts.ts`) in IndexedDB. Model bytes
(.onnx, 11–43 MB each) are cached in IndexedDB too: Hugging Face rate-limits
anonymous downloads.

## Layout

```
web/
  ARCHITECTURE.md            this file
  package.json, vite.config.ts, tsconfig*.json
  chunkmirage.lock           the chunkmirage commit; scripts/fetch-chunkmirage.sh clones it
  neuroglancer.lock          the Neuroglancer fork's commit; scripts/fetch-neuroglancer.sh builds it
  scripts/                   fetch-chunkmirage.sh, fetch-neuroglancer.sh (the fork that paints)
  build/render.py            renders cellmap_flow/dashboard/templates/index.html → web/index.html
  build/vite-plugins.ts      serves/copies the dashboard's static files and a local Neuroglancer build
  chunkmirage/               (gitignored) the clone; we import web/chunkmirage/web/src/engine.ts
  ng/                        (gitignored) a Neuroglancer build, served at ./ng/
  index.html                 (gitignored) rendered
  src/
    main.ts                  boot
    contracts.ts             the types the modules share
    sw.ts                    the service worker: /api/* and /virtual/* relays, /local/ files
    backend/                 the /api/* router and routes, the SW relay (page side)
      index.ts  createBackend(ctx): Backend
      relay.ts  registerServiceWorker(), listenForApiRequests(backend)
      router.ts, routes/{data,pipeline,models,finetune,config,stubs}.ts, sse.ts
    state/                   openSession(): SessionStore (IndexedDB)
    infer/                   createInference(): Inference — Engine setup, serve() views, produce()
      engine.ts, session.ts (ORT), produce.ts, norms.ts, post.ts, geometry.ts, encode.ts
    shared/                  args.ts (__CFLOW_ARGS__ codec, digest), model-info.ts
    models/                  createCatalog(): Catalog — Hugging Face manifests, variant choice, byte cache
    viewer/                  createViewer(): ViewerController — iframe, layers JSON (port of viewer/layers.py), position
  test/                      vitest; fixtures generated by test/fixtures/make_fixtures.py from the Python classes
  legacy/                    Marwan's prototype files, deleted as each is ported
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

The inference loop, the normalizer/postprocessor ports, the zarr reader and the
local-directory store come from Marwan Zouinkhi's serverless prototype
(`web/legacy/`, his commits on this branch). The engine is chunkmirage's.
