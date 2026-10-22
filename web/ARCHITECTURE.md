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
   │ clipped to the volume, its edge repeated past it); 512 MB cache of each stage's output
   │ under the stage's key; 256 MB cache of served chunks; whole-chunk layout
   ▼
stage 1, the model (src/infer/stages.ts modelStage)  ─ our code:
   normalizers on the data, then its edge repeated past the volume (as the
   Python server pads, after the chain) → one forward pass through chunkmirage's Onnx (onnx.ts:
   onnxruntime-web on WebGPU, one call into ORT at a time on the page whatever
   the model, its `exclusive`, which the finetune's feature session takes
   too; WASM fallback)
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
`stages.ts`, `asread.ts`, `kept.ts`), and a region of a view written out as
an OME-Zarr of its own (`materialize.ts`). Nothing of that is reimplemented
here.

cellmap-flow (`src/`) is the app: the model catalog and `model_info`
(`models/`, `shared/model-info.ts`), the `__CFLOW_ARGS__` codec
(`shared/args.ts`), the model's block geometry from its manifest
(`infer/geometry.ts`: the halo, the ratio, the level choice, the served level),
the normalizer and postprocessor bodies (`infer/norms.ts`, `post.ts` and the
segmentation steps), the two stages and their keys (`infer/stages.ts`), the
viewer's layer JSON (`viewer/`), the backend
behind the dashboard's routes (`backend/`), the session (`state/`), a folder
picked on this machine (`local-folder/`), finetuning (`finetune/`),
AI-assisted annotation (`ai-annotate/`), the Label it tools
(`label-it/`) and Run on a region (`run-region/`), all four below, and the
dashboard build (`build/`, `vite.config.ts`).

## Boot (src/main.ts)

0. `installApiRebase()` – the dashboard's root-relative `/api/...` URLs resolve
   against the page's directory (src/rebase.ts), since on GitHub Pages the app
   is under `/cellmap-flow/app/`, not at the origin's root. `installApiDownloads()`
   (same file) – a click on one of its download links (`<a download>`: Export
   Config, the Finetune tab's volume and model) is fetched and saved from a
   `blob:` URL, since Chrome sends a download link's own request to the
   network, past the service worker.
1. `registerServiceWorker()` – `sw.js` beside the page, scope `./`, classic (the
   same registration chunkmirage's Engine makes); waits until it controls the
   page (first visit: one reload).
2. `openSession()` – the session state from the tab's sessionStorage (dataset URL, chains,
   served models), which must survive the reload `connect.js` does after
   `/api/set-data`. A shared link's dataset (`datasetUrlFromPage()`) wins over it.
3. `createViewer`, `createInference`, `createCatalog`, then `createBackend(ctx)`
   and `listenForApiRequests(backend)`: a `route("/api/", …)` in chunkmirage's
   route table (routes.ts), the one listener the page has for the worker's
   asks; the Engine adds its `/virtual/<page>/` route there when it starts.
   Between the catalog and the backend, `resolveLinkedModels` and `linkedHash`
   (src/link.ts) put a link's models in the session and take out of its `#!`
   view what this browser cannot show, before `viewer.restore()` reads the
   hash; `showLinkNotice` says what was left out (README.md, "A link that
   opens a volume and runs models").
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
6. `viewer.restore()` – the iframe when a dataset is open (from a link's `#!`
   state when the page has one, else from the tab's copy of the viewer's
   state, share.ts), else the connect panel with a URL field. The address bar
   itself never carries the state: Chrome completes a typed `localhost:5174`
   to the last URL it saw, and a new tab must start fresh.
7. `import("./static/js/dashboard/main.js")` – the dashboard, unchanged; then
   `installLocalFolderUi()`, the panel's "Browse local folder…" button and
   drop target, which fill the URL field and press Connect, so connect.js
   must be listening first; the Share button; and `installRunRegionUi()`,
   the Run tab (below).

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
Review tab's SQLite index, the crops import) answer
`{success: false, available: false, error}` and their controls are hidden by
`[data-mode="browser"] .server-only { display: none }` (the template marks them
with the `server-only` class; render.py sets `data-mode`).

The service worker is one classic script, `sw.js`, bundled on its own from
`src/sw.ts` (build/vite-plugins.ts `serviceWorker`): bundled as one of the page's
entries it would import the chunk it shares with the page (the IndexedDB
helpers), which a classic worker cannot do. It is chunkmirage's `relay.ts`
(the relay for `/virtual/` and `/api/` alike, and the store cache) plus two
routes of ours directly under the page's directory: `local/`, which serves a
picked folder's files from the handle `src/local-folder` stored (and asks
permission for again after a reload; the worker cannot ask), and `paint/`,
the annotation volumes' files from the paint store (`state/paint-store.ts`,
IndexedDB) as an S3-compatible endpoint (`state/paint-route.ts`: GET, HEAD,
PUT, DELETE of whole objects), which the Neuroglancer fork's writable zarr
source paints into as it paints into MinIO in server mode; the Engine
registers the same `sw.js` by name, so both get one registration.

## One implementation per behaviour

A feature works in both modes only when its logic is written once, where
both modes run it: a shared Python core, chunkmirage (engine work), the
dashboard's JavaScript and templates (UI), or the generated contracts
(`cellmap_flow/schema.py`, below). Only the I/O adapters are written per
mode. A feature whose logic cannot live in one of those places is
mode-specific: the other mode answers `unavailable` with a reason
(`router.ts`), and the template hides its controls (`server-only`,
`[data-mode]`). A new experiment lands in one mode first and reaches the
other once its logic is portable; it is not ported by hand.

| both modes (logic written once) | browser only | server only |
|---|---|---|
| view; input and postprocess chains; paint volumes; Label it; rank, threshold, norm choice and re-check; targets; good regions; AI annotate (prompt, mask, fill); export config | ORT Web inference; head and TF.js finetune; local folders; Run on a region to disk; share links | LSF model servers; LoRA and whole-network torch training; blockwise; Review tab; crops import; bioimageio, dacapo and Cellpose models; pipeline builder |

Both modes have "a finetune", but not one algorithm: what they share is the
targets, the sampling plan, the loss definitions and the job card's JSON.
Several both-mode rows are still ports kept in step by the parity fixtures
(`test/parity`); the rule says where their one implementation goes next.

The shared Python core is `cellmap_flow/portable/`: modules importable with
the standard library, numpy, scipy, pydantic and Pillow alone
(`tests/portable/test_boundary.py` holds them to that), which the server
runs in its process and this page runs in chunkmirage's Pyodide worker.
`build/portable.ts` bundles them as text (`import.meta.glob`, as chunkmirage
bundles its own ops) and `infer/engine.ts` hands them to `Engine.start` as
`modules` when a dataset opens, with `python: "always"` and one worker, so
Pyodide and numpy (about 17 MB once per browser, cached after) load in the
background while the data opens; the log says when they are in.
`Inference.python("cellmap_flow.portable.calls:<fn>", args, arrays)` is
chunkmirage's `Engine.call` by module path: `fn(args, arrays)` in Python,
with the JSON arguments and the arrays' bytes, answering JSON, one array or
JSON with arrays (`portable/calls.py` decodes and encodes; the maths is in
the modules beside it). The workers take turns, so a call carries its own
state: the rank's tallies travel in its JSON. The first such Python is the
rank score and the paint carry (Finetuning, below); the TypeScript ports
they replace are gone with their unit tests, and
`test/unit/finetune-rank.test.ts` runs the routes on the same Python in
node's Pyodide (`test/unit/pyodide.ts`: the `pyodide` package at the version
the workers load, numpy's wheel from jsDelivr the first time, so without
the network those tests skip).

The contract is generated, not copied. `cellmap_flow/schema.py` exports a
JSON Schema of every pydantic request model (`dashboard/requests.py`), the
table of every route with where it is answered (both, stub, server, browser;
the method and path from the Flask app's `url_map`, the availability declared
beside it) and the constants both modes share (the rank score's, Undo's
depth, the AI annotation's caps, the Finetune form's defaults). `npm run gen`
writes `src/generated/cellmap-flow.schema.json`, `cellmap-flow.ts` (its types
by json-schema-to-typescript, with the constants, `ROUTES` and the chain
steps' parameters as values) and `docs/api.md`. `tests/web/test_schema.py`
fails when they are stale, when a Flask route is added without a declaration,
when the router here registers a route the table does not give the browser or
misses one it does, and when the server form's defaults leave
`FinetuneSubmit`'s; `test/unit/generated-routes.test.ts` holds the router to
`ROUTES`. TypeScript imports a shared number (`FINETUNE_DEFAULTS`,
`LABEL_MAX_CHARS`) from the generated file rather than writing it again.

## Finetuning (src/finetune/)

The Finetune tab works as on the server with the browser in the server's and
MinIO's places. `volume.ts` (the port of `finetune/session/volume.py`) plans
an annotation volume over the open dataset on the grid the model's
predictions are served on, one chunk per output block, and writes its zarr v2
metadata (uncompressed chunks, `.` separator) into the paint store;
`viewer/layers.ts` adds it as the fork's writable segmentation layer on
`s3+http(s)://<host><page dir>paint/<volume>/annotation`. Training fits the
model's head, the last 1×1 convolution of the ONNX graph: `onnx-head.ts` finds
it (walking back from the output through the shape-only ops) and makes two
edits with `onnx-proto.ts`'s field-level protobuf splicing, the head's input
tensor as a second graph output and new weights in the head's initializers;
`samples.ts` cuts, pads and normalizes each painted chunk's input block
exactly as the model stage does (the port of `finetune/data/reader.py`);
`features.ts` runs the model for its logits and features; `target-call.ts`
has the targets and masks made by `cellmap_flow.portable.targets`, the
Python `target_transforms.py` wraps on the server (`Inference.python`,
above), from the offsets the channel names say;
`head-trainer.ts` gathers the painted voxels' features and trains the head
from the model's own weights with Adam on a masked, class-balanced loss
(BCE-with-logits, MSE or margin: the server's per-voxel losses), plus λ ×
the distillation term toward the starting head on the voxels
`distillation.ts` samples from the same feature maps (the painted chunks'
unpainted ones, all of them, or the good regions' unpainted ones, which
`good-regions.ts` places on the volume's grid); `run.ts` is one run of all
of it. The
result is a model entry of the `finetuned` repo (`models/finetuned.ts`: a
record in IndexedDB, its bytes in the byte cache), served like any other;
a Restart's model replaces the job's previous entry, whose record and bytes
are deleted. Share puts a finetuned model in a link as its head's numbers
(`link.ts` `finetunedRef`, read from the bytes with `headOf`; README.md has
the format), which the page that opens it writes back into the base.
Which model to start from (`backend/routes/rank.ts`) scores a model on the
paint with the engine's own reads: the model served (or the served one
reused), `readOutput` over at most `RANK_BLOCKS` of its blocks under the
paint, each painted voxel's model voxel from Label it's placement
(`cellmap_flow.portable.label_it`, through `label-it/prediction.ts` `place`),
and the targets from `targetTransformFor`; then each chunk's labels,
targets and logits go to `cellmap_flow.portable.rank` (`Inference.python`,
above), where the tallies, the score at the model's own cut-off and the
stretched window are decided, in the same Python the server dashboard's
routes run in their process (`dashboard/routes/finetune/rank_routes.py`),
so there is no port to keep in step. *Use this model*'s carry of the paint
onto the new grid is `cellmap_flow.portable.carry` the same way, the paint
store's reads and writes around it here and the zarr's on the server
(`carry_paint` in `finetune/session/volume.py`).
`backend/routes/finetune.ts`, `good-regions.ts` and `finetune-jobs.ts` answer the tab's routes
with the Python routes' JSON (a job is a run in the page; its log streams as
server-sent events), so the tab's job card, loss plot and log stream are
unchanged. Shift+R over the viewer, bound by the job list route
(`ctx.viewer.bindKey`), sends the page `cellmap-flow:restart-finetune`, on
which the tab's `job-monitor.js` restarts the job as its Restart does,
without the confirm, or says why not; the server's key counts the press for
the tab's poll instead (the docs' architecture page, "A finetune job"). Why
the head: onnxruntime-web trains nothing on the GPU, so the whole network
trains only where it is rebuilt in a library that does.

That is `finetune/full/`, for the models whose manifest offers it.
`modes.ts` reads a manifest's `finetune` block (`models/manifest.ts`
`parseFinetune`): `head` when the served file is one of the head's `files`
(fp32 and fp16; an int8w file's weights are DequantizeLinear inputs), `full`
when its library is `tfjs` and the file is one of `write_back` (for a 3D
model, also with `torch_params` and a `weight_map` file, below), and without
the block the head the graph is found to end in.
`GET /api/finetune/models` answers them as `finetune_modes`, as the server's
`finetune_modes()` does, and `run.ts` sends a run to the head's trainer or
to `full/run-full.ts` by `FinetuneOptions.mode`. Marwan Zouinkhi exported
each model with a training graph (`train_graph.json`: NHWC ops for a 2D
model, named as the ONNX initializers; NDHWC for a 3D one, named as the
PyTorch state_dict, with `torch_params.json` and `weight_map.json` to rebuild
each param from the initializers and cut it back into them) and the
initializers' byte offsets in every `.onnx` (`initializers.json`), which
`files.ts` fetches from the folder's `finetune/` on the Hub, and wrote the
TF.js runtime that builds, trains and writes it back
(`cellmap-tfjs-train.ts`, his `finetune/cellmap_tfjs_train.js` vendored and
ported to TypeScript). `graph.ts` checks the graph and walks its shapes to
find the sides it takes (92 + 8k for every graph; the form offers a 2D
model's tiles nearest 172, 300, 428 and 556, with the manifest's measured
step time and memory, and a 3D model's blocks from 124³ to 172³, with
`run-full.ts`'s `BLOCK_COSTS`, measured on an RTX 2080 Ti); `tiles.ts` cuts
the paint into output-sized tiles in the plane each stroke was drawn in (the
server's `paint_depth_axes` and `slice_axes`), or for a 3D model a grid of
output-sized cubes over it, and reads each one's raw as the model stage
would; `augment.ts` draws each step's brightness, noise, flips and y–x turns
of a tile or block; `tfjs.ts` loads TF.js 4.22.0, in chunks of its own, when
a run starts (WebGPU in the page, which the routes require through
`webgpuMissing`; the CPU backend in Node, where the tests train) and resets
its engine when the last run ends, freeing the GPU buffers it pools;
`run-full.ts` checks the offsets against the served file, trains with the
head's loss (`loss.ts`) and Adam, BatchNorm's running statistics held
(`frozen`), and writes the trained weights into a copy of the bytes, which
then go the way a head's do. A 3D model's network is `net3d.ts`'s: the same
graph trained in a fraction of the memory plain autodiff takes (fused
convolution, batch norm and ReLU, the backward pass recomputed from
checkpoints), one block a step, a step that runs out of GPU memory
(`GpuMemoryError`) ending the run with a smaller block to try.

## AI-assisted annotation (src/ai-annotate/)

The port of `cellmap_flow/ai_annotate/` and of
`dashboard/routes/finetune/ai_annotate.py`, so the Finetune tab's AI panel
(`static/js/dashboard/finetune/ai-annotate.js`) drives the page as it drives
the server: `backend/routes/ai-annotate.ts` answers the Python routes' JSON
(config, credentials, settings, run, status with `?preview=1`, resend,
accept, reject; Accept's Undo is Label it's, below), and
`backend/ai-annotate-job.ts` is the job, one at a time, its result staged in
memory until Accept or Reject.
Who does what: `geometry.ts` plans the plane (the depth axis from the
viewer's layout, the raw box, the image laid out as the panel draws it, the
click in image pixels); `plane.ts` reads it from the raw level through
`Inference.readRaw` (clipped to the level and zero past it), resizes it to the image sent, normalizes it to uint8 RGBA (`mask.ts`)
and builds the request; `gemini.ts` is the two Google providers
(`gemini_api`: generativelanguage.googleapis.com with `x-goog-api-key`;
`vertex_gemini`: aiplatform.googleapis.com, location global, with a bearer
token), called straight from the page, with the Python's retries and
timeout, and errors in the module's own words, never the service's body;
`fake.ts` paints the dark pixels in a disc around the click, for trying the
panel and for the tests; `mask.ts` reads the model's image into a mask by
the colour the prompt asked for and draws the overlay; `labels.ts` and
`fill.ts` are `fill.py` (4-connected objects in the plane, ids seeded from
what the user painted beside, the box written to the paint store whole
chunk by whole chunk, with the undo entry); `prompts.ts` and `organelles.ts`
are the catalog; `credentials.ts`, `usage.ts` and `audit.ts` hold the
credentials, the daily count (localStorage, 200 a day) and the audit log,
one JSON line per event appended to `<volume>/ai_annotate_log.jsonl` in the
paint store, so it travels with the volume's zip. The viewer's side is
`viewer/index.ts`'s pointerPosition, viewDepthAxis, bindKey (Shift+G, bound
again when a new viewer is mounted) and refreshLayer.

Two things are the browser's own. Credentials: the page calls Google itself,
so the panel takes an API key, or a project and an access token
(`POST .../credentials`, posted by the panel as they are pasted), kept in
the page's memory only: not in localStorage, where a key would sit
unencrypted for any script on the site (an earlier build's remembered copy,
`cellmap-flow:ai-annotate:credentials`, is deleted on load), never in the
session state (which share links and the state store carry), never in a log
line, and in no answer beyond `configured: true/false` per provider. The layer
refresh: the server respells the paint layer's URL (a leading zero on the
port) so Neuroglancer makes a new source and re-reads it; here the paint
route's volume segment takes an optional `~<n>` suffix
(`paint/<volume>~1/annotation/...`), which `state/paint-route.ts` strips
before the store, and `refreshLayer` toggles it on the layer's source URL
through the viewer state, keeping the layer's tool, brush and paint value. A
state Neuroglancer throws on part way through taking it (seen restoring the
layout after a seed) opens the viewer again on that state, in a new iframe in
the old one's place, so the labels written are shown and the write is not
reported failed; an iframe reload would have gone back to the state the viewer
was first given, without the layers added since.

## Label it (src/label-it/)

The Finetune tab's one-click patch tools, Seed from Prediction, All
Background, Split Objects and Undo: `backend/routes/view-labels.ts` answers
`dashboard/routes/finetune/view_labels.py`'s routes (sources, seed,
background, split, undo) with their JSON, so the tab's `label-it.js` drives
both modes. A seed reads the chosen served model's own output under the box
on screen: `Inference.readOutput`, the model stage's output (the chain's
normalizers, none of its postprocessors), as the server reads its model
server with no postprocessing. `infer/engine.ts` serves that as a second
staged view beside the layer, `<view>~output`, the model stage alone under
the same stage object and key, so the chunks the viewer had computed come
out of the engine's staged cache; the page reads them over the view's URL
through the service worker, as any client reads a view (the engine answers
a page's requests through the route it registers, and has no other door to
a chunk), and `infer/output.ts` places them in the box. The seed's maths
is the server's own Python, `cellmap_flow/portable/label_it.py`, called
through `portable/label_it_calls.py` (`Inference.python`, above): the
request's options, the channels it reads, each annotation voxel on the
served voxel its centre lies in (any served model can seed, at its own
voxel size), the output as a probability by its output class, and threshold
and connected components; `label-it/prediction.ts` reads the served box it
names. For an affinity model the Python answers the mutex watershed's input
and `label-it/segment.ts` runs the watershed itself (`infer/affinity.ts`,
parity-tested against mwatershed, which has no Pyodide wheel). scipy, for
the components, loads the first time a seed needs it, not when a dataset
opens. The distance watershed and a model's own instances are not
offered: no browser model outputs a distance or instance ids. The ids, the
writes and the undo are `ai-annotate/labels.ts` and `fill.ts` (seedLabels,
relabelObjects for Split Objects, paintBox, rewriteForeground, restoreBox).
One undo stack per volume is the routes', and the AI annotation's Accept
pushes onto it (createRouter hands it over, as `ai_annotate.py` uses
`view_labels._undo_stack`), so one Undo takes back whichever came last.

## Run on a region (src/run-region/)

The Run tab (`templates/_run_tab.html`, included by `_dashboard.html` in
browser mode only) runs the served models over a region and saves their
predictions as OME-Zarr in a folder on this machine, where the viewer only
ever computes the chunks it shows. It is the one tab that is not a dashboard
script on `/api/` routes: its folder is picked with `showDirectoryPicker`,
whose picker and permission prompt need the user's click in this window, and
the engine writes into the directory handle, which no request through the
service worker can carry. So `panel.ts` wires the tab's DOM from the page,
as `share-link.ts` wires the header's Share button, and `main.ts` installs
it; `tests/web/test_render.py` checks the ids of its `IDS` table are on the
browser page and nowhere on the Flask one.

`region.ts` is the geometry: the region is a box in nm, z y x, from the
latest axis-aligned box in the `run_region` annotation layer (a local layer
at 1 nm, built as the good regions' layer is, with Neuroglancer's
`annotateBoundingBox` as its `tool`: a legacy annotation tool, which
`toolBindings` cannot take; `ViewerController.layer` reads it from the live
viewer's state, the tab polling once a second while shown), from the whole
volume (the finest level's extent), or typed (`withBox` redraws the box from
the fields once typing has paused for `TYPED_BOX_DELAY_MS`); each model's
box is the voxels of its served level the region touches (`servedLevel`,
the level `infer/engine.ts` serves its view at, voxel 0's centre its OME
translation, as a seed is placed on it), and its chunks are
its output blocks, which the summary counts. `run.ts` runs the models one
after another through `Inference.materialize`, with the progress across
them, the stop (an AbortSignal), and the provenance under `cellmap_flow` in
each zarr's group attributes (the model's name, id, repository, folder,
variant and file, its chain from the served view's blob, the dataset, the
region and the box, the voxel size, the time, the page's address).
`infer/engine.ts` maps materialize onto chunkmirage's `Engine.materialize`
on the layer's own view at its one level, with `extent: "box"` (an array of
exactly the box, its chunks stitched from the model's and the last ones
ending at the box), as zarr v2 with OME-Zarr 0.4 metadata, gzip, empty
chunks skipped; the engine walks the chunks through the view's stages,
caches and demand queue (behind the viewer's chunks, and nothing the viewer
has computed is computed again) and writes the metadata and the chunks,
none of which is here. `output.ts` is the rest: the folder's handle in
IndexedDB (`run-region:output`, beside the picked volumes'), write
permission asked in the Run click, a `<model>.zarr` in the way removed whole
after a confirm (empty chunks are skipped, so an old one would survive in a
new zarr), the zip of a run kept in memory where there is no File System
Access API (Firefox, Safari; up to `ZIP_LIMIT_BYTES` before compression,
with `finetune/zip.ts`), and the layer a saved zarr opens as: the model's
own layer (`predictionLayer`) on `zarr2://<page>local/<key>/<model>.zarr/`,
the folder's handle stored under a fresh `local/` key for the service
worker's `local/` route, so the saved predictions are read back from the
folder.

## State

`src/state/` keeps `SessionState` (see `contracts.ts`) in the tab's
sessionStorage. IndexedDB holds what outlives the tab: picked folders' handles
(`db.ts`; the Run tab's output folder among them), the paint (`paint-store.ts`), the model bytes (.onnx, 4–43 MB
each, keyed by URL and ETag: the page asks Hugging Face for a file's ETag
before serving its cached copy, so a model re-exported under the same name
is downloaded again; Hugging Face rate-limits anonymous downloads) and the finetuned
models' records and bytes (`models/`). localStorage holds the AI
annotation's daily call count (`ai-annotate/usage.ts`); its credentials are
in no storage at all (`credentials.ts`).

## Layout

```
web/
  ARCHITECTURE.md            this file; README.md says how to run and deploy it
  package.json, vite.config.ts, vitest.config.ts, playwright.config.ts, tsconfig*.json
  chunkmirage.lock           the chunkmirage commit; scripts/fetch-chunkmirage.sh clones it
  neuroglancer.lock          the Neuroglancer fork's commit; scripts/fetch-neuroglancer.sh builds it
  build/render.py            renders cellmap_flow/dashboard/templates/index.html → web/index.html
  build/export_data.py       the constants and catalogs shared with Python → src/generated/python-data.json
  build/portable.ts          the Python both modes run (cellmap_flow/portable) as text, for the engine's Pyodide worker
  build/vite-plugins.ts      the service worker bundle; serves/copies the dashboard's static files and a Neuroglancer build
  chunkmirage/               (gitignored) the clone; we import web/chunkmirage/web/src/{engine,stages,onnx,relay,routes,asread,materialize}.ts
                             and its Vite plugin web/vite-ort.ts (onnxruntime-web's wasm files served/copied under ./ort/)
  ng/                        (gitignored) the Neuroglancer build, served at ./ng/
  index.html                 (gitignored) rendered
  src/
    main.ts                  boot
    contracts.ts             the types the modules share
    generated/               python-data.json (written by build/export_data.py, committed; never edited by hand) and
                             python-data.ts, which gives it to the modules: the op registry and which ops the browser runs,
                             the default chain, the organelle catalog and prompt sentences, the shaders and colours, the
                             ranking's and the finetune log's constants, the unit table
    rebase.ts                root-relative /api/ URLs → the page's directory
    browser.css              hides .server-only under data-mode="browser"
    sw.ts                    the service worker: chunkmirage's relay for /api/* and /virtual/*, its store cache, our local/ files and paint/ store
    backend/                 the /api/* router and routes, the SW relay (page side)
      index.ts  createBackend(ctx), restoreSession(ctx), the catch-all for server-only routes
      relay.ts  registerServiceWorker(), listenForApiRequests(backend) on chunkmirage's route(); Ask/Reply/Later are chunkmirage's types
      router.ts, sse.ts, yaml.ts (the export), serving.ts (serve and re-serve the session's models, and stop serving one)
      finetune-jobs.ts (a finetune run as the dashboard's job), ai-annotate-job.ts (one AI annotation, staged for review)
      routes/{data,pipeline,models,config,finetune,good-regions,view-labels,ai-annotate}.ts
    state/                   openSession(): SessionStore (sessionStorage: the tab is the server's process lifetime); db.ts + local-store.ts
                             for picked directories; paint-store.ts + paint-route.ts: the annotation volumes' files and the worker's paint/ route
    finetune/                types.ts (the seams), volume.ts (planVolume, the zarr v2 files), samples.ts, target-call.ts, onnx-proto.ts,
                             onnx-head.ts, features.ts, head-trainer.ts, distillation.ts, good-regions.ts, run.ts (runFinetune),
                             zip.ts (the paint's download), modes.ts (what a manifest offers), augment.ts (the whole network's)
      full/                  the whole network in TF.js: cellmap-tfjs-train.ts (Marwan's runtime, the weights read and written),
                             graph.ts (the training graph, its tile and block sides), net3d.ts (a 3D model's network in a GPU's
                             memory), files.ts, tiles.ts (2D tiles, 3D blocks), tfjs.ts (lazy TF.js), loss.ts (the head's loss
                             in TF.js), run-full.ts
    ai-annotate/             AI-assisted annotation (above): types.ts, geometry.ts, plane.ts, mask.ts, labels.ts, fill.ts, prompts.ts,
                             organelles.ts (the pure ports), images.ts (the canvas codec), gemini.ts, fake.ts, providers.ts,
                             credentials.ts, usage.ts, audit.ts; labels.ts and fill.ts are Label it's too
    label-it/                Label it (above): segment.ts (the mutex watershed, the probability's call), prediction.ts (the served
                             level, the placement's call, the output under a box)
    run-region/              Run on a region (above): region.ts (the box on each model's level, the run_region layer), run.ts
                             (the run, its progress and provenance), output.ts (the folder, the zip, the saved layer), panel.ts (the tab)
    local-folder/            a folder on this machine: index.ts (the local URL and its key, the pick/drop flow,
                             the boot-time permission check), panel.ts (the connect panel's DOM: button, drop target, prompt)
    infer/                   createInference(): Inference — the Engine, one serveStages() view per model
      engine.ts (the views, the stats, a region written out: materialize), stages.ts (the model and postprocess stages, their keys),
      output.ts (a box of a model's own output from its chunks, readOutput's),
      norms.ts, post.ts,
      expression.ts (Lambda steps), geometry.ts (the manifest's blocks, halo, ratio, levels), dtype.ts (astype)
    shared/                  args.ts (__CFLOW_ARGS__ codec, digest), model-info.ts
    models/                  createCatalog(): Catalog — Hugging Face manifests, variant choice (the largest block that fits for a 3D model, the one nearest 512 a side for a 2D one), byte cache keyed by URL and ETag, finetuned.ts (the models finetuned here)
    viewer/                  createViewer(): ViewerController — iframe (index.ts), layers.ts (port of
                             viewer/layers.py), display.ts (shaders, ranges), state.ts, share.ts (the viewer's state: the tab's copy, a link's hash)
  test/
    unit/                    vitest: the backend's routes over the fakes in backend-fakes.ts, and what
                             parity does not cover (the ops that parity checks have no unit file);
                             pyodide.ts runs cellmap_flow.portable in node for the routes that call it
    parity/                  the TypeScript ports against test/fixtures/*.json, which
                             test/fixtures/make_*.py write from the Python classes
    e2e/                     Playwright: smoke, downloads, recent datasets, finetune, AI-annotate (E2E=1 npm run e2e)
```

## Running it

```
cd web
npm install
npm run fetch:chunkmirage                 # once, and after bumping chunkmirage.lock
bash scripts/fetch-neuroglancer.sh        # once (a minute or two): the fork that paints, into web/ng/
PYTHON=../.pixi/envs/default/bin/python npm run render   # after a template change
PYTHON=../.pixi/envs/default/bin/python npm run gen      # after changing a request model, a route or a shared constant
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
normalizer/postprocessor ports, the local-directory store and the viewer
state in a `#!` hash come from Marwan Zouinkhi's serverless prototype (his commits on this
branch, under `web/` before this restructuring), and the models are his
exports on Hugging Face. The engine, the reading and the relay are
chunkmirage's.
