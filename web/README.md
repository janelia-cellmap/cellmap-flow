# cellmap-flow in the browser

This directory is cellmap-flow's browser mode: the dashboard as a static page,
with model inference done in the page itself. There is no server. The page
reads the raw volume straight from its URL (an OME-Zarr on https, S3 or GCS),
runs the model with [onnxruntime-web](https://onnxruntime.ai/) on the
machine's GPU through WebGPU (or on its CPU through WebAssembly when there is
none), and serves the predictions to Neuroglancer as a zarr array through a
service worker, chunk by chunk, as the viewer asks for them. The models are
ONNX exports of cellmap's networks, fetched from Hugging Face
([cellmap/onnx-web-models](https://huggingface.co/cellmap/onnx-web-models))
and cached in the browser. Anyone with a recent browser can open a volume,
tick a model and look at its predictions, with nothing to install and no
cluster account: it is live at **https://janelia-cellmap.github.io/cellmap-flow/app/**,
deployed from every push to `main` ([Deployment](#deployment)), and the first
place to try cellmap-flow and to test a change.

It is the same dashboard as the Flask app's. The templates
(`cellmap_flow/dashboard/templates`) and the dashboard's JavaScript and CSS
(`cellmap_flow/dashboard/static`) are used as they are; `build/render.py`
renders the templates once, at build time, and the TypeScript in `src/`
answers the dashboard's `fetch("/api/...")` requests in place of the Flask
routes, with the same request and response JSON. The controls only a server
can serve (LSF settings, Job Logs, the Local Catalog of cluster paths, the
Review tab, and the Finetune tab's crops, seeds and AI annotation) are
hidden; everything else is the dashboard you know, finetuning included
([below](#finetuning-in-the-browser)). `ARCHITECTURE.md` describes how the
parts fit together.

## Running it locally

You need Node 22, and a Python with cellmap-flow's dependencies for the one
build step that renders the templates (the repository's pixi `default`
environment has them: `pixi install` at the repository's root).

```bash
cd web
npm install
npm run fetch:chunkmirage      # once, and again after bumping chunkmirage.lock
PYTHON=../.pixi/envs/default/bin/python npm run render   # again after a template change
npm run dev                    # then open http://localhost:5173/
```

`npm run render` writes `web/index.html` (it is not committed; the build
renders it). `npm run dev` serves the page, the dashboard's static files
under `./static/` and, in development, relays `./ng/` from
neuroglancer-demo.appspot.com: upstream Neuroglancer, which cannot paint.
`scripts/fetch-neuroglancer.sh` builds the fork that paints
(briossant/neuroglancer, the commit `neuroglancer.lock` names, the same one
pixi.toml pins for the server mode) into `web/ng/`, and the dev server serves
that instead when it exists; it takes a few minutes and about 700 MB of npm
packages, once per commit. The deployed site always carries it.

On the first visit the page reloads once: the service worker has to take
control of the page before the viewer can ask it for chunks.

Checks: `npm run typecheck` (TypeScript, the page and the service worker
separately), `npm test` (vitest unit tests in `test/`), and from the
repository's root `python -m pytest tests/web` for the rendered page.
`npm run build` writes the deployable site to `web/dist/`.

## A link that opens a volume and runs models

The page's query can name what Connect and Submit would be used for:

```
https://janelia-cellmap.github.io/cellmap-flow/app/?dataset=<url>&model=<name>&model=<name>
```

`dataset` is the volume's URL; each `model` is a model as the Models tab
lists it (`setup_53`) or its repository path
(`cellmap/onnx-web-models/setup_53`), repeated or comma-separated. At boot
(`src/main.ts`, `src/link.ts`) they go into the tab's session, the session's
restore opens the volume and serves the models, and the query is taken off
the address bar, so a later tab completed from the URL starts fresh. A model
the catalog does not know is logged and left out. The README's example link
opens an OpenOrganelle volume with `setup_53` running.

**Share** in the header (`src/share-link.ts`) copies such a link to what the
tab has open, with the viewer's `#!` state after it: the view (position,
zoom, layout, each layer as shown with its shader), whose model layers name
the chain they were drawn through; the recipient's page serves the models
through that chain and its Input and Postprocess lists show it (`main.js`'s
`showChain`, called by `src/main.ts` when the session's chain is not the
rendered default). The address bar never carries any of it: when the page's
URL mirrored the viewer's, Chrome's omnibox completed a typed address to the
last view and a new tab came back with it.

## Opening a folder on your machine

In Chrome or Edge the connect panel has a **Browse local folder…** button,
and takes a folder dropped on it from the file manager: an OME-Zarr, zarr or
N5 directory on your machine opens with no server and no https, at
`http://localhost` as well as on the deployed site. The browser asks you once
to allow reading the folder (the File System Access API); nothing is
uploaded. The page reads the files itself and serves them to the viewer and
the models under its own `local/<key>/` URL, through the service worker.

The page reloads once when a volume opens, and after a reload (or the next
day) Chrome has dropped its permission to read the folder, so the panel
shows an **Allow access to <folder>** button first; click it, and the viewer
comes back. Choosing "Allow on every visit" in Chrome's prompt skips this.
Firefox and Safari have no directory picker, so the button is not shown
there, and a link to the page with a local folder open works only in the
browser that picked the folder.

## Finetuning in the browser

The Finetune tab works as it does on the server, with the browser in the
server's and MinIO's places. *New Volume* plans an annotation volume over the
open dataset on the grid the model's predictions are served on
(`src/finetune/volume.ts`, the port of `finetune/session/volume.py`: one chunk
per model output block, labels 0 unpainted, 1 background, 2 and up objects)
and writes its zarr v2 metadata into the paint store, IndexedDB
(`src/state/paint-store.ts`). The service worker serves the volume at
`paint/<volume>/…` under the page as an S3-compatible endpoint (GET, HEAD,
PUT, DELETE; `src/state/paint-route.ts`), and the paint layer added to the
viewer is the Neuroglancer fork's writable zarr source on
`s3+https://<host>/<page>/paint/<volume>/annotation`, which PUTs each chunk
you paint there, as it PUTs to MinIO in server mode. Chunks are uncompressed,
so nothing decodes anything.

*Start Finetuning* trains the model's head (`src/finetune/`): the last 1×1
convolution of the ONNX graph, found by walking back from the output
(`onnx-head.ts`), is given its input tensor as a second output
(`onnx-proto.ts` splices the graph at protobuf field level, no dependency),
the model runs once per painted chunk on the block the model stage would feed
it (`samples.ts`, through the volume's normalizers) for its logits and
features (`features.ts`), and the head's weights are trained from the model's
own on the painted voxels' features with Adam on masked, class-balanced
BCE-with-logits (`head-trainer.ts`; the targets are `targets.ts`, the port of
`target_transforms.py`, checked against the Python classes). The trained
weights are written back into a copy of the model's initializers, and the
result is an ordinary model entry (`src/models/finetuned.ts`, repo
`finetuned`, kept in IndexedDB with its bytes): served in place of the base
when *Auto-serve* is on, listed on the Models tab, downloadable as `.onnx`.
The job routes answer as the server's do (`src/backend/routes/finetune.ts`,
`finetune-jobs.ts`), so the tab's own job card, loss plot and log stream
follow the run unchanged. Why the head and not the network: onnxruntime-web
trains nothing on the GPU; a WebGPU autograd library would have to rebuild
the network, which is a later step behind the same `RunFinetune` seam
(`src/finetune/types.ts`).

## Browser requirements

- **WebGPU.** Chrome or Edge (any current version), or Safari 26 and later.
  Firefox's WebGPU is behind a flag and not tested. Without WebGPU the page
  still works, on the CPU through WebAssembly: tens of seconds per chunk
  instead of well under one.
- **The URL must be `http://localhost:<port>/` or `https://`.** WebGPU is only
  available in a secure context, and onnxruntime-web does not fail without
  it: it silently falls back to WebAssembly, so the page runs about fifty
  times slower with no error to show for it. A VS Code tunnel host, a
  workstation's IP address or an `http://` hostname all lose WebGPU. When
  developing on a remote machine, forward the port and open `localhost`.
- **Memory.** The largest models' fp32 variants need about half a gigabyte of
  GPU memory for one chunk; the page picks a smaller variant (fp16, int8
  weights) when the adapter's limits call for it. Model files are 11–43 MB
  each and are kept in IndexedDB, because Hugging Face rate-limits anonymous
  downloads.
- **Cross-origin data.** The raw volume is fetched by the page, so its server
  must allow it with CORS (`Access-Control-Allow-Origin`). Public S3 and GCS
  buckets and GitHub Pages do; a plain http server on your machine may need a
  flag.
- **Data on a private network.** To Chrome, a host with a private address
  (`10.x`, `192.168.x`, a `.local` name) is on the user's *local network*,
  which a public site such as the deployed one may fetch from only with the
  user's permission (Chrome's Local Network Access). Chrome can ask for it
  when the page itself makes a request to such a host, and refuses one made
  by a service worker outright unless the permission was given before. So
  the service worker (`src/sw.ts`) leaves a volume's metadata requests, the
  first of any volume, to the page, as Neuroglancer's site reads a volume;
  on the deployed site a cellmap-vm1 volume had failed with "Failed to
  fetch" while Neuroglancer read it. When a fetch still gets no answer, the
  connect panel's error says what to do (`src/backend/routes/data.ts`). The
  permission is per site: in the site-information panel beside the address
  bar, or at `chrome://settings/content/localNetworkAccess`. A page on
  `http://localhost` is not a public site, so there is no such check there.

## Deployment

The site is published to GitHub Pages beside the Sphinx docs: the docs at
`https://janelia-cellmap.github.io/cellmap-flow/`, the dashboard at `/app/`
and its Neuroglancer build at `/app/ng/`, from one `gh-pages` branch that one
workflow, `.github/workflows/pages.yml`, builds and pushes on every push to
`main` that touches `web/`, the package, the docs or the README. It runs the
same steps as above (`fetch:chunkmirage`, `fetch-neuroglancer.sh`, `render`,
`build`) and copies `web/dist/` to the site's `app/`. Neuroglancer has to be
served from the same origin as the page because a service worker only sees
requests from pages on its own origin; that is why the site carries its own
build of the viewer rather than embedding neuroglancer-demo.appspot.com.

`vite.config.ts` builds with `base: "./"`, so the same build works under
`/app/` on Pages and at the root of a local preview (`npm run preview`). The
service worker keeps the fixed name `sw.js`, beside the page, because
chunkmirage's engine registers it by that name with scope `./`.

## Testing

```
cd web
npm run typecheck && npm test            # TypeScript, and the unit and parity suites (vitest)
E2E=1 npm run e2e                        # the browser smoke test: a public volume, a model, one chunk
```

The parity suites check the TypeScript ports of the normalizers, postprocessors,
the finetune's target transforms and the `__CFLOW_ARGS__` codec against
fixtures that `test/fixtures/make_fixtures.py` writes from the Python classes;
regenerate them with that script (the pixi python) when the Python changes,
and `tests/web` (pytest) fails while they are stale. The ONNX head editing
and the trainer run against `test/fixtures/head.onnx`, a small graph shaped
like the exports' tails that `test/fixtures/make_onnx_fixture.py` writes.

The smoke test runs headless Chromium through Playwright (`npx playwright
install chromium` once; `CHROME_PATH=/path/to/chrome` uses a Chromium of your
own). Headless Chromium has no WebGPU, so it runs on the WASM provider: about
half a minute for the first chunk.

## Credits

The browser mode is a port of
[Marwan Zouinkhi](https://github.com/mzouink)'s serverless prototype of
cellmap-flow: the inference loop (normalize, one `session.run` at a time on
WebGPU with a WebAssembly fallback, postprocess; now two stages of
chunkmirage's engine, `src/infer/stages.ts`), the ports of the normalizers
and postprocessors and the local-directory store all come from it, as do the
ONNX exports on Hugging Face and their benchmarks. His commits are on this
branch; the remaining files of the prototype are in `legacy/` and are deleted
as each is ported.

The on-demand chunk engine (the OME-Zarr reader, each chunk's input box with
its padding, the per-stage cache, the demand queue, the chunk cache, the
service worker relay and the page's route table) is
[chunkmirage](https://github.com/JaneliaSciComp/chunkmirage)'s browser
engine, which `scripts/fetch-chunkmirage.sh` clones at the commit
`chunkmirage.lock` names; `ARCHITECTURE.md` says what is the engine's and
what is this app's.
