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
cluster account.

It is the same dashboard as the Flask app's. The templates
(`cellmap_flow/dashboard/templates`) and the dashboard's JavaScript and CSS
(`cellmap_flow/dashboard/static`) are used as they are; `build/render.py`
renders the templates once, at build time, and the TypeScript in `src/`
answers the dashboard's `fetch("/api/...")` requests in place of the Flask
routes, with the same request and response JSON. The controls only a server
can serve (LSF settings, Job Logs, the Local Catalog of cluster paths, the
Finetune and Review tabs) are hidden; everything else is the dashboard you
know. `ARCHITECTURE.md` describes how the parts fit together.

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

## Credits

The browser mode is a port of
[Marwan Zouinkhi](https://github.com/mzouink)'s serverless prototype of
cellmap-flow: the inference loop (read a block with context, normalize, one
`session.run` at a time on WebGPU with a WebAssembly fallback, postprocess,
encode as zarr), the ports of the normalizers and postprocessors, the zarr
reader and the local-directory store all come from it, as do the ONNX
exports on Hugging Face and their benchmarks. His commits are on this branch;
the remaining files of the prototype are in `legacy/` and are deleted as each
is ported.

The on-demand chunk engine (the demand queue, the cache, the service worker
relay and the OME-Zarr reader) is [chunkmirage](https://github.com/yuriyzubov/chunkmirage)'s
browser engine, which `scripts/fetch-chunkmirage.sh` clones at the commit
`chunkmirage.lock` names.
