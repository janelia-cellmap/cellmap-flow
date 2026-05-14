# Deploying the cellmap-flow frontend

The `browser/` app is a static Vite build (`browser/dist/`). It does no
inference — it just loads Neuroglancer and the cellmap-flow dashboard
chrome, then points NG at *some* zarr URL you give it. The actual
inference happens on whatever backend the user pastes into the
"Inference server URL" field.

## Two-piece architecture

```
┌──────────────────────────────┐   HTTP    ┌────────────────────────────────┐
│  Static frontend (this app)  │ ────────► │  cellmap-flow inference server │
│  e.g. Cloudflare Pages /     │           │  e.g. HF Space (hf-space/)     │
│  Vercel / Netlify / GH Pages │           │       Colab + cloudflared      │
│                              │           │       your workstation         │
│  - Neuroglancer + UI         │           │  - cellmap_flow_server …       │
│  - dashboard pipeline UI     │           │  - serves /<ds>/.zattrs etc.   │
└──────────────────────────────┘           └────────────────────────────────┘
```

Frontend hosting is free and stateless; backend hosting is whatever you
pick.

## HuggingFace Spaces (Static SDK) — recommended

Best fit because:
- No per-file size limit (the 116 MB `hiding-blowfish` ONNX would be
  blocked by Cloudflare Pages' 25 MB free-tier limit and exceeds GitHub
  Pages' 100 MB limit).
- Respects the `_headers` file we need for COOP/COEP / WebGPU.
- Free, no credit card.
- Same org as your model weights — natural discovery surface.

### One-time setup

1. On `huggingface.co` → **New Space** → name it (e.g.
   `cellmap-flow-demo`), choose **Static** SDK, set to Public.
2. Locally:
   ```bash
   pip install -U huggingface_hub
   huggingface-cli login   # paste a write token from huggingface.co/settings/tokens
   ```

### Deploy

```bash
cd browser
npm run build
huggingface-cli upload <your-username>/cellmap-flow-demo \
    dist/ . \
    --repo-type=space \
    --commit-message "deploy $(date -Iseconds)"
```

(`huggingface_hub` auto-handles LFS for the 111 MB ONNX file.)

After ~1 min the Space URL `https://huggingface.co/spaces/<you>/cellmap-flow-demo`
serves the dashboard at `…/dashboard.html`.

### Re-deploying

Re-run the `huggingface-cli upload …` command after `npm run build`. The
CLI computes diffs and only re-uploads changed files.

### Embedding the backend URL

Same `?backend=…&dataset=…&model=…&raw=…` query params work on the HF
Spaces URL, e.g.

```
https://huggingface.co/spaces/<you>/cellmap-flow-demo/dashboard.html?model=hiding-blowfish&dataset=s3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.zarr/recon-1/em/fibsem-uint8
```

## Cloudflare Pages

Cloudflare Pages free tier: unlimited bandwidth, unlimited requests, 500
builds/month, free SSL, free custom domains. Perfect fit.

### One-time setup via the dashboard

1. Push the repo to GitHub (or GitLab, if you prefer).
2. In Cloudflare → **Workers & Pages → Pages → Create application →
   Connect to Git → pick the repo**.
3. Build configuration:
   - **Framework preset:** None
   - **Build command:** `cd browser && npm install && npm run build`
   - **Build output directory:** `browser/dist`
   - **Root directory:** *(leave blank, defaults to repo root)*
4. Save and deploy. First build takes ~2 minutes.

You'll get a `https://<project>.pages.dev` URL. Every push to `main`
rebuilds automatically.

### Per-deploy environment

The frontend doesn't need any secrets — it does no work and talks to a
backend the user supplies at runtime. If you want to **hardcode a default
backend** so visitors don't have to type a URL, see "Embedding the
backend URL" below.

### Headers

`browser/public/_headers` sets `Cross-Origin-Opener-Policy: same-origin`
and `Cross-Origin-Embedder-Policy: credentialless`. Cloudflare Pages
respects this format natively (same syntax as Netlify). These are
required for the in-browser BMZ demo path; harmless for the
server-backed path.

## Vercel / Netlify

Same as Cloudflare Pages, with the same build command + output dir. The
`_headers` file works on Netlify out of the box; for Vercel you'd need a
`vercel.json` (not included here — the project doesn't need either, pick
whichever host you prefer).

## GitHub Pages

Works too, but doesn't honor `_headers` so the in-browser BMZ path loses
cross-origin isolation. Server-backed flow still works. Use a GitHub
Action that runs `npm run build` and deploys `browser/dist/` to
`gh-pages`.

## Embedding the backend URL

Users can paste a backend URL into the dashboard, but for a curated demo
it's nicer to pre-fill. The dashboard accepts URL query params, so a
shareable link is:

```
https://<your-pages-url>/dashboard.html
  ?backend=https://<your-space>.static.hf.space
  &dataset=<dataset-slug>
```

For an in-browser BMZ demo:

```
https://<your-pages-url>/dashboard.html
  ?model=hiding-blowfish
  &dataset=s3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.zarr/recon-1/em/fibsem-uint8
```

Supported query params:

| param      | meaning                                                                                          |
| ---------- | ------------------------------------------------------------------------------------------------ |
| `backend`  | server URL (pre-fills "Inference server URL")                                                    |
| `server`   | alias for `backend`                                                                              |
| `dataset`  | dataset slug for server-backed; full zarr URL for in-browser BMZ                                 |
| `data`     | alias for `dataset`                                                                              |
| `raw`      | http(s) source zarr URL — mounts as a "raw" image layer alongside inference                     |
| `model`    | in-browser BMZ model id (e.g. `hiding-blowfish`); auto-switches to BMZ mode                      |
| `hf`       | display-only HF model repo                                                                       |

Query params take precedence over `localStorage`.

## Backend hosting options

| host                   | cost      | speed (2D U-Net 512²) | concurrency | always-on  |
| ---------------------- | --------- | --------------------- | ----------- | ---------- |
| HF Space CPU Basic     | $0        | 1–5 s/chunk           | ~4 threads  | yes (~30 s wake from idle) |
| HF Space ZeroGPU       | $0 + Pro  | ~100 ms/chunk         | shared queue| yes        |
| HF Space dedicated GPU | ~$0.40/hr | ~50 ms/chunk          | many        | yes        |
| Colab GPU              | $0        | ~100 ms/chunk         | 1 user      | tab-bound  |
| Your workstation       | $0        | depends on hardware   | many        | yes        |

For all backends, point at `hf-space/` (Docker template) or
`notebooks/cellmap-flow-colab.ipynb` (Colab) — both already wired to run
`cellmap_flow_server` with either a cellmap HF model or a BMZ model.
