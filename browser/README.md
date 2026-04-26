# cellmap-flow (browser)

Browser front-end for cellmap-flow. Renders cellmap-flow's actual dashboard
templates and embeds Neuroglancer; inference happens on an external
cellmap-flow server (Hugging Face Space or Colab + ngrok).

## Why no in-browser inference

Tried it. ONNX Runtime Web's WebGPU EP doesn't yet implement Conv3d; every
real cellmap-flow UNet hits "Only Conv1d and Conv2d are supported." 2D-only
demo models worked, but real models can't run client-side until ORT Web
catches up. So the browser app now points Neuroglancer at a remote
cellmap-flow inference server.

## Architecture

```
┌─────────────────────────┐       ┌──────────────────────────────┐
│  browser (this app)     │       │  cellmap-flow server          │
│                         │       │  (HF Space / Colab + ngrok)   │
│  - cellmap-flow         │       │                              │
│    dashboard chrome     │       │  - cellmap_flow_server        │
│  - bundled Neuroglancer │ HTTP  │    huggingface --repo X       │
│  - HF metadata loader   │ ───▶ │    -d <zarr URL>              │
│    (display only)       │       │                              │
│                         │       │  - virtual zarr at:           │
│  zarr://<server>/<ds>/  │       │    /<ds>/.zattrs              │
└─────────────────────────┘       │    /<ds>/sN/.zarray           │
                                  │    /<ds>/sN/Z.Y.X[.C]        │
                                  └──────────────────────────────┘
```

## Run the front-end

```bash
cd browser
npm install
npm run start          # builds + serves; opens at http://localhost:4173
```

> **Why not `npm run dev`?** Vite dev mode doesn't correctly serve
> Neuroglancer's worker bundles via the dep optimizer. `npm run start`
> uses `vite build && vite preview` instead.

## Stand up an inference server

Pick one:

### Hugging Face Space (free, slow, always-on)

See [`../hf-space/`](../hf-space/). Fork the Space template, set
`CELLMAP_HF_REPO` and `CELLMAP_DATASET` in the Space's Settings, and the
public URL `https://your-space.hf.space` is your inference server.

Free CPU tier: ~minutes per chunk on 3D UNets. Fine for demos. Sleeps
after idle (~30 s wake-up).

### Google Colab (free GPU, ephemeral)

Open [`notebooks/cellmap-flow-colab.ipynb`](../notebooks/cellmap-flow-colab.ipynb)
in Colab. Run all cells; copy the printed ngrok URL.

Free T4 GPU: real-time-ish inference. Catches: only works while *your*
Colab session is alive.

### Local cellmap-flow server

If you have a workstation with a GPU and `cellmap-flow` installed:

```bash
cellmap_flow_server huggingface \\
  --repo cellmap/jrc_mus-livers_16nm_to_8nm_mito \\
  -d s3://janelia-cosem-datasets/.../jrc_mus-liver.zarr/.../fibsem-uint8/ \\
  --port 8765
```

Tunnel it (e.g. via ngrok or `cloudflared tunnel`) and use that URL.

## Use it

1. Start the inference server (above).
2. Open this app's dashboard.
3. Paste the server URL + a dataset slug; click **Open in NG**.

## Layout

- `index.html` / `dashboard.html` — entry pages. Dashboard is the main one;
  it's rendered from `cellmap_flow/dashboard/templates/` via `scripts/render-dashboard.py`.
- `src/main.ts` — minimal entry for `index.html`.
- `src/dashboard-shim.ts` — wires the cellmap-flow dashboard's existing JS
  to our HF-Space/Colab flow (replaces Flask `/api/*` calls with browser
  equivalents, intercepts the HF accordion, mounts Neuroglancer).
- `src/ng-entry.ts` — Neuroglancer registrations + `setupDefaultViewer`,
  bundled by Vite from the `neuroglancer` npm package.
- `src/hf.ts` — fetch HF metadata.json for display.
- `scripts/render-dashboard.py` — renders cellmap-flow's Jinja templates to
  static HTML at build time.
