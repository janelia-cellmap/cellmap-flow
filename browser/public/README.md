---
title: cellmap-flow
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: static
pinned: false
short_description: Run BMZ models on EM data in your browser via WebGPU
---

# cellmap-flow (browser)

Static frontend for cellmap-flow with in-browser BMZ inference via
ONNX Runtime Web on WebGPU. No server needed; the user's GPU runs the
model.

**Entry points:**

- `/dashboard.html` — full cellmap-flow dashboard with Input /
  Postprocess pipeline editor. Pick a BMZ model, paste a zarr URL,
  click Open in NG.
- `/index.html` — minimal page with a "Run on test crop" demo for the
  in-browser model.

**Currently supported:**

- 2D BMZ models exported to static-shape ONNX (currently
  `hiding-blowfish` aka `EnhancerMitochondriaEM2D`).
- 3D models are blocked on ORT Web's WebGPU EP not yet implementing
  Conv3D.

See [the upstream repo](https://github.com/janelia-cellmap/cellmap-flow/tree/browser-inference/browser)
for the source.
