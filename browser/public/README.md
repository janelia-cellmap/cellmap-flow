---
title: cellmap-flow
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: static
pinned: false
short_description: cellmap-flow dashboard + Neuroglancer
---

# cellmap-flow (browser)

Static frontend for cellmap-flow. Bundles Neuroglancer + the cellmap-flow
dashboard chrome (Input / Postprocess / Models tabs). Points NG at a
cellmap-flow inference server (e.g. running on a Colab session, HF
Docker Space, or your workstation) supplied via URL params.

**Entry point:** `/dashboard.html`

Accepted URL params:

| param        | meaning |
| ------------ | ------- |
| `backend`    | inference server URL (HF Space, Colab + cloudflared, etc) |
| `server`     | alias for `backend` |
| `dataset`    | dataset slug — used as the layer path on the inference server |
| `data`       | alias for `dataset` |
| `raw`        | source EM zarr URL (rendered as a "raw" layer alongside inference) |
| `voxelSize`  | `Z,Y,X` in nm — force raw layer + NG world to this voxel size |

See [the upstream repo](https://github.com/janelia-cellmap/cellmap-flow/tree/browser-inference/browser)
for the source and `notebooks/cellmap-flow-colab.ipynb` for the
recommended Colab launcher.
