---
title: cellmap-flow
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
suggested_hardware: cpu-basic
---

# cellmap-flow inference backend (Hugging Face Space)

Hosts a cellmap-flow virtual-zarr server. The browser-side dashboard at
[cellmap-flow's `browser/` app](../browser/) points Neuroglancer at this
Space's URL; chunks are computed here on demand.

## Configure

In **Settings → Variables and secrets**, set:

| Variable           | Example                                                                                  |
| ------------------ | ---------------------------------------------------------------------------------------- |
| `CELLMAP_HF_REPO`  | `cellmap/jrc_mus-livers_16nm_to_8nm_mito`                                                |
| `CELLMAP_HF_NAME`  | `mito` (optional, defaults to the last segment of the repo)                              |
| `CELLMAP_DATASET`  | `s3://janelia-cosem-datasets/jrc_mus-liver/jrc_mus-liver.zarr/recon-1/em/fibsem-uint8/`  |

Restart the Space.

## Endpoints

Once running, the Space exposes cellmap-flow's standard virtual-zarr API:

- `https://<your-space>.hf.space/<dataset>/.zattrs` — top-level OME-Zarr attrs
- `https://<your-space>.hf.space/<dataset>/s<scale>/.zarray` — array metadata
- `https://<your-space>.hf.space/<dataset>/s<scale>/<z>.<y>.<x>[.<c>]` — chunk bytes

`<dataset>` is whatever path slug the client uses; cellmap-flow's server
treats it as a unique ID and rebinds to the configured input zarr.

## Plugging into the browser app

In the dashboard, paste your Space's URL into the **Inference server URL**
field. The browser will configure Neuroglancer's source URL as
`zarr://<your-space>.hf.space/<dataset>/`. No browser-side inference; this
Space does it.

## Free tier caveats

- **2 vCPU, 16 GB RAM, no GPU.** A 256³ chunk through a real cellmap UNet
  takes minutes on CPU. Move to a paid GPU tier or Colab GPU for speed.
- **Sleeps after idle.** First chunk after sleep takes ~30 s to wake.
- **Concurrency**: the server is single-threaded for inference. Multiple
  users queue.

## Forking for a different model

Each Space hosts one model + one dataset. To serve a different combo, fork
this Space and change `CELLMAP_HF_REPO` / `CELLMAP_DATASET`.
