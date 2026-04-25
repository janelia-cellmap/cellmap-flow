# cellmap-flow (browser)

In-browser cellmap-flow. Open Neuroglancer; every chunk it requests is the
output of an ONNX model running in your tab — no server. Mirrors the Python
pipeline (`server.py` + `inferencer.py`) entirely client-side.

## How it works

1. **Service worker** at `/sw.js` intercepts fetches under `/vz/*`.
2. Page holds an **ONNX Runtime Web** session and an open zarr (raw input).
   When the SW hands a chunk request to the page:
   - Compute write ROI (output voxels) → world units → expand by `context =
     (read_shape - write_shape) / 2` → read ROI in input voxels.
   - Read raw subvolume (zero-pad at edges).
   - Apply `normalize`.
   - Run model in the spec's `tensorLayout`.
   - Reshape output to `(C, Dz, Dy, Dx)`.
   - Apply `postprocess` ops.
   - Encode as Zarr v2 chunk in `outputDtype`.
3. **Neuroglancer is bundled inline** from the `neuroglancer` npm package
   (Vite). Pointed at `zarr://<origin>/vz/`, it fetches OME-Zarr metadata +
   chunks from the SW like a real volume on disk.

## Run locally

```bash
# one-time: build the demo ONNX + matching spec
python ../scripts/export_demo_onnx.py

# build + serve (production preview)
cd browser
npm install
npm run start
```

Open the printed URL, paste your zarr URL + ONNX URL + spec URL, click
"Activate & open in NG".

> **Why `npm run start` (production preview), not `npm run dev`** — Vite
> dev-mode doesn't correctly serve Neuroglancer's `chunk_worker.bundle.js`
> via its dep optimizer. `vite build && vite preview` works. Iteration is a
> build per change.

## Model spec (`spec.json`)

Browser-side analogue of cellmap-flow's Python `model_spec.py`. All shapes
in **world units** (e.g. nanometers).

```jsonc
{
  "inputVoxelSize":  [8, 8, 8],          // input zarr nm/voxel
  "outputVoxelSize": [16, 16, 16],       // model output nm/voxel
  "readShape":       [1728, 1728, 1728], // input ROI in nm  (216 vox * 8)
  "writeShape":      [1088, 1088, 1088], // output ROI in nm (68 vox * 16)
  "inputDtype":      "float32",          // dtype of values fed to ONNX
  "outputDtype":     "uint8",            // dtype of chunk bytes
  "outputChannels":  1,
  "blockShape":      [68, 68, 68],       // chunk shape in OUTPUT voxels (z,y,x)
  "tensorLayout":    "NCDHW",            // or "NDHWC" or "BatchZ_NCHW"

  "normalize": { "type": "scale_offset", "scale": 0.00392156862745, "offset": 0 },
  // alternatives:
  //   { "type": "identity" }
  //   { "type": "mean_std", "mean": 128, "std": 64 }
  //   { "type": "minmax", "min": 0, "max": 65535 }

  "postprocess": [
    { "type": "clip", "min": 0, "max": 1 },
    { "type": "scale", "factor": 255 }
    // also: { "type": "offset", "value": 5 }
    //       { "type": "channel", "index": 0 }
    //       { "type": "threshold", "value": 0.5, "below": 0, "above": 1 }
  ]
}
```

### Translating a Python `model_spec.py`

| Python                                     | JSON                          |
| ------------------------------------------ | ----------------------------- |
| `voxel_size` / `input_voxel_size`          | `inputVoxelSize`              |
| `output_voxel_size`                        | `outputVoxelSize`             |
| `read_shape` (already in world units)      | `readShape`                   |
| `write_shape` (already in world units)     | `writeShape`                  |
| `output_channels`                          | `outputChannels`              |
| `block_shape[:3]` (in output voxels)       | `blockShape`                  |
| Model input shape `(1, 1, D, H, W)`        | `tensorLayout: "NCDHW"`       |
| Per-Z 2D model `(Z, 1, H, W)`              | `tensorLayout: "BatchZ_NCHW"` |
| Channels-last (uncommon)                   | `tensorLayout: "NDHWC"`       |
| Custom `process_chunk` Python function     | not portable — express as `normalize` + `postprocess` |

If the Python spec applies `(raw - mean) / std` before the model: `normalize:
{ type: "mean_std", mean, std }`. If it `clip(0, 1) * 255` after: `postprocess:
[{ type: "clip", min: 0, max: 1 }, { type: "scale", factor: 255 }]`.

## Layout

- `index.html` — single-page app (zarr-url, onnx-url, spec-url + Activate).
- `src/main.ts` — wires up activate flow.
- `src/model-spec.ts` — `ModelSpec` type, normalize/postprocess executors,
  chunk encoder, tensor reshape utilities.
- `src/virtual-zarr.ts` — read-ROI/context expansion, runs the model,
  postprocess + chunk encode. The JS analogue of `Inferencer.process_chunk_basic`.
- `src/onnx-session.ts` — lazy ORT session create + serialized `runModel`.
- `src/zarr-client.ts` — zarrita wrappers for the input zarr.
- `src/sw-register.ts` — registers `/sw.js`, reloads on first install.
- `src/ng-entry.ts` — NG kvstore/datasource/layer registrations + `setupDefaultViewer`.
- `public/sw.js` — service worker. Forwards `/vz/*` requests to the page.

## Limitations / next steps

- LAN deployments need a secure context (use SSH tunnel to localhost or
  HTTPS). Service workers refuse to register otherwise.
- WASM ORT does not implement Conv3d. 3D UNets (most cellmap-flow models)
  require WebGPU. WebGPU is enabled by default in modern Chrome.
- Single-channel input only in this version. Multi-channel input
  (`(Cin, D, H, W)`) is straightforward but not yet wired through.
- No mirror-padding or smoothing at volume edges; we zero-pad. Most models
  tolerate this.
