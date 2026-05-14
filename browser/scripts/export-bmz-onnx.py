"""Download a BioImage Model Zoo torchscript model + test tensors, export
to ONNX at a fixed shape, and verify the ONNX numerically matches the
reference test_output. Writes artifacts to browser/public/bmz/<id>/.

The browser loads:
    /bmz/<id>/model.onnx          ONNX (static shape)
    /bmz/<id>/test_input.bin      raw float32 bytes, shape from manifest
    /bmz/<id>/test_output.bin     raw float32 bytes (reference, for sanity)
    /bmz/<id>/manifest.json       { shape_in, shape_out, normalization, ... }

Skips re-download/re-export if the outputs already exist. Re-run with
`--force` to redo it.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

# Skip cleanly if the in-browser-inference deps aren't installed. The
# server-backed flow (HF Space / Colab + cellmap_flow_server) doesn't need
# the browser-side ONNX export, so the npm build shouldn't fail just
# because torch/onnx aren't available in this env.
try:
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
except ImportError as e:
    print(
        f"[export-bmz-onnx] skipping: missing optional dep ({e.name}). "
        "Install `torch onnx onnxruntime` to export BMZ models for the "
        "in-browser inference path.",
        file=sys.stderr,
    )
    sys.exit(0)

HERE = Path(__file__).resolve().parent
OUT_ROOT = HERE.parent / "public" / "bmz"

# Hardcoded BMZ catalog of models we know convert cleanly. Adding more is
# straightforward: drop a new entry with the artifact URLs from collection.json.
MODELS: dict[str, dict] = {
    "hiding-blowfish": {
        "name": "EnhancerMitochondriaEM2D",
        "base": "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/hiding-blowfish/1/files/",
        "weights": "weights-torchscript.pt",
        "test_input": "test_input_0.npy",
        "test_output": "test_output_0.npy",
        # min-max to [0,1] per-sample (RDF: scale_range, min_percentile=0, max_percentile=100)
        "normalization": {"kind": "min_max"},
        "opset": 17,
        # Trained on EPFL Lucchi+ mito EM at ~5 nm/voxel. Pick the closest
        # multiscale level when streaming through a multiscale source.
        "preferred_voxel_size_nm": 8.0,
    },
}


def download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    print(f"  downloading {dest.name} ...", flush=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)


def export_one(model_id: str, *, force: bool) -> None:
    spec = MODELS[model_id]
    out = OUT_ROOT / model_id
    onnx_path = out / "model.onnx"
    if onnx_path.exists() and not force:
        print(f"[{model_id}] already exported at {onnx_path}, skipping (use --force to redo)")
        return

    cache = HERE.parent / ".bmz-cache" / model_id
    cache.mkdir(parents=True, exist_ok=True)

    ts_path = cache / spec["weights"]
    xin_path = cache / spec["test_input"]
    yref_path = cache / spec["test_output"]
    download(spec["base"] + spec["weights"], ts_path)
    download(spec["base"] + spec["test_input"], xin_path)
    download(spec["base"] + spec["test_output"], yref_path)

    print(f"[{model_id}] loading torchscript ...", flush=True)
    m = torch.jit.load(str(ts_path), map_location="cpu").eval()

    x_np = np.load(xin_path).astype(np.float32)
    y_ref = np.load(yref_path).astype(np.float32)
    print(f"[{model_id}] test_input  shape={x_np.shape}  dtype={x_np.dtype}")
    print(f"[{model_id}] test_output shape={y_ref.shape}")

    with torch.no_grad():
        y_torch = m(torch.from_numpy(x_np)).numpy()
    err_torch = float(np.max(np.abs(y_torch - y_ref)))
    print(f"[{model_id}] torch vs reference  max abs err = {err_torch:.3e}")
    if err_torch > 1e-3:
        sys.exit(f"[{model_id}] torch output diverges from reference (err={err_torch})")

    out.mkdir(parents=True, exist_ok=True)
    print(f"[{model_id}] exporting ONNX (opset {spec['opset']}, static shape) ...", flush=True)
    torch.onnx.export(
        m,
        torch.from_numpy(x_np),
        str(onnx_path),
        input_names=["input0"],
        output_names=["output0"],
        opset_version=spec["opset"],
        do_constant_folding=True,
    )

    g = onnx.load(str(onnx_path))
    ops = {}
    for n in g.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    print(f"[{model_id}] ONNX ops: {dict(sorted(ops.items()))}")

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    y_ort = sess.run(None, {"input0": x_np})[0]
    err_ort = float(np.max(np.abs(y_ort - y_ref)))
    print(f"[{model_id}] ONNX (cpu) vs reference  max abs err = {err_ort:.3e}")
    if err_ort > 1e-2:
        sys.exit(f"[{model_id}] ONNX output diverges (err={err_ort})")

    (out / "test_input.bin").write_bytes(x_np.tobytes())
    (out / "test_output.bin").write_bytes(y_ref.tobytes())
    manifest = {
        "id": model_id,
        "name": spec["name"],
        "shape_in": list(x_np.shape),
        "shape_out": list(y_ref.shape),
        "dtype": "float32",
        "normalization": spec["normalization"],
        "opset": spec["opset"],
        "preferred_voxel_size_nm": spec.get("preferred_voxel_size_nm"),
        "verification": {
            "torch_vs_reference_max_abs_err": err_torch,
            "onnx_cpu_vs_reference_max_abs_err": err_ort,
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    size_mb = onnx_path.stat().st_size / 1e6
    print(f"[{model_id}] done. {onnx_path} ({size_mb:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="re-download and re-export")
    ap.add_argument("--model", default=None, help="only export this BMZ id (default: all)")
    args = ap.parse_args()

    ids = [args.model] if args.model else list(MODELS)
    for mid in ids:
        if mid not in MODELS:
            sys.exit(f"unknown model id: {mid}. known: {sorted(MODELS)}")
        export_one(mid, force=args.force)


if __name__ == "__main__":
    main()
