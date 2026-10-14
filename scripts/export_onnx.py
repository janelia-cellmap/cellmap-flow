#!/usr/bin/env python
"""Export a CellMap Flow model script to ONNX for browser-side inference.

A model script is the same `-s/--script` file used by `cellmap_flow script`
(see cellmap_flow/models/models_config.py::ScriptModelConfig). It must define a
module-level `model` (torch.nn.Module) and `input_size` (z, y, x voxels).

Usage:
    python scripts/export_onnx.py -s models/setup16_lsd_400k_all_16_finetuned_*.py \
        -o web/models/mito_test.onnx

After export we statically scan the graph and flag 3D ConvTranspose nodes, which
were the suspected WebGPU gap (Conv3D landed in onnxruntime-web 1.25.0). If they
appear, confirm they run on WebGPU in the browser (env.logLevel='verbose'); if
they fall back to WASM, re-export with Resize + Conv3D replacing the transposed
convs (see web/README.md).
"""
import argparse
import importlib.util
import sys
from collections import Counter
from pathlib import Path


def load_script(path: str):
    spec = importlib.util.spec_from_file_location("cf_model_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-s", "--script", required=True, help="model script path")
    ap.add_argument("-o", "--output", required=True, help="output .onnx path")
    ap.add_argument("--opset", type=int, default=18)
    args = ap.parse_args()

    import torch

    mod = load_script(args.script)
    if not hasattr(mod, "model") or not hasattr(mod, "input_size"):
        sys.exit("script must define `model` and `input_size`")

    model = mod.model.to("cpu").eval()
    z, y, x = mod.input_size
    dummy = torch.zeros(1, 1, z, y, x, dtype=torch.float32)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Exporting {type(model).__name__} input (1,1,{z},{y},{x}) -> {out}")

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(out),
            opset_version=args.opset,
            input_names=["input"],
            output_names=["output"],
            do_constant_folding=True,
        )
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")

    # Static op-coverage scan.
    try:
        import onnx

        graph = onnx.load(str(out)).graph
        ops = Counter(n.op_type for n in graph.node)
        print("\nOp types:")
        for op, n in sorted(ops.items()):
            print(f"  {op:24s} {n}")

        conv_t_3d = [
            node.name or "<unnamed>"
            for node in graph.node
            if node.op_type == "ConvTranspose"
            and any(
                len(a.ints) == 3
                for a in node.attribute
                if a.name in ("kernel_shape", "strides")
            )
        ]
        if conv_t_3d:
            print(
                f"\n⚠  {len(conv_t_3d)} 3D ConvTranspose node(s) found. Verify these "
                "run on the WebGPU EP in the browser; if they fall back to WASM, "
                "re-export with Resize + Conv3D (see web/README.md)."
            )
        else:
            print("\n✓ No 3D ConvTranspose nodes detected.")
    except ImportError:
        print("(install `onnx` to enable the op-coverage scan)")


if __name__ == "__main__":
    main()
