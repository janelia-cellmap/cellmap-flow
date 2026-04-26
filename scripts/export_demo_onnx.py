"""Export a tiny demo ONNX model for browser-inference MVP.

Produces a batched 2D Laplacian edge-detection model with input/output
shape (B, 1, H, W) float32 so it runs on BOTH WebGPU and WASM backends
in ONNX Runtime Web. The WASM backend does not implement Conv3d, so the
3D demo has to be expressed as batched Conv2d; callers feed a (D, 1, H, W)
tensor and treat D as the batch.

Usage:
    python scripts/export_demo_onnx.py [output_path]

Default output: browser/public/demo-model.onnx
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn


class DemoEdgeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=True)
        with torch.no_grad():
            k = torch.tensor(
                [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]]
            )
            self.conv.weight.copy_(k.view(1, 1, 3, 3))
            self.conv.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / 255.0
        y = self.conv(x)
        return torch.sigmoid(y * 4.0)


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("browser/public/demo-model.onnx")
    out.parent.mkdir(parents=True, exist_ok=True)

    model = DemoEdgeModel().eval()
    dummy = torch.zeros(1, 1, 64, 64, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy,
        str(out),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "B", 2: "H", 3: "W"},
            "output": {0: "B", 2: "H", 3: "W"},
        },
        opset_version=17,
    )
    print(f"wrote {out} ({out.stat().st_size / 1024:.1f} KiB)")

    # Matching browser-side ModelSpec (see browser/src/model-spec.ts).
    import json
    spec = {
        "inputVoxelSize": [1, 1, 1],
        "outputVoxelSize": [1, 1, 1],
        "readShape": [16, 128, 128],
        "writeShape": [16, 128, 128],
        "inputDtype": "float32",
        "outputDtype": "uint8",
        "outputChannels": 1,
        "blockShape": [16, 128, 128],
        "tensorLayout": "BatchZ_NCHW",
        "normalize": [],
        "postprocess": [
            {
                "name": "DefaultPostprocessor",
                "clip_min": 0,
                "clip_max": 1,
                "bias": 0,
                "multiplier": 255,
            }
        ],
    }
    spec_path = out.with_suffix(".json")
    spec_path.write_text(json.dumps(spec, indent=2))
    print(f"wrote {spec_path}")


if __name__ == "__main__":
    main()
