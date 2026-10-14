#!/usr/bin/env python
"""Prune the base StandardUnet, then DISTILL it against the full model so the
small student recovers function, and export a single-channel ONNX for the browser.

Structured pruning alone collapses this model's output to a near-constant field
(verified: std ~0.006 vs ~0.21 for the full model). Finetuning fixes that. We use
knowledge distillation — no ground-truth labels needed: the full (unpruned) model
is the teacher, and the pruned student is trained to match the teacher's output on
raw EM blocks. Then we slice the head to one channel and export.

Requires: torch, fly_organelles, tensorstore (+ a CUDA GPU for any real speed).

Usage:
    python scripts/prune_finetune_onnx.py \
        --data https://host/.../fibsem-uint8/s1 \
        -o web/models/mito_small_ft.onnx \
        --ratio 0.85 --steps 2000

Watch the printed distillation loss fall, and the final output std (should climb
toward the teacher's ~0.2, not stay ~0). Tune --ratio (smaller model vs. easier
to recover) and --steps.
"""
import argparse

from prune_onnx import prune_unet, slice_conv, convs_of  # same dir

BASE_CHECKPOINT = "/groups/cellmap/cellmap/zouinkhim/exp_mito/runs/setup_16/model_checkpoint_400000"
NUM_CLASSES = 13


def build_model(classes, checkpoint, device):
    import torch
    import torch.nn as nn
    from fly_organelles.model import StandardUnet

    backbone = StandardUnet(classes)
    ckpt = torch.load(checkpoint, weights_only=True, map_location="cpu")
    backbone.load_state_dict(ckpt["model_state_dict"])
    return nn.Sequential(backbone, nn.Sigmoid()).to(device)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--data", required=True, help="EM zarr scale-array URL (raw input for distillation)")
    ap.add_argument("--checkpoint", default=BASE_CHECKPOINT)
    ap.add_argument("--classes", type=int, default=NUM_CLASSES)
    ap.add_argument("--ratio", type=float, default=0.85)
    ap.add_argument("--keep-channel", type=int, default=0)
    ap.add_argument("--input-size", type=int, nargs=3, default=[178, 178, 178])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--region", type=int, nargs=3, default=[320, 320, 320], help="EM region to preload")
    # Input normalization — MUST match what you set in the browser form and what
    # the model was trained on. Default: clip [min,max] -> [0,1] -> 2x-1 -> [-1,1].
    ap.add_argument("--clip-min", type=float, default=0.0)
    ap.add_argument("--clip-max", type=float, default=255.0)
    ap.add_argument("--no-pm1", action="store_true", help="skip 2*x-1 (use [0,1] instead of [-1,1])")
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import numpy as np
    import torch
    import torch.nn as nn
    import tensorstore as ts
    from pathlib import Path

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    z, y, x = args.input_size

    # Teacher (frozen) and student (pruned, full output channels for now).
    teacher = build_model(args.classes, args.checkpoint, device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = build_model(args.classes, args.checkpoint, device)
    prune_unet(student[0].unet_backbone, student[0].final_conv, args.ratio, keep_channel=None)
    student.train()
    n_t = sum(p.numel() for p in teacher.parameters())
    n_s = sum(p.numel() for p in student.parameters())
    print(f"teacher {n_t:,} params | student {n_s:,} params ({n_s / n_t:.1%})")

    # Preload an EM region once; sample random input-sized crops from it.
    arr = ts.open({"driver": "zarr", "kvstore": args.data.rstrip("/") + "/", "open": True}).result()
    rz, ry, rx = [min(args.region[i], arr.shape[i]) for i in range(3)]
    o = [(arr.shape[i] - [rz, ry, rx][i]) // 2 for i in range(3)]
    raw = arr[o[0]:o[0] + rz, o[1]:o[1] + ry, o[2]:o[2] + rx].read().result().astype(np.float32)
    region = np.clip(raw, args.clip_min, args.clip_max)
    region = (region - args.clip_min) / (args.clip_max - args.clip_min)  # -> [0,1]
    if not args.no_pm1:
        region = 2.0 * region - 1.0  # -> [-1,1]
    print(f"preloaded EM region {region.shape}, normalized mean {region.mean():.3f} "
          f"(range [{region.min():.2f}, {region.max():.2f}])")

    def sample():
        sz = [np.random.randint(0, region.shape[i] - [z, y, x][i] + 1) for i in range(3)]
        crop = region[sz[0]:sz[0] + z, sz[1]:sz[1] + y, sz[2]:sz[2] + x]
        return torch.from_numpy(crop)[None, None].to(device)

    opt = torch.optim.Adam(student.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    for step in range(args.steps):
        xb = sample()
        with torch.no_grad():
            target = teacher(xb)
        out = student(xb)
        loss = loss_fn(out, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == args.steps - 1:
            print(f"step {step:5d}  loss {loss.item():.5f}  student_std {out.std().item():.4f}")

    # Slice the head to the requested single channel, then export one file.
    student.eval()
    fc = student[0].final_conv
    slice_conv(fc, [args.keep_channel], list(range(fc.in_channels)))

    example = torch.zeros(1, 1, z, y, x, device=device)
    with torch.no_grad():
        out = student(example)
    print(f"final output shape {tuple(out.shape)}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            student.cpu(),
            (example.cpu(),),
            args.output,
            opset_version=args.opset,
            input_names=["input"],
            output_names=["output"],
            do_constant_folding=True,
        )
    import onnx

    onnx.save_model(onnx.load(args.output), args.output, save_as_external_data=False)
    sidecar = Path(args.output + ".data")
    if sidecar.exists():
        sidecar.unlink()
    print(f"wrote {args.output} ({Path(args.output).stat().st_size / 1e6:.1f} MB, single file)")


if __name__ == "__main__":
    main()
