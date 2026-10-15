#!/usr/bin/env python
"""Structurally prune the base StandardUnet and export a small single-channel ONNX
for browser-side (onnxruntime-web / WebGPU) inference.

The full model is ~792M params / 3.17 GB fp32 — far too big for the browser, with
single 1.29 GB conv weights at the 3456-channel bottleneck that exceed WebGPU
buffer limits. We keep the highest-L2 output channels of every conv and rewire
each conv's input to its producer's kept channels, then slice the output head to
a single channel.

Why a custom pruner (not torch-pruning): funlib's UNet uses 'valid' padding and
center-crops the skip features before `torch.cat`. That crop registers as a slice
op that severs auto-pruners' channel dependency — the skip branch of each decoder
concat never gets coupled to its consumer, producing inconsistent models. The
funlib structure is regular enough to prune deterministically instead.

No LoRA, no retraining. WARNING: in practice this model's output COLLAPSES to a
near-constant field without finetuning (verified: output std ~0.006 vs ~0.21 for
the full model) — structured channel pruning disrupts learned features and the
error compounds across layers. A pruned model is only useful after a short
finetune to recover function. Only supports constant_upsample=True (nn.Upsample,
no params), which StandardUnet hardcodes.

Usage:
    python scripts/prune_onnx.py -o web/models/mito_small.onnx --ratio 0.85

Pruning ratio p removes that fraction of output channels per conv; conv->conv
params scale ~(1-p)^2, so 0.85 takes ~792M -> ~20M params. Raise --ratio if still
too big for the browser, lower it to keep more accuracy.
"""
import argparse

BASE_CHECKPOINT = "/groups/cellmap/cellmap/zouinkhim/exp_mito/runs/setup_16/model_checkpoint_400000"
NUM_CLASSES = 13


def convs_of(convpass):
    import torch.nn as nn

    return [m for m in convpass.conv_pass if isinstance(m, nn.Conv3d)]


def slice_conv(conv, out_idx, in_idx):
    import torch.nn as nn

    w = conv.weight.data[out_idx][:, in_idx].contiguous()
    conv.weight = nn.Parameter(w)
    if conv.bias is not None:
        conv.bias = nn.Parameter(conv.bias.data[out_idx].contiguous())
    conv.out_channels, conv.in_channels = len(out_idx), len(in_idx)


def prune_unet(backbone, final_conv, ratio, keep_channel=0):
    """In-place structured channel prune of a funlib UNet + StandardUnet head.

    keep_channel: output channel to keep at the head, or None to keep ALL output
    channels (used during distillation finetuning; slice to one channel at export).
    """
    import torch

    head = 0
    levels = len(backbone.l_conv)  # encoder levels incl. bottleneck (last index)
    rconv = backbone.r_conv[head]

    # constant_upsample=True -> nn.Upsample (no params). Transposed-conv upsampling
    # would need its weights pruned too; refuse rather than silently corrupt.
    if any(p.numel() for p in backbone.r_up.parameters()):
        raise NotImplementedError("upsample has parameters (constant_upsample=False) — unsupported")

    out_keep, orig_out = {}, {}

    def choose(conv):
        w = conv.weight.data
        o = w.shape[0]
        k = max(1, round((1 - ratio) * o))
        norms = w.reshape(o, -1).norm(dim=1)
        return sorted(torch.argsort(norms, descending=True)[:k].tolist())

    convpasses = [backbone.l_conv[L] for L in range(levels)] + [rconv[L] for L in range(len(rconv))]
    for cp in convpasses:
        for c in convs_of(cp):
            orig_out[id(c)] = c.out_channels
            out_keep[id(c)] = choose(c)
    orig_out[id(final_conv)] = final_conv.out_channels
    final_keep = list(range(final_conv.out_channels)) if keep_channel is None else [keep_channel]
    out_keep[id(final_conv)] = final_keep

    # Encoder: conv0 input from previous level's last conv (downsample is param-free);
    # level 0 input is the single image channel.
    for L in range(levels):
        cs = convs_of(backbone.l_conv[L])
        first_in = [0] if L == 0 else out_keep[id(convs_of(backbone.l_conv[L - 1])[-1])]
        for ci, c in enumerate(cs):
            slice_conv(c, out_keep[id(c)], first_in if ci == 0 else out_keep[id(cs[ci - 1])])

    # Decoder: first conv input is cat([skip, upsampled]) -> [skip_keep, skipOrig + up_keep].
    maxdec = len(rconv) - 1
    for L in range(len(rconv)):
        cs = convs_of(rconv[L])
        skip_prod = convs_of(backbone.l_conv[L])[-1]
        up_prod = (
            convs_of(backbone.l_conv[levels - 1])[-1] if L == maxdec else convs_of(rconv[L + 1])[-1]
        )
        first_in = list(out_keep[id(skip_prod)]) + [
            orig_out[id(skip_prod)] + i for i in out_keep[id(up_prod)]
        ]
        for ci, c in enumerate(cs):
            slice_conv(c, out_keep[id(c)], first_in if ci == 0 else out_keep[id(cs[ci - 1])])

    slice_conv(final_conv, final_keep, out_keep[id(convs_of(rconv[0])[-1])])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--output", required=True, help="output .onnx path")
    ap.add_argument("--checkpoint", default=BASE_CHECKPOINT)
    ap.add_argument("--classes", type=int, default=NUM_CLASSES)
    ap.add_argument("--keep-channel", type=int, default=0, help="output channel to keep")
    ap.add_argument("--ratio", type=float, default=0.85, help="fraction of channels pruned per conv")
    ap.add_argument("--input-size", type=int, nargs=3, default=[178, 178, 178])
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    from pathlib import Path
    from fly_organelles.model import StandardUnet

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    backbone = StandardUnet(args.classes)
    ckpt = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    backbone.load_state_dict(ckpt["model_state_dict"])
    model = nn.Sequential(backbone, nn.Sigmoid()).to(device).eval()

    z, y, x = args.input_size
    example = torch.zeros(1, 1, z, y, x, device=device)

    n0 = sum(p.numel() for p in model.parameters())
    prune_unet(backbone.unet_backbone, backbone.final_conv, args.ratio, args.keep_channel)
    n1 = sum(p.numel() for p in model.parameters())
    print(f"params: {n0:,} -> {n1:,} ({n1 / n0:.2%}); ~{n1 * 4 / 1e6:.1f} MB fp32")

    with torch.no_grad():
        out = model(example)
    print(f"output shape: {tuple(out.shape)} (expected (1, 1, 56, 56, 56))")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            model.cpu(),
            (example.cpu(),),
            args.output,
            opset_version=args.opset,
            input_names=["input"],
            output_names=["output"],
            do_constant_folding=True,
        )

    # Some torch versions emit weights as an external .data sidecar, which
    # onnxruntime-web cannot load. Re-save as one self-contained file.
    import onnx

    onnx.save_model(onnx.load(args.output), args.output, save_as_external_data=False)
    sidecar = Path(args.output + ".data")
    if sidecar.exists():
        sidecar.unlink()

    sz = out_path.stat().st_size / 1e6
    print(f"wrote {args.output} ({sz:.1f} MB, single file)")
    if sz > 100:
        print("⚠ still >100 MB — raise --ratio for the browser.")


if __name__ == "__main__":
    main()
