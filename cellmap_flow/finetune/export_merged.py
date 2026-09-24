"""Fold a finetune into the base model's weights and re-export it at a larger tile.

Why this exists
---------------
Every cellmap UNet on HuggingFace ships two inference artefacts, and both are
locked to the 178^3 -> 56^3 tile they were exported at: the ``.pt2``
(what ``type: finetune`` serves) refuses any other input, and the TorchScript
(what ``type: huggingface`` serves) silently crops a larger input back to
56^3. A valid-padding UNet spends 32 input voxels per output voxel at that
tile; at 306^3 -> 184^3 it spends 4.6. The eager ``model.pt`` is not shape
locked, so: load it, fold the LoRA adapter (or load full-finetune weights),
and export the plain module again at the tile you want. The result is an
ordinary CellmapModel folder served with ``type: cellmap`` -- no PEFT, no
fly_organelles, no adapter at inference time.

Tile rule: these UNets pool 2x2x2 three times, so the tile must be
178 + 16k (290, 306, 322, ...). Any other size shifts the pooling phase
and the output no longer matches what the 178 tile computes at the same
place (measured: 298 differs by up to 2.3 in logit space; 290/306 differ by
0.0). The 378 in the repos' metadata.json is NOT such a size.

Usage::

    python -m cellmap_flow.finetune.export_merged \\
        --repo cellmap/mito-aff-unet-setup-16 \\
        --lora-adapter-path <run>/lora_adapter \\
        --tile 306 --output <session>/exports/<dataset>/<class>/lora_aug_tile306

then in a cellmap-flow yaml::

    models:
      - type: cellmap
        folder_path: <that output dir>
        name: <anything>

Requires ``fly_organelles`` importable (the eager pickle references its
classes) -- present in the cellmap-flow-finetune env.
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime

import torch
from torch import nn

logger = logging.getLogger(__name__)

POOL_FACTOR = 8      # 3 x MaxPool3d(2) in the cellmap StandardUnet / UNet
TRAIN_TILE = 178
CONTEXT = 122        # in - out for these valid-padded nets (178 -> 56)


def valid_tile(tile):
    """True when ``tile`` keeps the pooling phase of the 178 training tile."""
    return tile >= TRAIN_TILE and (tile - TRAIN_TILE) % (2 * POOL_FACTOR) == 0


def merge_lora_into_conv3d(peft_model):
    """Add each LoRA pair into its base Conv3d weight in place; return count.

    PEFT's own ``merge_and_unload`` assumes 2-D conv kernels and fails on
    Conv3d. For lora_A: Conv3d(Cin, r, k^3) and lora_B: Conv3d(r, Cout, 1^3),
    delta[o, i, z, y, x] = scale * sum_r B[o, r] A[r, i, z, y, x].
    """
    from peft.tuners.lora import LoraLayer

    n = 0
    for mod in peft_model.modules():
        if isinstance(mod, LoraLayer) and isinstance(mod.base_layer, nn.Conv3d):
            for ad in mod.active_adapters:
                a_w = mod.lora_A[ad].weight
                b_w = mod.lora_B[ad].weight
                if tuple(b_w.shape[2:]) != (1, 1, 1):
                    raise NotImplementedError(f"lora_B kernel {tuple(b_w.shape[2:])} != 1^3")
                delta = torch.einsum("or,rizyx->oizyx", b_w[:, :, 0, 0, 0], a_w) * mod.scaling[ad]
                mod.base_layer.weight.data += delta.to(mod.base_layer.weight.dtype)
                n += 1
    return n


def strip_lora_layers(module):
    """Replace every LoraLayer under ``module`` by its (now merged) base layer."""
    from peft.tuners.lora import LoraLayer

    for parent in list(module.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, LoraLayer):
                setattr(parent, name, child.base_layer)
    return module


def load_eager_base(folder_path):
    """The eager ``model.pt`` of a CellmapModel folder, eval mode, on CPU."""
    path = os.path.join(folder_path, "model.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{folder_path} has no model.pt; only the eager pickle can change tile size")
    try:
        model = torch.load(path, map_location="cpu", weights_only=False)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"{e}. The eager pickle references the training package; make it "
            "importable (fly_organelles is in the cellmap-flow-finetune env)."
        ) from e
    return model.eval()


def apply_finetune(eager, lora_adapter_path=None, weights_path=None):
    """Return a plain nn.Module with the finetune folded in.

    Both finetune flavours were trained on ``BatchLoopWrapper(export)``, whose
    child is named ``model``, so adapter target names and full-finetune state
    dict keys carry a ``model.`` prefix. Wrapping the eager module the same
    way reproduces those names exactly (checked: 38/38 adapter tensors
    identical, 0.0 output difference at 178^3).
    """
    from cellmap_flow.finetune.lora_wrapper import BatchLoopWrapper

    if bool(lora_adapter_path) == bool(weights_path):
        raise ValueError("pass exactly one of lora_adapter_path / weights_path")
    wrapped = BatchLoopWrapper(eager).eval()
    if lora_adapter_path:
        from cellmap_flow.finetune.lora_wrapper import load_lora_adapter

        peft = load_lora_adapter(wrapped, lora_adapter_path, is_trainable=False).eval()
        n = merge_lora_into_conv3d(peft)
        base = peft.get_base_model()          # the BatchLoopWrapper
        strip_lora_layers(base)
        logger.info(f"Merged {n} LoRA conv pairs into the base weights")
        merged = base.model
    else:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        missing, unexpected = wrapped.load_state_dict(state, strict=True)
        logger.info(f"Loaded full-finetune weights: {len(state)} tensors")
        merged = wrapped.model
    return merged.eval()


def check_equivalence(module, reference, tile, seed=0):
    """Max |diff| between ``module`` and ``reference`` at 178^3, and between the
    centre 56^3 of ``module``'s output at ``tile`` and its own 178 output there."""
    torch.manual_seed(seed)
    with torch.no_grad():
        x = torch.rand(1, 1, TRAIN_TILE, TRAIN_TILE, TRAIN_TILE)
        d_ref = (module(x) - reference(x)).abs().max().item() if reference is not None else None
        xb = torch.rand(1, 1, tile, tile, tile)
        yb = module(xb)
        c = (tile - TRAIN_TILE) // 2
        ys = module(xb[:, :, c:c + TRAIN_TILE, c:c + TRAIN_TILE, c:c + TRAIN_TILE])
        oc = (yb.shape[2] - ys.shape[2]) // 2
        o = ys.shape[2]
        d_tile = (yb[:, :, oc:oc + o, oc:oc + o, oc:oc + o] - ys).abs().max().item()
    return d_ref, d_tile, tuple(yb.shape[2:])


def export_folder(module, base_folder, output, tile, out_shape, provenance, name_suffix):
    """Write model.pt2 + model.ts + metadata.json (+ README) for ``module`` at ``tile``."""
    os.makedirs(output, exist_ok=True)
    example = torch.rand(1, 1, tile, tile, tile)
    t0 = time.time()
    ep = torch.export.export(module, (example,))
    torch.export.save(ep, os.path.join(output, "model.pt2"))
    logger.info(f"model.pt2 exported at {tile}^3 in {time.time() - t0:.0f}s")
    t0 = time.time()
    with torch.no_grad():
        ts = torch.jit.trace(module, example)
    ts.save(os.path.join(output, "model.ts"))
    logger.info(f"model.ts traced at {tile}^3 in {time.time() - t0:.0f}s")

    meta = json.load(open(os.path.join(base_folder, "metadata.json")))
    meta.update({
        "model_name": f"{meta.get('model_name', 'model')}_{name_suffix}",
        "input_shape": [tile] * 3,
        "output_shape": list(out_shape),
        "inference_input_shape": [tile] * 3,
        "inference_output_shape": list(out_shape),
        "description": (meta.get("description", "") + f" | Finetune folded into the weights and "
                        f"re-exported at {tile}^3 -> {out_shape[0]}^3 by cellmap_flow.finetune.export_merged "
                        f"on {datetime.now():%Y-%m-%d}. See export_record.json.").strip(" |"),
    })
    with open(os.path.join(output, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(output, "export_record.json"), "w") as f:
        json.dump(provenance, f, indent=2)
    with open(os.path.join(output, "README.md"), "w") as f:
        f.write(f"# {meta['model_name']}\n\nFinetuned cellmap model, adapter folded into the weights, exported at "
                f"{tile}^3 -> {out_shape[0]}^3 (both model.pt2 and model.ts are locked to that tile).\n"
                f"Serve with `type: cellmap`, `folder_path: {os.path.abspath(output)}`.\n\n"
                f"Provenance: export_record.json. Base: {provenance.get('base_repo') or base_folder}.\n")
    return output


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--repo", help="HuggingFace repo of the base model")
    src.add_argument("--base-folder", help="local CellmapModel folder of the base model")
    ft = p.add_mutually_exclusive_group(required=True)
    ft.add_argument("--lora-adapter-path")
    ft.add_argument("--weights-path", help="full-finetune model_state_dict.pt")
    p.add_argument("--tile", type=int, default=306, help="input tile; must be 178 + 16k (default 306 -> 184)")
    p.add_argument("--output", required=True, help="CellmapModel folder to create")
    p.add_argument("--name-suffix", default=None)
    p.add_argument("--skip-check", action="store_true", help="skip the CPU equivalence check (saves ~3 min)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if not valid_tile(args.tile):
        sys.exit(f"--tile {args.tile} shifts the pooling phase; use 178 + 16k (290, 306, 322, ...)")
    if os.path.exists(os.path.join(args.output, "model.pt2")):
        sys.exit(f"{args.output} already holds an export")

    if args.repo:
        from cellmap_models.model_export.cellmap_model import get_huggingface_model
        cm = get_huggingface_model(args.repo)
        base_folder = cm.folder_path
    else:
        base_folder = args.base_folder
    eager = load_eager_base(base_folder)
    merged = apply_finetune(eager, args.lora_adapter_path, args.weights_path)

    d_ref = d_tile = None
    if not args.skip_check:
        # reference: the export + adapter path type: finetune serves today
        reference = None
        if args.lora_adapter_path:
            from cellmap_models.model_export.cellmap_model import CellmapModel
            from cellmap_flow.finetune.lora_wrapper import BatchLoopWrapper, load_lora_adapter
            exp = BatchLoopWrapper(CellmapModel(base_folder).train()).eval()
            reference = load_lora_adapter(exp, args.lora_adapter_path, is_trainable=False).eval()
        d_ref, d_tile, out_shape = check_equivalence(merged, reference, args.tile)
        logger.info(f"equivalence: vs served finetune path at 178^3 max|diff|={d_ref}; "
                    f"{args.tile}-tile centre vs 178-tile max|diff|={d_tile}")
        if d_ref is not None and d_ref > 1e-3:
            sys.exit(f"merged model differs from the served finetune path by {d_ref}; refusing to export")
        if d_tile > 1e-3:
            sys.exit(f"{args.tile}-tile output differs from the 178-tile output by {d_tile}; refusing to export")
    else:
        with torch.no_grad():
            out_shape = tuple(merged(torch.rand(1, 1, args.tile, args.tile, args.tile)).shape[2:])

    suffix = args.name_suffix or f"tile{args.tile}"
    prov = {
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "base_repo": args.repo, "base_folder": base_folder,
        "lora_adapter_path": args.lora_adapter_path, "weights_path": args.weights_path,
        "tile": args.tile, "output_shape": list(out_shape),
        "equivalence_178_vs_served_path": d_ref, "tile_phase_check": d_tile,
        "torch": torch.__version__,
    }
    export_folder(merged, base_folder, args.output, args.tile, out_shape, prov, suffix)
    print(json.dumps({"output": os.path.abspath(args.output), "tile": args.tile, "output_shape": list(out_shape)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
