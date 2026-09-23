"""Build a finetuning corrections directory from a crops manifest, headlessly.

The dashboard builds the same thing interactively: it creates an
``annotation_volume.zarr`` covering the raw dataset at the model's output
voxel size, imports each crop of a YAML manifest into it at the crop's
physical offset, and writes the ``_virtual_sources.json`` manifest the
trainer reads. That path needs a running Flask session, a MinIO server and
the model's geometry from a live inference server. None of that is needed to
train from ground-truth crops on the cluster, so this module does the same
three steps with the same functions and nothing else::

    python -m cellmap_flow.finetune.build_corrections \\
        --raw /nrs/cellmap/data/<ds>/<ds>.zarr/recon-1/em/fibsem-uint8 \\
        --repo cellmap/mito-aff-unet-setup-16 \\
        --crops finetune_yamls/<ds>/mito_group.yaml \\
        --output /nrs/.../training/<session>/corrections \\
        --patches-per-epoch 360

Geometry comes from the HuggingFace ``metadata.json`` (input/output shape
and voxel size), so no weights are downloaded and no GPU is touched. Pass
``--input-shape/--output-shape/--input-voxel-size/--output-voxel-size`` to
override or to describe a model that has no metadata.

Every build leaves ``build_record.json`` next to the manifest: the crops
YAML verbatim, the resolved geometry, per-crop voxel counts and offsets,
and the git commit of this checkout. That is the provenance a run registry
can point at.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import uuid
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)

# What both cellmap UNet families expect at the input: uint8 EM scaled to
# [-1, 1]. The same normalisation the dashboard sessions for these models
# used (see my_yamls/*_mito.yaml, *_nuc.yaml), so adapters trained here
# behave identically when served.
DEFAULT_INPUT_NORM = {
    "MinMaxNormalizer": {"min_value": 0.0, "max_value": 255.0, "invert": False},
    "LambdaNormalizer": {"expression": "x*2-1"},
}


def _git_commit(repo_dir):
    try:
        return subprocess.run(
            ["git", "-C", repo_dir, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return None


def geometry_from_hf_metadata(repo, revision=None):
    """(input_shape, output_shape, input_voxel_size, output_voxel_size,
    out_channels, channels_names, model_name) from a repo's metadata.json."""
    from cellmap_flow.models.models_config import HuggingFaceModelConfig

    meta = HuggingFaceModelConfig(repo, revision=revision)._load_metadata()
    if not meta:
        raise RuntimeError(f"{repo} has no metadata.json; pass the geometry flags explicitly")
    required = ["input_shape", "output_shape", "input_voxel_size", "output_voxel_size"]
    missing = [k for k in required if k not in meta]
    if missing:
        raise RuntimeError(f"{repo} metadata.json lacks {missing}")
    return dict(
        input_shape=[int(v) for v in meta["input_shape"]],
        output_shape=[int(v) for v in meta["output_shape"]],
        input_voxel_size=[float(v) for v in meta["input_voxel_size"]],
        output_voxel_size=[float(v) for v in meta["output_voxel_size"]],
        out_channels=int(meta.get("out_channels", 1)),
        channels_names=list(meta.get("channels_names") or []),
        model_name=str(meta.get("model_name") or repo.split("/")[-1]),
    )


def build_corrections(
    *,
    raw_dataset_path,
    crops_yaml,
    output_dir,
    repo=None,
    revision=None,
    input_shape=None,
    output_shape=None,
    input_voxel_size=None,
    output_voxel_size=None,
    model_name=None,
    patches_per_epoch=None,
    jitter_voxels=None,
    seed=0,
    dense_to_sparse_ratio=None,
    input_norm=None,
    postprocess=None,
    extra_record=None,
):
    """Create ``output_dir`` with the annotation volume, imported crops and
    trainer manifest. Returns the build record (also written to disk).

    ``output_dir`` must not already contain a volume: this builds from
    scratch so the record describes everything in it.
    """
    from cellmap_flow.dashboard.finetune_utils import create_annotation_volume_zarr
    from cellmap_flow.dashboard.routes.finetune.yaml_crops import _write_crop_into_volume
    from cellmap_flow.finetune.crop_loader import parse_crops_yaml
    from cellmap_flow.finetune.virtual_dataset import write_manifest
    from cellmap_flow.image_data_interface import ImageDataInterface
    from cellmap_flow.utils.neuroglancer_utils import get_raw_closest_scale

    if os.path.isdir(output_dir) and any(
        n.endswith(".zarr") or n == "_virtual_sources.json" for n in os.listdir(output_dir)
    ):
        raise FileExistsError(f"{output_dir} already holds a corrections build; pick a new dir")

    geom = {}
    if repo:
        geom = geometry_from_hf_metadata(repo, revision)
    overrides = dict(
        input_shape=input_shape, output_shape=output_shape,
        input_voxel_size=input_voxel_size, output_voxel_size=output_voxel_size,
    )
    for k, v in overrides.items():
        if v is not None:
            geom[k] = list(v)
    for k in ("input_shape", "output_shape", "input_voxel_size", "output_voxel_size"):
        if k not in geom:
            raise ValueError(f"No {k}: pass --repo with metadata or --{k.replace('_', '-')}")
    model_name = model_name or geom.get("model_name") or "model"

    input_size = np.array(geom["input_shape"], dtype=int)
    output_size = np.array(geom["output_shape"], dtype=int)
    claimed_in_vs = np.array(geom["input_voxel_size"], dtype=float)
    claimed_out_vs = np.array(geom["output_voxel_size"], dtype=float)

    # Same resolution snapping the dashboard does: a model that says 16 nm
    # runs on whichever raw pyramid level is closest to 16 nm.
    eff_out_vs = np.array(get_raw_closest_scale(raw_dataset_path, tuple(claimed_out_vs)) or claimed_out_vs, dtype=float)
    eff_in_vs = np.array(get_raw_closest_scale(raw_dataset_path, tuple(claimed_in_vs)) or claimed_in_vs, dtype=float)

    idi = ImageDataInterface(raw_dataset_path, voxel_size=eff_out_vs)
    dataset_offset_nm = np.array(idi.roi.offset, dtype=float)
    dataset_shape_nm = np.array(idi.roi.shape, dtype=float)
    dataset_shape_voxels = (dataset_shape_nm / eff_out_vs).astype(int)
    dataset_shape_voxels = np.ceil(dataset_shape_voxels / output_size).astype(int) * output_size

    os.makedirs(output_dir, exist_ok=True)
    volume_id = f"vol-{uuid.uuid4().hex[:8]}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    zarr_path = os.path.join(output_dir, f"{volume_id}.zarr")
    input_norm = input_norm if input_norm is not None else DEFAULT_INPUT_NORM

    ok, info = create_annotation_volume_zarr(
        zarr_path=zarr_path,
        dataset_shape_voxels=dataset_shape_voxels,
        output_voxel_size=eff_out_vs,
        dataset_offset_nm=dataset_offset_nm,
        chunk_size=output_size,
        dataset_path=raw_dataset_path,
        model_name=model_name,
        input_size=input_size,
        input_voxel_size=eff_in_vs,
        claimed_output_voxel_size=claimed_out_vs,
        claimed_input_voxel_size=claimed_in_vs,
        input_norm_config=input_norm,
        postprocess_config=postprocess,
    )
    if not ok:
        raise RuntimeError(f"create_annotation_volume_zarr failed: {info}")
    logger.info(f"Created {zarr_path}: {dataset_shape_voxels.tolist()} voxels at {eff_out_vs.tolist()} nm")

    volume_meta = {
        "zarr_path": zarr_path,
        "output_voxel_size": eff_out_vs.tolist(),
        "dataset_offset_nm": dataset_offset_nm.tolist(),
    }
    cfg = parse_crops_yaml(crops_yaml)
    if not cfg.crops:
        raise ValueError(f"{crops_yaml} lists no crops")
    imported = []
    for entry in cfg.crops:
        n_fg = _write_crop_into_volume(volume_meta, entry)
        logger.info(f"Imported {entry.name or entry.path}: {n_fg} fg voxels")
        imported.append({"name": entry.name, "path": entry.path, "fg_ids": entry.fg_ids,
                         "mode": entry.mode, "connected_components": entry.connected_components,
                         "n_fg_voxels": int(n_fg)})
    total_fg = sum(c["n_fg_voxels"] for c in imported)
    if total_fg == 0:
        raise RuntimeError("No foreground voxel was imported from any crop; check fg_ids")

    # The manifest's own fields win over the CLI's so a manifest that says
    # patches_per_epoch travels with its crops.
    manifest = {
        "kind": "volume_zarr_v1",
        "volume_zarr_path": zarr_path,
        "raw_dataset_path": raw_dataset_path,
        "input_size_voxels": input_size.tolist(),
        "output_size_voxels": output_size.tolist(),
        "input_voxel_size_nm": eff_in_vs.tolist(),
        "output_voxel_size_nm": eff_out_vs.tolist(),
        "patches_per_epoch": cfg.patches_per_epoch if cfg.patches_per_epoch is not None else patches_per_epoch,
        "jitter_voxels": cfg.jitter_voxels if cfg.jitter_voxels is not None else jitter_voxels,
        "seed": cfg.seed if cfg.seed is not None else seed,
        "input_norm": input_norm,
        "dense_to_sparse_ratio": cfg.dense_to_sparse_ratio if cfg.dense_to_sparse_ratio is not None else dense_to_sparse_ratio,
    }
    if postprocess is not None:
        manifest["postprocess"] = postprocess
    manifest_path = write_manifest(output_dir, manifest)

    repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    record = {
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "cellmap_flow_commit": _git_commit(repo_dir),
        "raw_dataset_path": raw_dataset_path,
        "model": {"repo": repo, "revision": revision, "model_name": model_name,
                  "out_channels": geom.get("out_channels"), "channels_names": geom.get("channels_names")},
        "geometry": {
            "input_shape": input_size.tolist(), "output_shape": output_size.tolist(),
            "claimed_input_voxel_size_nm": claimed_in_vs.tolist(),
            "claimed_output_voxel_size_nm": claimed_out_vs.tolist(),
            "effective_input_voxel_size_nm": eff_in_vs.tolist(),
            "effective_output_voxel_size_nm": eff_out_vs.tolist(),
            "dataset_offset_nm": dataset_offset_nm.tolist(),
            "dataset_shape_voxels": dataset_shape_voxels.tolist(),
        },
        "crops_yaml_path": os.path.abspath(crops_yaml),
        "crops_yaml": open(crops_yaml).read() if os.path.exists(crops_yaml) else None,
        "crops": imported,
        "total_fg_voxels": int(total_fg),
        "manifest_path": manifest_path,
        "volume_zarr_path": zarr_path,
    }
    if extra_record:
        record.update(extra_record)
    record_path = os.path.join(output_dir, "build_record.json")
    with open(record_path, "w") as f:
        json.dump(record, f, indent=2)
    logger.info(f"Wrote {manifest_path} and {record_path}")
    return record


def _triple(kind):
    def parse(s):
        vals = [kind(v) for v in s.replace(",", " ").split()]
        if len(vals) == 1:
            vals = vals * 3
        if len(vals) != 3:
            raise argparse.ArgumentTypeError(f"expected 1 or 3 values, got {s!r}")
        return vals
    return parse


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--raw", required=True, help="raw EM multiscale group (the dataset the crops annotate)")
    p.add_argument("--crops", required=True, help="crops manifest YAML (CropsConfig schema)")
    p.add_argument("--output", required=True, help="corrections dir to create (must be new)")
    p.add_argument("--repo", help="HuggingFace repo whose metadata.json supplies the geometry")
    p.add_argument("--revision")
    p.add_argument("--model-name", help="name recorded in the volume (default: metadata model_name)")
    p.add_argument("--input-shape", type=_triple(int))
    p.add_argument("--output-shape", type=_triple(int))
    p.add_argument("--input-voxel-size", type=_triple(float))
    p.add_argument("--output-voxel-size", type=_triple(float))
    p.add_argument("--patches-per-epoch", type=int, help="fixed patches per epoch (default: one per fg-bearing chunk)")
    p.add_argument("--jitter-voxels", type=_triple(int))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dense-to-sparse-ratio", type=float)
    p.add_argument("--input-norm", help="JSON input_norm block (default: MinMax 0-255 then x*2-1)")
    p.add_argument("--postprocess", help="JSON postprocess block recorded for the served yaml")
    p.add_argument("--record", action="append", default=[],
                   help="extra key=value pairs for build_record.json (e.g. dataset=jrc_x class=mito_group)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    extra = {}
    for kv in args.record:
        k, _, v = kv.partition("=")
        extra[k] = v
    record = build_corrections(
        raw_dataset_path=args.raw, crops_yaml=args.crops, output_dir=args.output,
        repo=args.repo, revision=args.revision, model_name=args.model_name,
        input_shape=args.input_shape, output_shape=args.output_shape,
        input_voxel_size=args.input_voxel_size, output_voxel_size=args.output_voxel_size,
        patches_per_epoch=args.patches_per_epoch, jitter_voxels=args.jitter_voxels,
        seed=args.seed, dense_to_sparse_ratio=args.dense_to_sparse_ratio,
        input_norm=json.loads(args.input_norm) if args.input_norm else None,
        postprocess=json.loads(args.postprocess) if args.postprocess else None,
        extra_record=extra or None,
    )
    print(json.dumps({k: record[k] for k in ("manifest_path", "volume_zarr_path", "total_fg_voxels")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
