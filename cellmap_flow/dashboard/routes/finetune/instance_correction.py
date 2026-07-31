"""HTTP handlers for instance-correction workflows.

Three ``*_response(data)`` handlers re-authored from vacc-compat's
``finetune_routes.py`` (lines 661, 979, 1036) for the post-refactor
``routes/finetune/`` layout:

- ``create_instance_correction_response``: seed or reattach a paintable
  annotation layer for a ROI (fresh-seed or reuse-existing modes)
- ``sync_instance_correction_response``: snapshot the paintable zarr
  from MinIO to a local destination
- ``cc3d_relabel_annotation_response``: split a fused label via
  26-connectivity cc3d

Phase 2 of the cellmap-flow upstream integration
(notes/260518_integration_and_scaling_plan.md §4.4).
"""
import logging
import os

import neuroglancer
from flask import jsonify, request

from cellmap_flow.dashboard.finetune_utils import (
    cc3d_relabel_instance_correction,
    create_instance_annotation_volume_from_seg,
    ensure_minio_serving,
    minio_backing_store_populated,
    sync_instance_correction_from_minio,
)
from cellmap_flow.dashboard.routes.finetune.common import rewrite_minio_url_for_proxy
from cellmap_flow.globals import g

logger = logging.getLogger(__name__)


def create_instance_correction_response(data):
    """Create or reattach a paintable annotation layer for a ROI.

    Two modes:

    - **Fresh seed (default, reuse_existing=False)**: reads a uint32 instance
      zarr (from run_postprocess_on_subvolume.py), computes a dilation shell
      around each instance as "confident background", and writes a uint16
      annotation zarr in cellmap-flow's AffinityTargetTransform label scheme
      (0=unannotated, 1=background shell, 2+=instance IDs). Then serves via
      MinIO and wires a writable SegmentationLayer into the viewer.

    - **Reuse existing (reuse_existing=True)**: skips the seeding step
      entirely and reattaches to an already-annotation-formatted zarr.
      Used by the multi-ROI workflow to load any dated snapshot as the
      current paintable layer without redoing the dilation/labeling
      pass. Either:
        - pass `source_zarr_path` to load from an explicit path (the
          preferred path for dated-snapshot workflows), or
        - omit `source_zarr_path` and the route falls back to the
          conventional `<output_dir>/<roi_name>_annotation.zarr`
          location (Patch 41b historical behavior).
      The path must already exist and contain `annotation/s0/`;
      `instance_zarr_path` is ignored in this mode.

    The MinIO bucket object is always named `<roi_name>_annotation.zarr`
    regardless of the source path on disk. This keeps the bucket name
    stable across sessions so save / sync routes can always target the
    same bucket key without knowing which snapshot was loaded.

    POST body:
      roi_name:               str, required (short label, e.g. "roi3")
      reuse_existing:         bool, default False
      instance_zarr_path:     str, required if reuse_existing=False
      source_zarr_path:       str, optional (reuse_existing=True only)
                              explicit path to an existing annotation zarr
      dilation_radius_voxels: int, default 5 (fresh-seed mode only)
      model_name:             str, default "mito_aff_trichocyst"
      annotation_dtype:       "uint16" (default) or "uint32" (fresh-seed only)
      output_dir:             str, default = sibling/instance_corrections/
                              (required if reuse_existing=True without
                              source_zarr_path)
      layer_name:             NG layer name, default "{roi_name}_annotation"

    Returns:
      {success, zarr_path, minio_url, neuroglancer_url, layer_name,
       reload_page, mode: "fresh_seed" or "reuse_existing"}
    """
    try:
        instance_zarr_path = data.get("instance_zarr_path")
        roi_name = data.get("roi_name")
        reuse_existing = bool(data.get("reuse_existing", False))
        source_zarr_path = data.get("source_zarr_path")
        if not roi_name:
            return (
                jsonify({"success": False, "error": "roi_name is required"}),
                400,
            )
        if not reuse_existing:
            if source_zarr_path:
                return (
                    jsonify({
                        "success": False,
                        "error": (
                            "source_zarr_path is only valid when "
                            "reuse_existing=True"
                        ),
                    }),
                    400,
                )
            if not instance_zarr_path:
                return (
                    jsonify({
                        "success": False,
                        "error": (
                            "instance_zarr_path is required when "
                            "reuse_existing=False"
                        ),
                    }),
                    400,
                )
            if not os.path.exists(instance_zarr_path):
                return (
                    jsonify({
                        "success": False,
                        "error": f"instance_zarr_path does not exist: {instance_zarr_path}",
                    }),
                    400,
                )

        dilation_radius = int(data.get("dilation_radius_voxels", 5))
        model_name = data.get("model_name", "mito_aff_trichocyst")
        annotation_dtype = data.get("annotation_dtype", "uint16")

        # Resolve model config to pull input_size / input_voxel_size.
        model_config = None
        for mc in getattr(g, "models_config", []):
            if getattr(mc, "name", None) == model_name:
                model_config = mc.config
                break
        if model_config is None:
            return (
                jsonify({
                    "success": False,
                    "error": f"model '{model_name}' not found in dashboard config",
                }),
                400,
            )
        # model_config.read_shape is in nm; the zarr attr is expected in
        # voxel units (see extract_correction_from_chunk's read_shape_nm =
        # input_size * input_voxel_size). Convert before passing through.
        input_voxel_size = list(model_config.input_voxel_size)
        input_size = [
            int(ns / vs) for ns, vs in zip(model_config.read_shape, input_voxel_size)
        ]

        # Resolve output_dir (MinIO backing store location) and
        # effective_zarr_path (what gets uploaded into the MinIO bucket).
        #
        # - Fresh seed: conventional layout, output_dir =
        #   <sibling>/instance_corrections, effective_zarr_path =
        #   <output_dir>/<roi_name>_annotation.zarr (the file we'll write).
        # - Reuse + source_zarr_path: load from the explicit path;
        #   derive output_dir as the snapshot's grandparent so MinIO's
        #   .minio/ lands alongside instance_corrections, not inside a
        #   per-ROI subdir. Override with `output_dir` if needed.
        # - Reuse without source_zarr_path (Patch 41b historical):
        #   require output_dir, effective_zarr_path is the conventional
        #   <output_dir>/<roi_name>_annotation.zarr.
        if reuse_existing:
            if source_zarr_path:
                # Default output_dir = grandparent of the snapshot
                # (e.g. snapshot at instance_corrections/roi3/roi3_<ts>.zarr
                # -> output_dir = instance_corrections/).
                default_output_dir = os.path.dirname(
                    os.path.dirname(os.path.normpath(source_zarr_path))
                )
                output_dir = data.get("output_dir", default_output_dir)
                effective_zarr_path = source_zarr_path
            else:
                output_dir = data.get("output_dir")
                if not output_dir:
                    return (
                        jsonify({
                            "success": False,
                            "error": (
                                "output_dir is required when "
                                "reuse_existing=True and source_zarr_path "
                                "is not provided"
                            ),
                        }),
                        400,
                    )
                effective_zarr_path = os.path.join(
                    output_dir, f"{roi_name}_annotation.zarr"
                )
        else:
            default_parent = os.path.join(
                os.path.dirname(instance_zarr_path), "instance_corrections"
            )
            output_dir = data.get("output_dir", default_parent)
            effective_zarr_path = os.path.join(
                output_dir, f"{roi_name}_annotation.zarr"
            )
        os.makedirs(output_dir, exist_ok=True)

        # The MinIO bucket object name is always `<roi_name>_annotation.zarr`
        # regardless of the on-disk source path. Keeps the bucket key stable
        # across dated-snapshot reattaches so save/sync routes can always
        # target the same key without knowing which snapshot was loaded.
        mc_target_name = f"{roi_name}_annotation.zarr"

        if reuse_existing:
            # Must already exist and look like a valid annotation zarr.
            if not os.path.isdir(effective_zarr_path):
                return (
                    jsonify({
                        "success": False,
                        "error": (
                            f"reuse_existing=True but {effective_zarr_path} "
                            "does not exist"
                        ),
                    }),
                    404,
                )
            s0_check = os.path.join(effective_zarr_path, "annotation", "s0")
            if not os.path.isdir(s0_check):
                return (
                    jsonify({
                        "success": False,
                        "error": (
                            f"{effective_zarr_path} does not look like an "
                            f"annotation zarr (missing annotation/s0)"
                        ),
                    }),
                    400,
                )
            logger.info(
                f"Reattaching paintable layer for {roi_name}: "
                f"{effective_zarr_path} (reuse_existing)"
            )
        else:
            if os.path.exists(effective_zarr_path):
                return (
                    jsonify({
                        "success": False,
                        "error": f"output already exists: {effective_zarr_path}",
                        "hint": (
                            "Delete it or use a different roi_name/output_dir "
                            "to re-seed. Re-seeding will clobber in-progress edits. "
                            "To reattach to the existing zarr without re-seeding, "
                            "POST with reuse_existing=true."
                        ),
                    }),
                    409,
                )

            # Clobber guard: even if the user-visible zarr path is gone (e.g.
            # the user deleted it intending to start over), MinIO's on-disk
            # backing store may still hold prior brush edits. Re-seeding here
            # would cause ensure_minio_serving's initial `mc mirror <seed>
            # <minio>` to overwrite those edits with the stale seed. Refuse
            # and point at the sync route.
            if minio_backing_store_populated(output_dir, mc_target_name):
                return (
                    jsonify({
                        "success": False,
                        "error": (
                            f"MinIO backing store for {mc_target_name} already populated at "
                            f"{os.path.join(output_dir, '.minio', 'annotations', mc_target_name)} "
                            "— refusing to re-seed because prior brush edits would be lost"
                        ),
                        "hint": (
                            "POST /api/viewer/sync-instance-correction with "
                            "{zarr_path: <effective_zarr_path>} first to pull edits "
                            "into the user-visible zarr, then either (a) keep using "
                            "the pulled zarr as your source of truth, or (b) delete "
                            f"{os.path.join(output_dir, '.minio', 'annotations', mc_target_name)} "
                            "to genuinely start over."
                        ),
                        "output_zarr_path": effective_zarr_path,
                    }),
                    409,
                )

            logger.info(
                f"Creating instance correction for {roi_name}: "
                f"{instance_zarr_path} -> {effective_zarr_path}"
            )

            success, info = create_instance_annotation_volume_from_seg(
                output_zarr_path=effective_zarr_path,
                instance_zarr_path=instance_zarr_path,
                dataset_path=g.dataset_path,
                model_name=model_name,
                input_size=input_size,
                input_voxel_size=input_voxel_size,
                dilation_radius_voxels=dilation_radius,
                annotation_dtype=annotation_dtype,
            )
            if not success:
                return jsonify({"success": False, "error": info}), 500

        # MinIO + viewer wiring. mc_target_name pins the bucket object key
        # to the stable `<roi_name>_annotation.zarr` name regardless of the
        # on-disk source filename (important for multi-ROI workflows where
        # the source is a dated snapshot).
        volume_id = f"{roi_name}_instance_annotation"
        minio_url = ensure_minio_serving(
            effective_zarr_path,
            volume_id,
            output_base_dir=output_dir,
            mc_target_name=mc_target_name,
        )
        minio_url = rewrite_minio_url_for_proxy(minio_url, request)

        if not hasattr(g, "viewer") or g.viewer is None:
            return (
                jsonify({"success": False, "error": "viewer not initialized"}),
                400,
            )

        layer_name = data.get("layer_name", f"{roi_name}_annotation")
        with g.viewer.txn() as s:
            if layer_name in s.layers:
                del s.layers[layer_name]
            source_config = {
                "url": f"s3+{minio_url}/annotation",
                "subsources": {
                    "default": {"writingEnabled": True},
                    "bounds": {},
                },
            }
            s.layers[layer_name] = neuroglancer.SegmentationLayer(
                source=source_config,
            )
        logger.info(
            f"Added paintable layer {layer_name} -> {minio_url}/annotation"
        )

        return jsonify({
            "success": True,
            "mode": "reuse_existing" if reuse_existing else "fresh_seed",
            "zarr_path": effective_zarr_path,
            "minio_url": minio_url,
            "neuroglancer_url": f"{minio_url}/annotation",
            "layer_name": layer_name,
            "reload_page": True,
        })
    except Exception as e:
        logger.error(f"Error creating instance correction: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500




def sync_instance_correction_response(data):
    """Snapshot a paintable instance-correction zarr from MinIO to a local
    destination.

    Pulls the current contents of the MinIO-backed annotation zarr (where
    NG brush edits actually land) into a destination zarr path. Call this:
      - before Run 12 training-data extraction, so `finetune_cli` can read
        `annotation/s0` directly from the user-visible path;
      - before any dashboard restart with live edits, so `ensure_minio_serving`
        can't clobber them during its next initial mirror;
      - any time you want a durable on-disk snapshot of in-progress
        proofreading state (e.g. for rollback safety or dated audit).

    Uses `s3fs` + `zarr.copy_store` under the hood (via
    `_diff_and_sync_chunks`) — does NOT shell out to `mc`, which is not
    on PATH on h2node10.

    POST body:
      zarr_path: str, required. Absolute path to the user-visible zarr
          (e.g. `/.../instance_corrections/roi3_annotation.zarr`). Only
          used to derive the MinIO bucket key from its basename; the
          file itself is not opened.
      dst_path:  str, optional. Absolute path to write the snapshot to.
          Defaults to `zarr_path` (in-place pull-back). Prefer a fresh
          dated path (e.g. `.../roi3_annotation_FINAL_session14_<ts>.zarr`)
          to avoid any hardlink / aliasing hazards with provenance
          snapshots — see the helper docstring for the inode-sharing
          detail.

    Returns:
      {success, zarr_path, dst_path, chunks_synced, chunks_removed}
    """
    try:
        zarr_path = data.get("zarr_path")
        dst_path = data.get("dst_path")
        if not zarr_path:
            return (
                jsonify({"success": False, "error": "zarr_path is required"}),
                400,
            )

        success, info = sync_instance_correction_from_minio(
            zarr_path, dst_path=dst_path
        )
        if not success:
            return jsonify({"success": False, "error": info}), 500

        return jsonify({"success": True, **info})
    except Exception as e:
        logger.error(f"Error syncing instance correction: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500



def cc3d_relabel_annotation_response(data):
    """Split a single label in a paintable instance-correction zarr via
    26-connectivity cc3d.

    Typical workflow: the user has erased a thin bridge between two fused
    mitos in NG's brush tool (still sharing `target_label`), then POSTs
    this route with that label. The server reads the MinIO-backed
    annotation/s0, snapshots it to a local DirectoryStore for rollback,
    runs cc3d on the target mask, keeps the largest component under
    `target_label`, and reassigns all smaller components to fresh unused
    instance IDs starting at `max(existing) + 1`. Then writes the full
    array back to MinIO. After the POST completes, the user must hard
    reload the NG tab to see the new split colors (NG does not auto-
    invalidate segmentation chunks on back-channel writes).

    POST body:
      zarr_path:     str, required. Absolute path to the user-visible zarr.
      target_label:  int, required. The instance ID to split (must be >= 2).
      snapshot_dir:  str, optional. Where to drop rollback snapshots.
                     Defaults to `<parent_of_zarr>/snapshots/`.

    Returns:
      {success, zarr_path, target_label, n_components, kept_voxels,
       splits: [{new_label, voxels}, ...], snapshot_path,
       reload_hint: "hard reload NG tab to see split"}
    """
    try:
        zarr_path = data.get("zarr_path")
        target_label = data.get("target_label")
        if not zarr_path or target_label is None:
            return (
                jsonify({
                    "success": False,
                    "error": "zarr_path and target_label are required",
                }),
                400,
            )
        snapshot_dir = data.get("snapshot_dir")

        success, info = cc3d_relabel_instance_correction(
            zarr_path=zarr_path,
            target_label=int(target_label),
            snapshot_dir=snapshot_dir,
        )
        if not success:
            return jsonify({"success": False, "error": info}), 500

        return jsonify({
            "success": True,
            "reload_hint": "hard reload NG tab to see split",
            **info,
        })
    except Exception as e:
        logger.error(f"Error in cc3d-relabel-annotation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500
