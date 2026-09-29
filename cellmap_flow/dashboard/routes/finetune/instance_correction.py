"""HTTP handlers for instance-correction workflows.

Three ``*_response(data)`` handlers:

- ``create_instance_correction_response``: seed or reattach a paintable
  annotation layer for a ROI (fresh-seed or reuse-existing modes)
- ``sync_instance_correction_response``: snapshot the paintable zarr
  from MinIO to a local destination
- ``cc3d_relabel_annotation_response``: split a fused label via
  26-connectivity cc3d

The volumes are ordinary annotation volumes (``type: annotation_volume``),
registered under the id their MinIO bucket key names, so the periodic sync,
the pull before every mirror, session listing and the overlay handle them
like any other.
"""
import logging
import os

import neuroglancer
import numpy as np
import zarr
from flask import jsonify

from cellmap_flow.dashboard.finetune_utils import (
    cc3d_relabel_instance_correction,
    create_instance_annotation_volume_from_seg,
    ensure_minio_serving,
    minio_backing_store_populated,
    sync_instance_correction_from_minio,
)
from cellmap_flow.dashboard.routes.finetune.annotation_core import _get_selected_model_config
from cellmap_flow.globals import current_input_norm_config, current_postprocess_config, g
from cellmap_flow.utils.model_geometry import resolve_model_geometry

logger = logging.getLogger(__name__)

# The type the first version of these volumes wrote. Nothing reads it:
# every reader wants "annotation_volume".
_LEGACY_TYPE = "instance_annotation_volume"


def _error(message, status=400, **extra):
    return jsonify({"success": False, "error": message, **extra}), status


def _seed_geometry(model_name, dataset_path):
    """The model geometry a seeded volume records, without building the model.

    The same sources as create-volume (the running server, then the geometry
    cache) and the same snapping of the input voxel size to a raw scale.
    Returns ``(kwargs, None)`` or ``(None, error_response)``.
    """
    from cellmap_flow.utils.neuroglancer_utils import get_raw_closest_scale

    model_config, error_response = _get_selected_model_config(model_name)
    if error_response is not None:
        return None, error_response
    config = resolve_model_geometry(model_name, model_config)
    claimed_input_voxel_size = np.array(config.input_voxel_size, dtype=float)
    claimed_output_voxel_size = np.array(config.output_voxel_size, dtype=float)
    try:
        input_voxel_size = np.array(
            get_raw_closest_scale(dataset_path, tuple(claimed_input_voxel_size))
            or claimed_input_voxel_size
        )
    except Exception:
        input_voxel_size = claimed_input_voxel_size
    return {
        "input_size": (np.array(config.read_shape) / claimed_input_voxel_size).astype(int).tolist(),
        "input_voxel_size": input_voxel_size.tolist(),
        # One chunk per training sample, as in every annotation volume.
        "chunk_size": (np.array(config.write_shape) / claimed_output_voxel_size).astype(int).tolist(),
        "claimed_input_voxel_size": claimed_input_voxel_size.tolist(),
        "claimed_output_voxel_size": claimed_output_voxel_size.tolist(),
    }, None


def _register_volume(volume_id, zarr_path, corrections_dir, minio_url):
    """Record the volume as annotation_volume records are kept, keeping the
    chunk state the pull before the mirror just recorded."""
    attrs = dict(zarr.open(zarr_path, mode="r").attrs)
    entry = g.annotation_volumes.setdefault(volume_id, {"chunk_sync_state": {}})
    entry.update(
        zarr_path=zarr_path,
        model_name=attrs.get("model_name", ""),
        output_size=attrs.get("chunk_size"),
        input_size=attrs.get("input_size"),
        input_voxel_size=attrs.get("input_voxel_size"),
        output_voxel_size=attrs.get("output_voxel_size"),
        dataset_path=attrs.get("dataset_path", ""),
        dataset_offset_nm=attrs.get("dataset_offset_nm"),
        corrections_dir=corrections_dir,
        minio_url=minio_url,
    )


def create_instance_correction_response(data):
    """Create or reattach a paintable annotation layer for a ROI.

    Two modes:

    - **Fresh seed (default, reuse_existing=False)**: reads an instance
      zarr, computes a dilation shell around each instance as "confident
      background", and writes a uint16 annotation zarr in cellmap-flow's
      AffinityTargetTransform label scheme (0=unannotated, 1=background
      shell, 2+=instance IDs). Then serves via MinIO and wires a writable
      SegmentationLayer into the viewer.

    - **Reuse existing (reuse_existing=True)**: skips the seeding step
      entirely and reattaches to an already-annotation-formatted zarr.
      Used by the multi-ROI workflow to load any dated snapshot as the
      current paintable layer without redoing the dilation/labeling
      pass. Either:
        - pass `source_zarr_path` to load from an explicit path (the
          preferred path for dated-snapshot workflows), or
        - omit `source_zarr_path` and the route falls back to the
          conventional `<output_dir>/<roi_name>_annotation.zarr`
          location.
      The path must already exist and contain `annotation/s0/`;
      `instance_zarr_path` is ignored in this mode. Strokes MinIO already
      holds under the ROI's bucket key are pulled into that zarr before it
      is mirrored, as for every annotation volume, so reattaching never
      drops unsaved edits; to start again from an older snapshot, give it
      a new roi_name.

    The MinIO bucket object is always named `<roi_name>_annotation.zarr`
    regardless of the source path on disk. This keeps the bucket name
    stable across sessions so save / sync routes can always target the
    same bucket key without knowing which snapshot was loaded. The volume
    is registered as `<roi_name>_annotation`, the key without ".zarr".

    POST body:
      roi_name:               str, required (short label, e.g. "roi3")
      reuse_existing:         bool, default False
      instance_zarr_path:     str, required if reuse_existing=False
      source_zarr_path:       str, optional (reuse_existing=True only)
                              explicit path to an existing annotation zarr
      dilation_radius_voxels: int, default 5 (fresh-seed mode only)
      model_name:             str, required if reuse_existing=False: the
                              model whose geometry the volume records
      annotation_dtype:       "uint16" (default) or "uint32" (fresh-seed only)
      output_dir:             str, default = sibling/instance_corrections/
                              (required if reuse_existing=True without
                              source_zarr_path)
      layer_name:             NG layer name, default "{roi_name}_annotation"

    Returns:
      {success, volume_id, zarr_path, minio_url, neuroglancer_url,
       layer_name, reload_page, mode: "fresh_seed" or "reuse_existing"}
    """
    try:
        instance_zarr_path = data.get("instance_zarr_path")
        roi_name = data.get("roi_name")
        reuse_existing = bool(data.get("reuse_existing", False))
        source_zarr_path = data.get("source_zarr_path")
        model_name = data.get("model_name")
        if not roi_name:
            return _error("roi_name is required")
        if getattr(g, "viewer", None) is None:
            return _error("viewer not initialized")
        if not reuse_existing:
            if source_zarr_path:
                return _error("source_zarr_path is only valid when reuse_existing=True")
            if not instance_zarr_path:
                return _error("instance_zarr_path is required when reuse_existing=False")
            if not os.path.exists(instance_zarr_path):
                return _error(f"instance_zarr_path does not exist: {instance_zarr_path}")
            if not model_name:
                return _error("model_name is required when reuse_existing=False")

        dilation_radius = int(data.get("dilation_radius_voxels", 5))
        annotation_dtype = data.get("annotation_dtype", "uint16")

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
        # - Reuse without source_zarr_path:
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
            root = zarr.open(effective_zarr_path, mode="r+")
            if root.attrs.get("type") == _LEGACY_TYPE:
                root.attrs["type"] = "annotation_volume"
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

            dataset_path = getattr(g, "dataset_path", None)
            if not dataset_path:
                return _error("No dataset path configured")
            geometry, error_response = _seed_geometry(model_name, dataset_path)
            if error_response is not None:
                return error_response
            success, info = create_instance_annotation_volume_from_seg(
                output_zarr_path=effective_zarr_path,
                instance_zarr_path=instance_zarr_path,
                dataset_path=dataset_path,
                model_name=model_name,
                dilation_radius_voxels=dilation_radius,
                annotation_dtype=annotation_dtype,
                input_norm_config=current_input_norm_config(),
                postprocess_config=current_postprocess_config(),
                **geometry,
            )
            if not success:
                return jsonify({"success": False, "error": info}), 500

        # MinIO + viewer wiring. mc_target_name pins the bucket object key
        # to the stable `<roi_name>_annotation.zarr` name regardless of the
        # on-disk source filename (important for multi-ROI workflows where
        # the source is a dated snapshot). The sync finds a volume's chunks
        # by its id, so the id is that key without ".zarr". A record from
        # an earlier attach may point at another snapshot: drop it, so the
        # pull before the mirror fills this zarr from the whole bucket.
        volume_id = mc_target_name[: -len(".zarr")]
        g.annotation_volumes.pop(volume_id, None)
        minio_url = ensure_minio_serving(
            effective_zarr_path,
            volume_id,
            output_base_dir=output_dir,
            mc_target_name=mc_target_name,
        )
        _register_volume(volume_id, effective_zarr_path, output_dir, minio_url)

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
            "volume_id": volume_id,
            "zarr_path": effective_zarr_path,
            "minio_url": minio_url,
            "neuroglancer_url": f"{minio_url}/annotation",
            "layer_name": layer_name,
            "reload_page": True,
        })
    except Exception as e:
        logger.error(f"Error creating instance correction: {e}", exc_info=True)
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
