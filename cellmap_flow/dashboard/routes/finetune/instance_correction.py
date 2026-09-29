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
import re

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
from cellmap_flow.dashboard.routes.finetune.common import rewrite_minio_url_for_proxy
from cellmap_flow.globals import current_input_norm_config, current_postprocess_config, g
from cellmap_flow.io.multiscale import closest_raw_scale
from cellmap_flow.utils.model_geometry import resolve_model_geometry

logger = logging.getLogger(__name__)

# The type the first version of these volumes wrote. Nothing reads it:
# every reader wants "annotation_volume".
_LEGACY_TYPE = "instance_annotation_volume"


def _error(message, status=400, **extra):
    return jsonify({"success": False, "error": message, **extra}), status


# Where requests may write. The dashboard listens on every interface, and
# these routes create directories, write seeds and copy MinIO objects over
# whatever paths they are given, so each path must be a zarr, in a
# directory that already exists, next to the volume it belongs to.

# roi_name becomes a file name and a bucket key.
_ROI_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class _Refused(ValueError):
    """Request input these routes will not act on; answered with 400."""


def _real(path):
    return os.path.realpath(os.path.expanduser(str(path)))


def _is_zarr(path):
    return any(os.path.isfile(os.path.join(path, key)) for key in (".zgroup", ".zarray"))


def _existing_dir(path, name):
    real = _real(path)
    if not os.path.isdir(real):
        raise _Refused(f"{name} must be an existing directory: {path}")
    return real


def _zarr_target(path, name, beside=None):
    """``path`` resolved, if a request may write a zarr there.

    It must be named *.zarr once links are resolved (checked before the disk
    is looked at), lie in an existing directory -- the directory of
    ``beside``, when given -- and, if it exists, be a zarr.
    """
    if not path:
        raise _Refused(f"{name} is required")
    real = _real(path)
    if not real.endswith(".zarr"):
        raise _Refused(f"{name} must be a .zarr path: {path}")
    parent = os.path.dirname(real)
    if beside is not None and parent != os.path.dirname(beside):
        raise _Refused(f"{name} must be in the same directory as {beside}")
    if not os.path.isdir(parent):
        raise _Refused(f"the directory of {name} does not exist: {path}")
    if os.path.exists(real) and not _is_zarr(real):
        raise _Refused(f"{name} exists and is not a zarr: {path}")
    return real


def _int(data, key, default=None):
    value = data.get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        raise _Refused(f"{key} must be an integer, got {value!r}")


def _seed_geometry(model_name, dataset_path):
    """The model geometry a seeded volume records, without building the model.

    The same sources as create-volume (the running server, then the geometry
    cache) and the same snapping of the input voxel size to a raw scale.
    Returns ``(kwargs, None)`` or ``(None, error_response)``.
    """
    model_config, error_response = _get_selected_model_config(model_name)
    if error_response is not None:
        return None, error_response
    config = resolve_model_geometry(model_name, model_config)
    claimed_input_voxel_size = np.array(config.input_voxel_size, dtype=float)
    claimed_output_voxel_size = np.array(config.output_voxel_size, dtype=float)
    input_voxel_size = np.array(
        closest_raw_scale(dataset_path, tuple(claimed_input_voxel_size))
        or claimed_input_voxel_size
    )
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
        if not _ROI_NAME.match(str(roi_name)):
            return _error("roi_name may hold only letters, digits, '_', '-' and '.'")
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

        dilation_radius = _int(data, "dilation_radius_voxels", 5)
        annotation_dtype = data.get("annotation_dtype", "uint16")
        if annotation_dtype not in ("uint16", "uint32"):
            return _error("annotation_dtype must be uint16 or uint32")

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
        #
        # A requested output_dir must already exist. Only the default
        # instance_corrections/, beside an instance zarr that exists, is
        # made here.
        if reuse_existing:
            if source_zarr_path:
                source_zarr_path = _real(source_zarr_path)
                if not source_zarr_path.endswith(".zarr"):
                    return _error("source_zarr_path must be a .zarr path")
                # Default output_dir = grandparent of the snapshot
                # (e.g. snapshot at instance_corrections/roi3/roi3_<ts>.zarr
                # -> output_dir = instance_corrections/).
                default_output_dir = os.path.dirname(os.path.dirname(source_zarr_path))
                output_dir = _existing_dir(data.get("output_dir", default_output_dir), "output_dir")
                effective_zarr_path = source_zarr_path
            else:
                if not data.get("output_dir"):
                    return _error(
                        "output_dir is required when reuse_existing=True "
                        "and source_zarr_path is not provided"
                    )
                output_dir = _existing_dir(data["output_dir"], "output_dir")
                effective_zarr_path = os.path.join(
                    output_dir, f"{roi_name}_annotation.zarr"
                )
        else:
            if data.get("output_dir"):
                output_dir = _existing_dir(data["output_dir"], "output_dir")
            else:
                output_dir = os.path.join(
                    os.path.dirname(_real(instance_zarr_path)), "instance_corrections"
                )
                os.makedirs(output_dir, exist_ok=True)
            effective_zarr_path = os.path.join(
                output_dir, f"{roi_name}_annotation.zarr"
            )

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
        minio_url = rewrite_minio_url_for_proxy(minio_url)
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
    except _Refused as e:
        return _error(str(e))
    except Exception as e:
        logger.error(f"Error creating instance correction: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


def sync_instance_correction_response(data):
    """Snapshot a paintable instance-correction zarr from MinIO to a local
    destination.

    Copies the MinIO object that brush edits land in, metadata and all, to
    a zarr on disk: a dated snapshot for rollback or audit, or a copy to
    train from. The served volume itself is kept current by the periodic
    sync, like any annotation volume. Uses s3fs, not `mc`.

    POST body:
      zarr_path: str, required. The volume, e.g.
          `/.../instance_corrections/roi3_annotation.zarr`; its basename is
          the MinIO bucket key.
      dst_path:  str, optional. Where to write the copy: a `.zarr` in the
          same directory as `zarr_path`, new or an existing zarr. Defaults
          to `zarr_path`. Prefer a fresh dated path (e.g.
          `.../roi3_annotation_<ts>.zarr`); see the helper docstring for
          why an in-place copy can corrupt hardlinked snapshots.

    Returns:
      {success, zarr_path, dst_path, keys_copied, keys_skipped, bytes_copied}
    """
    try:
        zarr_path = _zarr_target(data.get("zarr_path"), "zarr_path")
        dst_path = data.get("dst_path")
        if dst_path:
            dst_path = _zarr_target(dst_path, "dst_path", beside=zarr_path)

        success, info = sync_instance_correction_from_minio(
            zarr_path, dst_path=dst_path
        )
        if not success:
            return jsonify({"success": False, "error": info}), 500

        return jsonify({"success": True, **info})
    except _Refused as e:
        return _error(str(e))
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
      snapshot_dir:  str, optional. Where to drop rollback snapshots: a
                     directory beside `zarr_path`. Defaults to
                     `<parent_of_zarr>/snapshots/`.

    Returns:
      {success, zarr_path, target_label, n_components, kept_voxels,
       splits: [{new_label, voxels}, ...], snapshot_path,
       reload_hint: "hard reload NG tab to see split"}
    """
    try:
        if not data.get("zarr_path") or data.get("target_label") is None:
            return _error("zarr_path and target_label are required")
        zarr_path = _zarr_target(data["zarr_path"], "zarr_path")
        target_label = _int(data, "target_label")
        if target_label < 2:
            return _error("target_label must be >= 2: 0 is unannotated and 1 is background")
        default_snapshot_dir = os.path.join(os.path.dirname(zarr_path), "snapshots")
        snapshot_dir = _real(data.get("snapshot_dir") or default_snapshot_dir)
        if os.path.dirname(snapshot_dir) != os.path.dirname(zarr_path):
            return _error(f"snapshot_dir must be in the same directory as {zarr_path}")
        if os.path.exists(snapshot_dir) and not os.path.isdir(snapshot_dir):
            return _error(f"snapshot_dir is not a directory: {snapshot_dir}")

        success, info = cc3d_relabel_instance_correction(
            zarr_path=zarr_path,
            target_label=target_label,
            snapshot_dir=snapshot_dir,
        )
        if not success:
            return jsonify({"success": False, "error": info}), 500

        return jsonify({
            "success": True,
            "reload_hint": "hard reload NG tab to see split",
            **info,
        })
    except _Refused as e:
        return _error(str(e))
    except Exception as e:
        logger.error(f"Error in cc3d-relabel-annotation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500
