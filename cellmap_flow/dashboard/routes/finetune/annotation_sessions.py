import json
import logging
import os
from datetime import datetime

from flask import jsonify

from cellmap_flow.dashboard.finetune_utils import ensure_minio_serving
from cellmap_flow.dashboard.routes.finetune.common import ensure_corrections_storage
from cellmap_flow.dashboard.routes.finetune.overlay import refresh_annotated_regions_layer
from cellmap_flow.globals import g

logger = logging.getLogger(__name__)


def _register_annotation_volume(volume_id, **volume_data):
    if not hasattr(g, "annotation_volumes"):
        g.annotation_volumes = {}
    g.annotation_volumes[volume_id] = {
        **volume_data,
        "extracted_chunks": set(),
        "chunk_sync_state": {},
    }


def list_existing_sessions_response(data):
    try:
        output_path = data.get("output_path", "")
        if not output_path:
            return jsonify({"success": False, "error": "output_path required"}), 400

        base = os.path.expanduser(output_path)
        if not os.path.isdir(base):
            return jsonify({"success": True, "sessions": []})

        sessions = []
        for entry in sorted(os.listdir(base), reverse=True):
            session_dir = os.path.join(base, entry)
            corrections_dir = os.path.join(session_dir, "corrections")
            if not os.path.isdir(corrections_dir):
                continue

            volumes = []
            chunks = []
            for item in os.listdir(corrections_dir):
                if not item.endswith(".zarr"):
                    continue
                full = os.path.join(corrections_dir, item)
                if "_chunk_" in item:
                    chunks.append(item)
                else:
                    volumes.append({"volume_id": item.replace(".zarr", ""), "path": full})

            if volumes or chunks:
                sessions.append(
                    {
                        "session_id": entry,
                        "session_path": session_dir,
                        "volumes": volumes,
                        "chunk_count": len(chunks),
                    }
                )

        return jsonify({"success": True, "sessions": sessions})
    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


def load_existing_volume_response(data):
    try:
        import shutil

        from cellmap_flow.dashboard.finetune_utils import minio_state

        source_session_path = data.get("source_session_path")
        output_path = data.get("output_path")
        if not source_session_path or not output_path:
            return jsonify(
                {"success": False, "error": "source_session_path and output_path required"}
            ), 400

        source_session_path = os.path.expanduser(source_session_path)
        source_corrections = os.path.join(source_session_path, "corrections")
        if not os.path.isdir(source_corrections):
            return jsonify({"success": False, "error": f"No corrections found in {source_session_path}"}), 404

        volume_entries = [
            entry
            for entry in os.listdir(source_corrections)
            if entry.endswith(".zarr") and "_chunk_" not in entry
        ]
        if not volume_entries:
            return jsonify(
                {"success": False, "error": f"No annotation volume found in {source_corrections}"}
            ), 404

        volume_dir = volume_entries[0]
        volume_id = volume_dir.replace(".zarr", "")
        new_session_path, new_corrections = ensure_corrections_storage(output_path)

        copied = []
        for item in os.listdir(source_corrections):
            if not item.endswith(".zarr"):
                continue
            src = os.path.join(source_corrections, item)
            dst = os.path.join(new_corrections, item)
            if os.path.exists(dst):
                logger.info(f"Skipping {item} (already exists in target)")
                continue
            shutil.copytree(src, dst)
            copied.append(item)

        source_minio = os.path.join(source_corrections, ".minio")
        new_minio = os.path.join(new_corrections, ".minio")
        copied_minio = False
        if os.path.isdir(source_minio):
            if minio_state.get("process") is not None and minio_state["process"].poll() is None:
                logger.warning(
                    "MinIO already running with a different output_base; cannot rebind. "
                    "Falling back to mc mirror upload - painted data may be incomplete "
                    "if the source had unsynced chunks."
                )
            elif not os.path.exists(new_minio):
                shutil.copytree(source_minio, new_minio)
                copied_minio = True

        lineage_file = os.path.join(new_session_path, "loaded_from.json")
        with open(lineage_file, "w") as f:
            json.dump(
                {
                    "source_session_path": source_session_path,
                    "loaded_at": datetime.now().isoformat(),
                    "copied_files": copied,
                },
                f,
                indent=2,
            )

        new_volume_path = os.path.join(new_corrections, volume_dir)
        zattrs_file = os.path.join(new_volume_path, ".zattrs")
        volume_meta = {}
        if os.path.exists(zattrs_file):
            with open(zattrs_file) as f:
                volume_meta = json.load(f)

        s0_dir = os.path.join(new_volume_path, "annotation", "s0")
        s0_count = 0
        if os.path.isdir(s0_dir):
            s0_count = sum(1 for entry in os.listdir(s0_dir) if not entry.startswith("."))

        minio_url = ensure_minio_serving(new_volume_path, volume_id, output_base_dir=new_corrections)
        _register_annotation_volume(
            volume_id,
            zarr_path=new_volume_path,
            model_name=volume_meta.get("model_name"),
            output_size=volume_meta.get("chunk_size"),
            input_size=volume_meta.get("input_size"),
            input_voxel_size=volume_meta.get("input_voxel_size"),
            output_voxel_size=volume_meta.get("output_voxel_size"),
            dataset_path=volume_meta.get("dataset_path"),
            dataset_offset_nm=volume_meta.get("dataset_offset_nm"),
            corrections_dir=new_corrections,
        )
        refresh_annotated_regions_layer()

        return jsonify(
            {
                "success": True,
                "volume_id": volume_id,
                "new_session_path": new_session_path,
                "zarr_path": new_volume_path,
                "minio_url": minio_url,
                "neuroglancer_url": f"{minio_url}/annotation",
                "copied_count": len(copied),
                "copied_minio": copied_minio,
                "painted_chunk_count": s0_count,
                "metadata": volume_meta,
            }
        )
    except Exception as e:
        logger.error(f"Error loading existing volume: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
