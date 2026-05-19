from flask import Blueprint, request

from cellmap_flow.dashboard.routes.finetune.annotation import (
    add_crop_to_viewer_response,
    create_annotation_crop_response,
    create_annotation_volume_response,
    get_finetune_models_response,
    get_user_prefs_response,
    get_view_center_response,
    list_existing_sessions_response,
    load_existing_volume_response,
    refresh_annotated_regions_layer,
    set_user_prefs_response,
    sync_annotations_manually_response,
)
from cellmap_flow.dashboard.routes.finetune.annotation_sessions import (
    get_resume_progress_response,
)
from cellmap_flow.dashboard.routes.finetune.instance_correction import (
    cc3d_relabel_annotation_response,
    create_instance_correction_response,
    sync_instance_correction_response,
)
from cellmap_flow.dashboard.routes.finetune.training import (
    cancel_job_response,
    get_job_logs_response,
    get_job_status_response,
    get_inference_server_status_response,
    list_finetuning_jobs_response,
    restart_finetuning_job_response,
    stop_training_early_response,
    stream_job_logs_response,
    submit_finetuning_response,
)
from cellmap_flow.dashboard.routes.finetune.viewer import (
    add_finetuned_layer_to_viewer_response,
)
from cellmap_flow.dashboard.routes.finetune.yaml_crops import (
    get_load_crops_progress_response,
    load_crops_from_yaml_response,
    read_yaml_file_response,
)

finetune_bp = Blueprint("finetune", __name__)


@finetune_bp.route("/api/finetune/models", methods=["GET"])
def get_finetune_models():
    return get_finetune_models_response()


@finetune_bp.route("/api/finetune/view-center", methods=["GET"])
def get_view_center():
    return get_view_center_response()


@finetune_bp.route("/api/finetune/create-crop", methods=["POST"])
def create_annotation_crop():
    return create_annotation_crop_response(request.get_json() or {})


@finetune_bp.route("/api/finetune/create-volume", methods=["POST"])
def create_annotation_volume():
    return create_annotation_volume_response(request.get_json() or {})


@finetune_bp.route("/api/finetune/load-crops", methods=["POST"])
def load_crops_from_yaml():
    return load_crops_from_yaml_response(request.get_json() or {})


@finetune_bp.route("/api/finetune/read-yaml", methods=["GET"])
def read_yaml_file():
    return read_yaml_file_response(request.args.get("path"))


@finetune_bp.route("/api/finetune/load-crops-progress", methods=["GET"])
def get_load_crops_progress():
    return get_load_crops_progress_response(request.args.get("load_id"))


@finetune_bp.route("/api/finetune/user-prefs", methods=["GET"])
def get_user_prefs():
    return get_user_prefs_response()


@finetune_bp.route("/api/finetune/user-prefs", methods=["POST"])
def set_user_prefs():
    return set_user_prefs_response(request.get_json() or {})


@finetune_bp.route("/api/finetune/list-existing-sessions", methods=["POST"])
def list_existing_sessions():
    return list_existing_sessions_response(request.get_json() or {})


@finetune_bp.route("/api/finetune/load-existing-volume", methods=["POST"])
def load_existing_volume():
    return load_existing_volume_response(request.get_json() or {})


@finetune_bp.route("/api/finetune/load-existing-volume-progress", methods=["GET"])
def load_existing_volume_progress():
    return get_resume_progress_response(request.args.get("load_id"))


@finetune_bp.route("/api/finetune/add-to-viewer", methods=["POST"])
def add_crop_to_viewer():
    return add_crop_to_viewer_response(request.get_json() or {})


@finetune_bp.route("/api/finetune/sync-annotations", methods=["POST"])
def sync_annotations_manually():
    return sync_annotations_manually_response(request.get_json() or {})


@finetune_bp.route("/api/finetune/submit", methods=["POST"])
def submit_finetuning():
    return submit_finetuning_response(request.get_json() or {})


@finetune_bp.route("/api/finetune/jobs", methods=["GET"])
def get_finetuning_jobs():
    return list_finetuning_jobs_response()


@finetune_bp.route("/api/finetune/job/<job_id>/status", methods=["GET"])
def get_job_status(job_id):
    return get_job_status_response(job_id)


@finetune_bp.route("/api/finetune/job/<job_id>/logs", methods=["GET"])
def get_job_logs(job_id):
    return get_job_logs_response(job_id)


@finetune_bp.route("/api/finetune/job/<job_id>/logs/stream", methods=["GET"])
def stream_job_logs(job_id):
    return stream_job_logs_response(job_id)


@finetune_bp.route("/api/finetune/job/<job_id>/cancel", methods=["POST"])
def cancel_job(job_id):
    return cancel_job_response(job_id)


@finetune_bp.route("/api/finetune/job/<job_id>/stop-early", methods=["POST"])
def stop_training_early(job_id):
    return stop_training_early_response(job_id)


@finetune_bp.route("/api/finetune/job/<job_id>/inference-server", methods=["GET"])
def get_inference_server_status(job_id):
    return get_inference_server_status_response(job_id)


@finetune_bp.route("/api/viewer/add-finetuned-layer", methods=["POST"])
def add_finetuned_layer_to_viewer():
    return add_finetuned_layer_to_viewer_response(request.get_json() or {})


@finetune_bp.route("/api/viewer/create-instance-correction", methods=["POST"])
def create_instance_correction():
    return create_instance_correction_response(request.get_json() or {})


@finetune_bp.route("/api/viewer/sync-instance-correction", methods=["POST"])
def sync_instance_correction():
    return sync_instance_correction_response(request.get_json() or {})


@finetune_bp.route("/api/viewer/cc3d-relabel-annotation", methods=["POST"])
def cc3d_relabel_annotation():
    return cc3d_relabel_annotation_response(request.get_json() or {})


@finetune_bp.route("/api/finetune/job/<job_id>/restart", methods=["POST"])
def restart_finetuning_job(job_id):
    return restart_finetuning_job_response(job_id, request.get_json() or {})


__all__ = ["finetune_bp", "refresh_annotated_regions_layer"]
