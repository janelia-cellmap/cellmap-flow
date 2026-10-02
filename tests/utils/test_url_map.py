"""Every URL the dashboard answers, and the methods it takes on each.

Pages, scripts and the finetune tab call these by path, so a route that
moves or goes away breaks a caller without a failing import. The Phase 3
refactors keep every one (the K8/K11 guard); a change here is a change of
the dashboard's HTTP surface. Endpoint names are Flask's own and may change
when a blueprint is reorganized.
"""

URLS = {
    "/": "GET",
    "/api/bbx-generator": "POST",
    "/api/bbx-generator/finalize": "POST",
    "/api/bbx-generator/status": "GET",
    "/api/bioimage-models": "GET",
    "/api/bioimage-models/refresh": "POST",
    "/api/blockwise-config": "GET, POST",
    "/api/blockwise/generate": "POST",
    "/api/blockwise/precheck": "POST",
    "/api/blockwise/submit": "POST",
    "/api/blockwise/validate": "POST",
    "/api/create-model-config": "POST",
    "/api/export-config": "GET",
    "/api/finetune/add-to-viewer": "POST",
    "/api/finetune/create-volume": "POST",
    "/api/finetune/good-regions": "GET",
    "/api/finetune/good-regions/delete": "POST",
    "/api/finetune/good-regions/mark-view": "POST",
    "/api/finetune/job/<job_id>/cancel": "POST",
    "/api/finetune/job/<job_id>/logs": "GET",
    "/api/finetune/job/<job_id>/logs/stream": "GET",
    "/api/finetune/job/<job_id>/restart": "POST",
    "/api/finetune/job/<job_id>/status": "GET",
    "/api/finetune/job/<job_id>/stop-early": "POST",
    "/api/finetune/jobs": "GET",
    "/api/finetune/list-existing-sessions": "POST",
    "/api/finetune/load-crops": "POST",
    "/api/finetune/load-crops-progress": "GET",
    "/api/finetune/load-existing-volume": "POST",
    "/api/finetune/load-existing-volume-progress": "GET",
    "/api/finetune/models": "GET",
    "/api/finetune/read-yaml": "GET",
    "/api/finetune/refresh-annotated-regions": "POST",
    "/api/finetune/submit": "POST",
    "/api/finetune/sync-annotations": "POST",
    "/api/finetune/user-prefs": "GET, POST",
    "/api/finetune/view-labels/background": "POST",
    "/api/finetune/view-labels/seed": "POST",
    "/api/finetune/view-labels/sources": "GET",
    "/api/finetune/view-labels/split": "POST",
    "/api/gpu-queues": "GET",
    "/api/huggingface-models": "GET",
    "/api/huggingface-models/refresh": "POST",
    "/api/job-logs": "GET",
    "/api/logs/stream": "GET",
    "/api/model-config-types": "GET",
    "/api/model_advice": "GET",
    "/api/models": "POST",
    "/api/pipeline": "PUT",
    "/api/pipeline/apply": "POST",
    "/api/process": "POST",
    "/api/review/next": "GET",
    "/api/review/open": "POST",
    "/api/review/pick_stream": "GET",
    "/api/review/progress": "GET",
    "/api/review/show/<int:instance_id>": "GET",
    "/api/review/status": "GET",
    "/api/review/undo": "POST",
    "/api/review/verdict": "POST",
    "/api/server-config": "GET, POST",
    "/api/set-data": "POST",
    "/api/templates/bbox-json": "GET",
    "/api/viewer/add-image-layer": "POST",
    "/api/viewer/add-segmentation-layer": "POST",
    "/api/viewer/cc3d-relabel-annotation": "POST",
    "/api/viewer/create-instance-correction": "POST",
    "/api/viewer/remove-layer": "POST",
    "/api/viewer/rename-layer": "POST",
    "/api/viewer/sync-instance-correction": "POST",
    "/pipeline-builder": "GET",
    "/static/<path:filename>": "GET",
    "/update/equivalences": "POST",
}


def test_the_dashboard_answers_the_same_urls_and_methods():
    from cellmap_flow.dashboard.app import app

    answered = {}
    for rule in app.url_map.iter_rules():
        answered.setdefault(rule.rule, set()).update(rule.methods - {"HEAD", "OPTIONS"})
    assert {rule: ", ".join(sorted(methods)) for rule, methods in answered.items()} == URLS
