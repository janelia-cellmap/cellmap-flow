# The dashboard's routes

Generated from `cellmap_flow/schema.py` by `npm run gen` in web/ (or `python -m cellmap_flow.schema --docs docs/api.md`); do not edit. The method and path are the Flask app's; *answered by* says which mode answers the route: **both** (47) the server dashboard and the browser app, **stub** (7) both, the browser with a fixed or empty answer in the server's shape, **server** (36) the server dashboard only (the browser answers `{success: false, available: false, error}`), **browser** (9) the browser app only. *Body* names the pydantic model of `cellmap_flow/dashboard/requests.py` (or the route's file) the body is parsed with; the generated `web/src/generated/cellmap-flow.ts` has its type.

| Method | Path | Answered by | Body |
|---|---|---|---|
| GET | `/` | server |  |
| GET | `/pipeline-builder` | server |  |
| POST | `/api/set-data` | both | `SetData` |
| GET | `/api/huggingface-models` | both |  |
| POST | `/api/huggingface-models/refresh` | both |  |
| GET | `/api/bioimage-models` | stub |  |
| POST | `/api/bioimage-models/refresh` | stub |  |
| POST | `/api/models` | both | `SubmitModels` |
| GET | `/api/models/loaded` | both |  |
| POST | `/api/models/remove` | both | `RemoveModel` |
| POST | `/api/models/resolve` | both | `ResolveModel` |
| POST | `/api/models/add` | both | `AddModel` |
| GET | `/api/model_advice` | both |  |
| PUT | `/api/pipeline` | both | `PipelineUpdate` |
| POST | `/api/pipeline/apply` | server |  |
| POST | `/api/process` | server |  |
| POST | `/update/equivalences` | server | `Equivalences` |
| GET | `/api/server-config` | both |  |
| POST | `/api/server-config` | both | `ServerConfigUpdate` |
| GET | `/api/gpu-queues` | stub |  |
| GET | `/api/job-logs` | stub |  |
| GET | `/api/export-config` | both |  |
| GET | `/api/logs/stream` | server |  |
| GET | `/api/blockwise-config` | server |  |
| POST | `/api/blockwise-config` | server | `BlockwiseSettings` |
| POST | `/api/blockwise/validate` | server | `BlockwiseValidate` |
| POST | `/api/blockwise/generate` | server | `BlockwiseGenerate` |
| POST | `/api/blockwise/precheck` | server | `BlockwisePrecheck` |
| POST | `/api/blockwise/submit` | server | `BlockwiseSubmit` |
| POST | `/api/bbx-generator` | server | `BbxGenerator` |
| POST | `/api/bbx-generator/finalize` | server |  |
| GET | `/api/bbx-generator/status` | server |  |
| GET | `/api/model-config-types` | server |  |
| POST | `/api/create-model-config` | server | `CreateModelConfig` |
| GET | `/api/templates/bbox-json` | server |  |
| POST | `/api/review/open` | server |  |
| GET | `/api/review/next` | server |  |
| POST | `/api/review/verdict` | server |  |
| POST | `/api/review/undo` | server |  |
| GET | `/api/review/progress` | server |  |
| GET | `/api/review/show/<int:instance_id>` | server |  |
| GET | `/api/review/pick_stream` | server |  |
| GET | `/api/review/status` | server |  |
| POST | `/api/viewer/add-image-layer` | server |  |
| POST | `/api/viewer/add-segmentation-layer` | server |  |
| POST | `/api/viewer/remove-layer` | server |  |
| POST | `/api/viewer/rename-layer` | server |  |
| POST | `/api/viewer/create-instance-correction` | server |  |
| POST | `/api/viewer/sync-instance-correction` | server |  |
| POST | `/api/viewer/cc3d-relabel-annotation` | server |  |
| GET | `/api/finetune/user-prefs` | both |  |
| POST | `/api/finetune/user-prefs` | both |  |
| GET | `/api/finetune/models` | both |  |
| POST | `/api/finetune/create-volume` | both | `CreateVolume` |
| POST | `/api/finetune/add-to-viewer` | both |  |
| POST | `/api/finetune/list-existing-sessions` | both |  |
| POST | `/api/finetune/load-existing-volume` | both |  |
| GET | `/api/finetune/load-existing-volume-progress` | stub |  |
| POST | `/api/finetune/sync-annotations` | stub |  |
| POST | `/api/finetune/refresh-annotated-regions` | stub |  |
| GET | `/api/finetune/session-volume` | browser |  |
| GET | `/api/finetune/volume/<volume_id>/download` | browser |  |
| POST | `/api/finetune/delete-volume` | browser |  |
| POST | `/api/finetune/load-crops` | server | `LoadCrops` |
| GET | `/api/finetune/load-crops-progress` | server |  |
| GET | `/api/finetune/read-yaml` | server |  |
| POST | `/api/finetune/submit` | both | `FinetuneSubmit` |
| GET | `/api/finetune/jobs` | both |  |
| GET | `/api/finetune/job/<job_id>/status` | both |  |
| GET | `/api/finetune/job/<job_id>/logs` | both |  |
| GET | `/api/finetune/job/<job_id>/logs/stream` | both |  |
| POST | `/api/finetune/job/<job_id>/cancel` | both |  |
| POST | `/api/finetune/job/<job_id>/stop-early` | both |  |
| POST | `/api/finetune/job/<job_id>/restart` | both | `FinetuneRestart` |
| GET | `/api/finetune/model/<name>/download` | browser |  |
| GET | `/api/huggingface/credentials` | browser |  |
| POST | `/api/huggingface/credentials` | browser | `HuggingFaceCredentials` |
| POST | `/api/finetune/model/<name>/publish` | browser | `FinetunePublish` |
| GET | `/api/finetune/model/<name>/publish` | browser |  |
| GET | `/api/finetune/good-regions` | both |  |
| POST | `/api/finetune/good-regions/mark-view` | both |  |
| POST | `/api/finetune/good-regions/delete` | both |  |
| GET | `/api/finetune/rank/candidates` | both |  |
| POST | `/api/finetune/rank/score` | both |  |
| POST | `/api/finetune/rank/use` | both |  |
| POST | `/api/finetune/rank/window` | both |  |
| GET | `/api/finetune/view-labels/sources` | both |  |
| POST | `/api/finetune/view-labels/seed` | both |  |
| POST | `/api/finetune/view-labels/background` | both |  |
| POST | `/api/finetune/view-labels/split` | both |  |
| POST | `/api/finetune/view-labels/undo` | both |  |
| GET | `/api/finetune/ai-annotate/config` | both |  |
| POST | `/api/finetune/ai-annotate/credentials` | browser |  |
| POST | `/api/finetune/ai-annotate/settings` | both | `AIAnnotateSettings` |
| POST | `/api/finetune/ai-annotate/run` | both | `AIAnnotateRun` |
| GET | `/api/finetune/ai-annotate/status` | both |  |
| POST | `/api/finetune/ai-annotate/resend` | both | `AIAnnotateResend` |
| POST | `/api/finetune/ai-annotate/accept` | both | `AIAnnotateDecision` |
| POST | `/api/finetune/ai-annotate/reject` | both | `AIAnnotateDecision` |
