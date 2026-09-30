"""Temporary: finetune_job_manager.py's import of ``resolve_model_geometry``.

model_geometry is ``cellmap_flow.models.geometry_cache`` now. The job manager
is being split in parallel (W4-D2), so its imports are retargeted when the
two land together, from the import table in
cleanup_review/reports/phase3_w4_utils.md; this module is deleted then.
Nothing else may import it.

The job manager imports the name when it submits a job, and the name is
looked up in geometry_cache then, so a test's patch of
``geometry_cache.resolve_model_geometry`` reaches it.
"""


def __getattr__(name):
    if name != "resolve_model_geometry":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from cellmap_flow.models import geometry_cache

    return geometry_cache.resolve_model_geometry
