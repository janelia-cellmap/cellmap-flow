"""Temporary: finetune_job_manager.py's import of ``encode_to_str``.

web_utils is ``cellmap_flow.serving.protocol`` now. The job manager is being
split in parallel (W4-D2), so its imports are retargeted when the two land
together, from the import table in cleanup_review/reports/phase3_w4_utils.md;
this module is deleted then. Nothing else may import it.
"""

from cellmap_flow.serving.protocol import encode_to_str  # noqa: F401
