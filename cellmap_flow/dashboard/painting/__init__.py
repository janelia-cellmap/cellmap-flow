"""
Neuroglancer painting/annotation module for interactive finetuning workflow.

This module contains the MinIO-backed annotation painting system that allows
users to interactively paint corrections in Neuroglancer. It was extracted
from the main dashboard app to keep the core training flow clean.

To reintegrate, import and register the Flask routes from routes.py,
and ensure MinIO dependencies (minio CLI, s3fs) are available.
"""
