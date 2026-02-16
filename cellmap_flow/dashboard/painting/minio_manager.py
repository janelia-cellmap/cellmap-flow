"""
MinIO server management for Neuroglancer annotation painting.

Handles starting/stopping the MinIO server, uploading zarr files,
and providing S3-compatible URLs for Neuroglancer to write to.
"""

import logging
import os
import socket
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# MinIO state for finetune annotation crops
minio_state = {
    "process": None,  # subprocess.Popen object
    "port": None,  # int
    "ip": None,  # str
    "bucket": "annotations",
    "minio_root": None,  # Path to MinIO storage directory
    "output_base": None,  # Base output directory for syncing back
    "last_sync": {},  # Track last sync time per crop_id
    "sync_thread": None,  # Background sync thread
}

# Track annotation volumes for sparse annotation workflow
# Maps volume_id -> volume metadata dict
annotation_volumes = {}


def get_local_ip():
    """Get the local IP address for MinIO server."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "127.0.0.1"


def find_available_port(start_port=9000):
    """Find an available port pair for MinIO server (API on port, console on port+1)."""
    for port in range(start_port, start_port + 100):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s1:
                s1.bind(("", port))
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s2:
                    s2.bind(("", port + 1))
                    return port
        except OSError:
            continue
    raise RuntimeError("Could not find available port for MinIO")


def ensure_minio_serving(zarr_path, crop_id, output_base_dir=None):
    """
    Ensure MinIO is running and upload zarr file.

    Args:
        zarr_path: Path to zarr file to upload
        crop_id: Unique identifier for the crop
        output_base_dir: Base output directory (MinIO will use output_base_dir/.minio)

    Returns:
        MinIO URL for the zarr file
    """
    from .annotation_sync import start_periodic_sync

    # Check if MinIO is already running
    if minio_state["process"] is None or minio_state["process"].poll() is not None:
        # Determine MinIO storage location
        if output_base_dir:
            minio_root = Path(output_base_dir) / ".minio"
            minio_state["output_base"] = output_base_dir
        else:
            minio_root = Path("~/.minio-server").expanduser()
            minio_state["output_base"] = None

        minio_root.mkdir(parents=True, exist_ok=True)
        minio_state["minio_root"] = str(minio_root)

        ip = get_local_ip()
        port = find_available_port()

        env = os.environ.copy()
        env["MINIO_ROOT_USER"] = "minio"
        env["MINIO_ROOT_PASSWORD"] = "minio123"
        env["MINIO_API_CORS_ALLOW_ORIGIN"] = "*"

        minio_cmd = [
            "minio",
            "server",
            str(minio_root),
            "--address",
            f"{ip}:{port}",
            "--console-address",
            f"{ip}:{port+1}",
        ]

        logger.info(f"Starting MinIO server at {ip}:{port}")
        minio_proc = subprocess.Popen(
            minio_cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        time.sleep(3)

        if minio_proc.poll() is not None:
            stderr = minio_proc.stderr.read().decode() if minio_proc.stderr else ""
            raise RuntimeError(f"MinIO failed to start: {stderr}")

        minio_state["process"] = minio_proc
        minio_state["port"] = port
        minio_state["ip"] = ip

        logger.info(f"MinIO started (PID: {minio_proc.pid})")

        # Configure mc client
        subprocess.run(
            [
                "mc",
                "alias",
                "set",
                "myserver",
                f"http://{ip}:{port}",
                "minio",
                "minio123",
            ],
            check=True,
            capture_output=True,
        )
        logger.info("MC client configured")

        # Create bucket if needed
        result = subprocess.run(
            ["mc", "mb", f"myserver/{minio_state['bucket']}"],
            capture_output=True,
            text=True,
        )

        # Ignore error if bucket already exists
        if result.returncode != 0 and "already" not in result.stderr.lower():
            logger.warning(f"Bucket creation returned: {result.stderr}")

        # Make bucket public
        subprocess.run(
            ["mc", "anonymous", "set", "public", f"myserver/{minio_state['bucket']}"],
            check=True,
            capture_output=True,
        )
        logger.info(f"Bucket {minio_state['bucket']} is public")

        # Start periodic sync thread
        start_periodic_sync()

    # Upload zarr file
    zarr_name = Path(zarr_path).name
    target = f"myserver/{minio_state['bucket']}/{zarr_name}"

    logger.info(f"Uploading {zarr_name} to MinIO")
    result = subprocess.run(
        ["mc", "mirror", "--overwrite", zarr_path, target],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Failed to upload to MinIO: {result.stderr}")

    logger.info(f"Uploaded {zarr_name} to MinIO")

    # Return MinIO URL
    minio_url = f"http://{minio_state['ip']}:{minio_state['port']}/{minio_state['bucket']}/{zarr_name}"
    return minio_url
