#!/usr/bin/env python
"""
Set up MinIO with a clean bucket structure for serving zarr files.

This script:
1. Starts MinIO with a dedicated data directory
2. Creates a bucket
3. Uploads existing zarr files to the bucket
"""

import argparse
import subprocess
import sys
import time
import socket
from pathlib import Path
import os


def get_local_ip():
    """Get the local IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "127.0.0.1"


def find_available_port(start_port=9000):
    """Find an available port."""
    for port in range(start_port, start_port + 100):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", port))
                return port
        except OSError:
            continue
    raise RuntimeError("Could not find available port")


def main():
    parser = argparse.ArgumentParser(description="Set up MinIO for serving zarr files")
    parser.add_argument("--data-dir", required=True, help="Directory containing zarr files to serve")
    parser.add_argument("--minio-root", default="~/.minio-server", help="MinIO server data directory")
    parser.add_argument("--bucket", default="annotations", help="Bucket name")
    parser.add_argument("--port", type=int, default=None, help="Port to use")

    args = parser.parse_args()

    # Resolve paths
    data_dir = Path(args.data_dir).expanduser().resolve()
    minio_root = Path(args.minio_root).expanduser().resolve()

    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} does not exist")
        return 1

    # Create MinIO root
    minio_root.mkdir(parents=True, exist_ok=True)

    # Get network config
    ip = get_local_ip()
    port = args.port if args.port else find_available_port()

    print("="*60)
    print("Setting up MinIO")
    print("="*60)
    print(f"MinIO data: {minio_root}")
    print(f"Source data: {data_dir}")
    print(f"Server: http://{ip}:{port}")
    print(f"Bucket: {args.bucket}")
    print("="*60)

    # Start MinIO server in background
    print("\nStarting MinIO server...")
    env = os.environ.copy()
    env["MINIO_ROOT_USER"] = "minio"
    env["MINIO_ROOT_PASSWORD"] = "minio123"
    env["MINIO_API_CORS_ALLOW_ORIGIN"] = "*"

    minio_cmd = [
        "minio", "server", str(minio_root),
        "--address", f"{ip}:{port}",
        "--console-address", f"{ip}:{port+1}"
    ]

    minio_proc = subprocess.Popen(minio_cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(3)

    if minio_proc.poll() is not None:
        print("✗ MinIO failed to start")
        return 1

    print(f"✓ MinIO started (PID: {minio_proc.pid})")

    # Configure mc client
    print("\nConfiguring mc client...")
    subprocess.run(
        ["mc", "alias", "set", "myserver", f"http://{ip}:{port}", "minio", "minio123"],
        check=True, capture_output=True
    )
    print("✓ Client configured")

    # Create bucket
    print(f"\nCreating bucket '{args.bucket}'...")
    result = subprocess.run(
        ["mc", "mb", f"myserver/{args.bucket}"],
        capture_output=True, text=True
    )
    if result.returncode != 0 and "already" not in result.stderr.lower():
        print(f"✗ Failed: {result.stderr}")
        minio_proc.terminate()
        return 1
    print(f"✓ Bucket ready")

    # Make bucket public
    print("\nMaking bucket public...")
    subprocess.run(
        ["mc", "anonymous", "set", "public", f"myserver/{args.bucket}"],
        check=True, capture_output=True
    )
    print("✓ Bucket is public")

    # Upload all zarr files
    print(f"\nUploading zarr files from {data_dir}...")
    zarr_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.suffix == '.zarr']

    if not zarr_dirs:
        print("⚠ No .zarr directories found")
    else:
        for zarr_dir in zarr_dirs:
            print(f"  Uploading {zarr_dir.name}...")
            result = subprocess.run(
                ["mc", "mirror", "--overwrite", str(zarr_dir), f"myserver/{args.bucket}/{zarr_dir.name}"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                print(f"  ✓ {zarr_dir.name}")
            else:
                print(f"  ✗ {zarr_dir.name}: {result.stderr}")

    # Print summary
    print("\n" + "="*60)
    print("MinIO Ready!")
    print("="*60)
    print(f"API Endpoint: http://{ip}:{port}")
    print(f"Console:      http://{ip}:{port+1}")
    print(f"Bucket:       {args.bucket}")
    print(f"\nAccess zarr files at:")
    print(f"  http://{ip}:{port}/{args.bucket}/<zarr-name>")
    print(f"\nFor Neuroglancer:")
    for zarr_dir in zarr_dirs:
        print(f"  http://{ip}:{port}/{args.bucket}/{zarr_dir.name}")
    print("="*60)
    print(f"\nMinIO PID: {minio_proc.pid}")
    print("To stop: kill {minio_proc.pid}")
    print("\nTo sync new files:")
    print(f"  mc mirror <local-zarr> myserver/{args.bucket}/<zarr-name>")

    return 0


if __name__ == "__main__":
    sys.exit(main())
