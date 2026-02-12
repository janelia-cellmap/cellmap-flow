#!/usr/bin/env python
"""
Simplified MinIO script that serves a directory directly as a bucket.

Usage:
    python host_minio_simple.py /path/to/serve bucket-name

The entire directory will be accessible as a single bucket.
"""

import argparse
import socket
import subprocess
import sys
import os
import time
from pathlib import Path


def get_local_ip():
    """Get the local IP address that's accessible on the network."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "127.0.0.1"


def find_available_port(start_port=9000, max_attempts=100):
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"Could not find available port in range {start_port}-{start_port + max_attempts}")


def main():
    parser = argparse.ArgumentParser(
        description="Serve a directory via MinIO (directory name becomes bucket name)"
    )
    parser.add_argument(
        "path",
        help="Path to the directory to serve (directory name will be used as bucket name)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to use (default: auto-detect starting from 9000)"
    )
    parser.add_argument(
        "--username",
        default="minio",
        help="MinIO username (default: minio)"
    )
    parser.add_argument(
        "--password",
        default="minio123",
        help="MinIO password (default: minio123)"
    )

    args = parser.parse_args()

    # Get directory info
    data_path = Path(args.path).expanduser().resolve()
    if not data_path.exists():
        print(f"Error: Path {data_path} does not exist")
        sys.exit(1)
    if not data_path.is_dir():
        print(f"Error: Path {data_path} is not a directory")
        sys.exit(1)

    # Use parent directory as MinIO root, directory name as bucket
    minio_root = data_path.parent
    bucket_name = data_path.name

    if len(bucket_name) < 3:
        print(f"Error: Directory name '{bucket_name}' is too short for a bucket name (min 3 characters)")
        print(f"Please rename the directory or use a different path")
        sys.exit(1)

    # Get network configuration
    local_ip = get_local_ip()
    port = args.port if args.port else find_available_port()

    print("=" * 60)
    print("MinIO Server Setup (Simple Mode)")
    print("=" * 60)
    print(f"Serving directory: {data_path}")
    print(f"MinIO root: {minio_root}")
    print(f"Bucket name: {bucket_name}")
    print(f"Address: {local_ip}:{port}")
    print(f"Username: {args.username}")
    print(f"Password: {args.password}")
    print(f"CORS: Enabled (allow all origins)")
    print("=" * 60)

    # Set up environment
    env = os.environ.copy()
    env["MINIO_ROOT_USER"] = args.username
    env["MINIO_ROOT_PASSWORD"] = args.password
    env["MINIO_API_CORS_ALLOW_ORIGIN"] = "*"

    # Start MinIO server
    print("\nStarting MinIO server...")
    # Disable console to avoid CORS redirect issues
    cmd = ["minio", "server", str(minio_root),
           "--address", f"{local_ip}:{port}",
           "--console-address", f"{local_ip}:{port + 1}"]
    minio_process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    time.sleep(2)

    if minio_process.poll() is not None:
        stdout, stderr = minio_process.communicate()
        print("MinIO server failed to start!")
        print("STDOUT:", stdout)
        print("STDERR:", stderr)
        sys.exit(1)

    print(f"✓ MinIO server started (PID: {minio_process.pid})")

    # Configure mc client
    endpoint = f"http://{local_ip}:{port}"
    alias = "local"

    print(f"\nConfiguring MinIO client...")
    result = subprocess.run(
        ["mc", "alias", "set", alias, endpoint, args.username, args.password],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"Failed to set alias: {result.stderr}")
        minio_process.terminate()
        sys.exit(1)
    print(f"✓ Alias configured")

    # Make bucket public
    print(f"Making bucket '{bucket_name}' public...")
    result = subprocess.run(
        ["mc", "anonymous", "set", "public", f"{alias}/{bucket_name}"],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"Note: Could not set public access: {result.stderr}")
        print(f"Bucket may already be configured")
    else:
        print(f"✓ Bucket is public")

    # Print access info
    print("\n" + "=" * 60)
    print("MinIO Server Running")
    print("=" * 60)
    print(f"API Endpoint: http://{local_ip}:{port}")
    print(f"Web Console:  http://{local_ip}:{port + 1}")
    print(f"Bucket URL:   http://{local_ip}:{port}/{bucket_name}")
    print(f"\nExample file access:")
    print(f"  http://{local_ip}:{port}/{bucket_name}/<filename>")
    print(f"\nFor Neuroglancer:")
    print(f"  zarr://http://{local_ip}:{port}/{bucket_name}/<zarr-path>")
    print("\nTest if bucket is accessible:")
    print(f"  curl http://{local_ip}:{port}/{bucket_name}/")
    print("=" * 60)
    print("\nPress Ctrl+C to stop the server...")

    # Wait for server
    try:
        minio_process.wait()
    except KeyboardInterrupt:
        print("\n\nStopping MinIO server...")
        minio_process.terminate()
        minio_process.wait()
        print("✓ Server stopped")


if __name__ == "__main__":
    main()
