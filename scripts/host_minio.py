#!/usr/bin/env python
"""
Script to host data via MinIO server with automatic network configuration.

Usage:
    python host_minio.py [path_to_serve]
    python host_minio.py /groups/cellmap/cellmap/ackermand/Programming/neuroglancer/tmp

If no path is provided, uses ~/minio-data as default.
"""

import argparse
import socket
import subprocess
import sys
import time
import os
from pathlib import Path


def get_local_ip():
    """Get the local IP address that's accessible on the network."""
    try:
        # Create a socket to an external address to find local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Doesn't need to be reachable, just used to determine local IP
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        # Fallback to localhost if unable to determine
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


def start_minio_server(data_path, ip, port, username="minio", password="minio123"):
    """Start MinIO server in background."""
    data_path = Path(data_path).expanduser().resolve()
    data_path.mkdir(parents=True, exist_ok=True)

    print(f"Starting MinIO server...")
    print(f"  Data path: {data_path}")
    print(f"  Address: {ip}:{port}")
    print(f"  Username: {username}")
    print(f"  Password: {password}")
    print(f"  CORS: Enabled (allow all origins)")

    env = os.environ.copy()
    env["MINIO_ROOT_USER"] = username
    env["MINIO_ROOT_PASSWORD"] = password
    # Enable CORS to allow requests from Neuroglancer and other web apps
    env["MINIO_API_CORS_ALLOW_ORIGIN"] = "*"

    # Start minio server with console on different port to avoid redirect issues
    cmd = ["minio", "server", str(data_path),
           "--address", f"{ip}:{port}",
           "--console-address", f"{ip}:{port + 1}"]
    minio_process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Wait a bit for server to start
    time.sleep(2)

    # Check if process is still running
    if minio_process.poll() is not None:
        stdout, stderr = minio_process.communicate()
        print("MinIO server failed to start!")
        print("STDOUT:", stdout)
        print("STDERR:", stderr)
        sys.exit(1)

    print(f"✓ MinIO server started (PID: {minio_process.pid})")
    return minio_process


def configure_minio_client(ip, port, username="minio", password="minio123", bucket_name="neuroglancer"):
    """Configure MinIO client and create public bucket."""
    endpoint = f"http://{ip}:{port}"
    alias = "local"

    print(f"\nConfiguring MinIO client...")
    print(f"  Endpoint: {endpoint}")
    print(f"  Alias: {alias}")

    # Set up alias
    cmd = ["mc", "alias", "set", alias, endpoint, username, password]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Failed to set alias: {result.stderr}")
        return False
    print(f"✓ Alias '{alias}' configured")

    # Create bucket
    print(f"  Creating bucket: {bucket_name}")
    cmd = ["mc", "mb", f"{alias}/{bucket_name}"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Check if bucket already exists (various error messages)
        error_lower = result.stderr.lower()
        if "already exists" in error_lower or "already own it" in error_lower or "bucket exists" in error_lower:
            print(f"✓ Bucket '{bucket_name}' already exists")
        else:
            print(f"Failed to create bucket: {result.stderr}")
            return False
    else:
        print(f"✓ Bucket '{bucket_name}' created")

    # Make bucket public
    print(f"  Making bucket public...")
    cmd = ["mc", "anonymous", "set", "public", f"{alias}/{bucket_name}"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Failed to make bucket public: {result.stderr}")
        return False
    print(f"✓ Bucket '{bucket_name}' is now public")
    print(f"✓ CORS is enabled (configured at server startup)")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Host data via MinIO with automatic network configuration"
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="~/minio-data",
        help="Path to serve (default: ~/minio-data)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to use (default: auto-detect starting from 9000)"
    )
    parser.add_argument(
        "--bucket",
        default="neuroglancer",
        help="Bucket name (default: neuroglancer, min 3 characters)"
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

    # Validate bucket name
    if len(args.bucket) < 3:
        print(f"Error: Bucket name '{args.bucket}' is too short. Minimum length is 3 characters.")
        sys.exit(1)

    # Get network configuration
    local_ip = get_local_ip()
    port = args.port if args.port else find_available_port()

    print("="*60)
    print("MinIO Server Setup")
    print("="*60)

    # Start MinIO server
    minio_process = start_minio_server(
        args.path, local_ip, port, args.username, args.password
    )

    # Configure client
    if not configure_minio_client(local_ip, port, args.username, args.password, args.bucket):
        print("\nFailed to configure MinIO client. Stopping server...")
        minio_process.terminate()
        sys.exit(1)

    # Print connection info
    print("\n" + "="*60)
    print("MinIO Server Running")
    print("="*60)
    print(f"Web Console:  http://{local_ip}:{port}")
    print(f"API Endpoint: http://{local_ip}:{port}")
    print(f"Bucket URL:   http://{local_ip}:{port}/{args.bucket}")
    print(f"Username:     {args.username}")
    print(f"Password:     {args.password}")
    print("="*60)
    print("\nPress Ctrl+C to stop the server...")

    # Keep running
    try:
        minio_process.wait()
    except KeyboardInterrupt:
        print("\n\nStopping MinIO server...")
        minio_process.terminate()
        minio_process.wait()
        print("✓ Server stopped")


if __name__ == "__main__":
    main()
