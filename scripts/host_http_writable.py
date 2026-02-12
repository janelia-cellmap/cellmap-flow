#!/usr/bin/env python
"""
HTTP server with CORS support that allows both reading and writing.

Usage:
    python host_http_writable.py /path/to/serve [--port PORT]
"""

import argparse
import http.server
import socketserver
import socket
import os
from pathlib import Path
from urllib.parse import unquote


class WritableHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP request handler with CORS headers and write support."""

    def end_headers(self):
        # Add CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, HEAD, PUT, POST, DELETE, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.send_header('Access-Control-Expose-Headers', '*')
        super().end_headers()

    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.end_headers()

    def do_PUT(self):
        """Handle PUT requests to upload/update files."""
        path = self.translate_path(self.path)

        # Get content length
        length = int(self.headers.get('Content-Length', 0))

        # Create parent directories if needed
        os.makedirs(os.path.dirname(path), exist_ok=True)

        try:
            # Write the file
            with open(path, 'wb') as f:
                f.write(self.rfile.read(length))

            self.send_response(201)  # Created
            self.end_headers()
            print(f"✓ Wrote {length} bytes to {path}")
        except Exception as e:
            self.send_error(500, f"Error writing file: {e}")
            print(f"✗ Error writing to {path}: {e}")

    def do_POST(self):
        """Handle POST requests (same as PUT for simplicity)."""
        self.do_PUT()

    def do_DELETE(self):
        """Handle DELETE requests to remove files."""
        path = self.translate_path(self.path)

        try:
            if os.path.isfile(path):
                os.remove(path)
                self.send_response(204)  # No Content
                self.end_headers()
                print(f"✓ Deleted {path}")
            else:
                self.send_error(404, "File not found")
        except Exception as e:
            self.send_error(500, f"Error deleting file: {e}")
            print(f"✗ Error deleting {path}: {e}")


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


def find_available_port(start_port=8000, max_attempts=100):
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
        description="Serve a directory via HTTP with CORS and write support"
    )
    parser.add_argument(
        "path",
        help="Path to the directory to serve"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to use (default: auto-detect starting from 8000)"
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Host/IP to bind to (default: auto-detect local IP)"
    )

    args = parser.parse_args()

    # Validate path
    serve_path = Path(args.path).expanduser().resolve()
    if not serve_path.exists():
        print(f"Error: Path {serve_path} does not exist")
        return 1
    if not serve_path.is_dir():
        print(f"Error: Path {serve_path} is not a directory")
        return 1

    # Get network configuration
    host = args.host if args.host else get_local_ip()
    port = args.port if args.port else find_available_port()

    print("=" * 60)
    print("HTTP Server with CORS and Write Support")
    print("=" * 60)
    print(f"Serving directory: {serve_path}")
    print(f"Server address: http://{host}:{port}")
    print("=" * 60)

    # Change to the directory to serve
    os.chdir(serve_path)

    # Create server
    with socketserver.TCPServer((host, port), WritableHTTPRequestHandler) as httpd:
        print(f"\n✓ Server running (read + write enabled)")
        print(f"\nFor Neuroglancer, use URLs like:")
        print(f"  http://{host}:{port}/my_annotation.zarr")
        print(f"\nWrite support:")
        print(f"  PUT/POST to upload files")
        print(f"  DELETE to remove files")
        print(f"\nPress Ctrl+C to stop the server...\n")

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\nStopping server...")
            httpd.shutdown()
            print("✓ Server stopped")

    return 0


if __name__ == "__main__":
    exit(main())
