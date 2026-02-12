#!/bin/bash
# Simple bash script to host data via MinIO
# Usage: ./host_minio.sh [path_to_serve] [port]

set -e

# Default values
DATA_PATH="${1:-$HOME/minio-data}"
PORT="${2:-9000}"
BUCKET_NAME="neuroglancer"
MINIO_ROOT_USER="minio"
MINIO_ROOT_PASSWORD="minio123"

# Get local IP address
get_local_ip() {
    # Try to get the IP from hostname first
    LOCAL_IP=$(hostname -I | awk '{print $1}')

    # If that fails, use a different method
    if [ -z "$LOCAL_IP" ]; then
        LOCAL_IP=$(ip route get 8.8.8.8 | awk -F"src " 'NR==1{split($2,a," ");print a[1]}')
    fi

    # Fallback to localhost
    if [ -z "$LOCAL_IP" ]; then
        LOCAL_IP="127.0.0.1"
    fi

    echo "$LOCAL_IP"
}

# Find available port
find_available_port() {
    local start_port=$1
    local port=$start_port

    while ! nc -z localhost $port 2>/dev/null; do
        if [ $((port - start_port)) -gt 100 ]; then
            echo "Could not find available port" >&2
            exit 1
        fi
        # Check if port is available
        if ! lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
            echo $port
            return 0
        fi
        port=$((port + 1))
    done
}

# Setup
LOCAL_IP=$(get_local_ip)
if [ "$PORT" = "auto" ]; then
    PORT=$(find_available_port 9000)
fi

echo "========================================================"
echo "MinIO Server Setup"
echo "========================================================"
echo "Data path: $DATA_PATH"
echo "Address: $LOCAL_IP:$PORT"
echo "Username: $MINIO_ROOT_USER"
echo "Password: $MINIO_ROOT_PASSWORD"
echo "========================================================"

# Create data directory
mkdir -p "$DATA_PATH"

# Export credentials and enable CORS
export MINIO_ROOT_USER=$MINIO_ROOT_USER
export MINIO_ROOT_PASSWORD=$MINIO_ROOT_PASSWORD
export MINIO_API_CORS_ALLOW_ORIGIN="*"

# Cleanup function
cleanup() {
    echo ""
    echo "Stopping MinIO server..."
    kill $MINIO_PID 2>/dev/null || true
    wait $MINIO_PID 2>/dev/null || true
    echo "✓ Server stopped"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start MinIO server in background
# Set console to different port to avoid redirect issues
echo "Starting MinIO server..."
CONSOLE_PORT=$((PORT + 1))
minio server "$DATA_PATH" --address "$LOCAL_IP:$PORT" --console-address "$LOCAL_IP:$CONSOLE_PORT" &
MINIO_PID=$!

# Wait for server to start
sleep 3

# Check if server is running
if ! kill -0 $MINIO_PID 2>/dev/null; then
    echo "Error: MinIO server failed to start"
    exit 1
fi
echo "✓ MinIO server started (PID: $MINIO_PID)"

# Configure MinIO client
echo ""
echo "Configuring MinIO client..."
mc alias set local "http://$LOCAL_IP:$PORT" "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD"
echo "✓ Alias 'local' configured"

# Create bucket
echo "Creating bucket: $BUCKET_NAME"
mc mb "local/$BUCKET_NAME" 2>/dev/null || echo "✓ Bucket '$BUCKET_NAME' already exists"

# Make bucket public
echo "Making bucket public..."
mc anonymous set public "local/$BUCKET_NAME"
echo "✓ Bucket '$BUCKET_NAME' is now public"

# Print connection info
echo ""
echo "========================================================"
echo "MinIO Server Running"
echo "========================================================"
echo "Web Console:  http://$LOCAL_IP:$PORT"
echo "API Endpoint: http://$LOCAL_IP:$PORT"
echo "Bucket URL:   http://$LOCAL_IP:$PORT/$BUCKET_NAME"
echo "Username:     $MINIO_ROOT_USER"
echo "Password:     $MINIO_ROOT_PASSWORD"
echo "========================================================"
echo ""
echo "Press Ctrl+C to stop the server..."

# Wait for server process
wait $MINIO_PID
