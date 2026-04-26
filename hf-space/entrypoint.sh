#!/usr/bin/env bash
# Start cellmap-flow's virtual-zarr server on the port HF Spaces expects.
# Reads model + dataset config from env vars.

set -euo pipefail

REPO="${CELLMAP_HF_REPO:-}"
NAME="${CELLMAP_HF_NAME:-}"
DATA="${CELLMAP_DATASET:-}"
PORT="${PORT:-7860}"

if [[ -z "$REPO" ]]; then
  echo "ERROR: CELLMAP_HF_REPO is not set." >&2
  echo "Set it in HF Space Settings → Variables and secrets, e.g.:" >&2
  echo "  CELLMAP_HF_REPO=cellmap/jrc_mus-livers_16nm_to_8nm_mito" >&2
  exit 1
fi
if [[ -z "$DATA" ]]; then
  echo "ERROR: CELLMAP_DATASET is not set." >&2
  echo "Set it to the input zarr URL, e.g.:" >&2
  echo "  CELLMAP_DATASET=s3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.zarr/recon-1/em/fibsem-uint8/" >&2
  exit 1
fi
NAME="${NAME:-${REPO##*/}}"

echo "[cellmap-hf-space] starting cellmap_flow_server"
echo "  repo:    $REPO"
echo "  name:    $NAME"
echo "  data:    $DATA"
echo "  port:    $PORT"

exec cellmap_flow_server huggingface \
  --repo "$REPO" \
  --name "$NAME" \
  -d "$DATA" \
  --port "$PORT"
