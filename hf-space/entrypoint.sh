#!/usr/bin/env bash
# Start cellmap-flow's virtual-zarr server on the port HF Spaces expects.
# Reads model + dataset config from env vars.
#
# Two modes, selected by CELLMAP_MODEL_TYPE:
#   huggingface  (default) — load a cellmap-flow HF model
#   bioimage              — load a BioImage Model Zoo model
#
# Required env vars:
#   CELLMAP_DATASET           input zarr URL (s3:// or https://)
#   one of:
#     CELLMAP_MODEL_TYPE=huggingface   (default)
#       CELLMAP_HF_REPO       e.g. cellmap/jrc_mus-livers_16nm_to_8nm_mito
#       CELLMAP_HF_NAME       optional display name
#     CELLMAP_MODEL_TYPE=bioimage
#       CELLMAP_BMZ_MODEL     BMZ id, e.g. hiding-blowfish
#       CELLMAP_VOXEL_SIZE    comma-separated nm (e.g. 8,8,8)
#       CELLMAP_BMZ_NAME      optional display name (defaults to BMZ id)

set -euo pipefail

MODE="${CELLMAP_MODEL_TYPE:-huggingface}"
DATA="${CELLMAP_DATASET:-}"
PORT="${PORT:-7860}"

if [[ -z "$DATA" ]]; then
  echo "ERROR: CELLMAP_DATASET is not set." >&2
  echo "Set it to the input zarr URL, e.g.:" >&2
  echo "  CELLMAP_DATASET=s3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.zarr/recon-1/em/fibsem-uint8/" >&2
  exit 1
fi

case "$MODE" in
  huggingface)
    REPO="${CELLMAP_HF_REPO:-}"
    NAME="${CELLMAP_HF_NAME:-}"
    if [[ -z "$REPO" ]]; then
      echo "ERROR: CELLMAP_MODEL_TYPE=huggingface requires CELLMAP_HF_REPO." >&2
      echo "  e.g. CELLMAP_HF_REPO=cellmap/jrc_mus-livers_16nm_to_8nm_mito" >&2
      exit 1
    fi
    NAME="${NAME:-${REPO##*/}}"

    echo "[cellmap-hf-space] starting cellmap_flow_server (huggingface)"
    echo "  repo:    $REPO"
    echo "  name:    $NAME"
    echo "  data:    $DATA"
    echo "  port:    $PORT"

    exec cellmap_flow_server huggingface \
      --repo "$REPO" \
      --name "$NAME" \
      -d "$DATA" \
      --port "$PORT"
    ;;

  bioimage)
    BMZ="${CELLMAP_BMZ_MODEL:-}"
    VOXEL="${CELLMAP_VOXEL_SIZE:-}"
    BMZ_NAME="${CELLMAP_BMZ_NAME:-$BMZ}"
    if [[ -z "$BMZ" ]]; then
      echo "ERROR: CELLMAP_MODEL_TYPE=bioimage requires CELLMAP_BMZ_MODEL." >&2
      echo "  e.g. CELLMAP_BMZ_MODEL=hiding-blowfish" >&2
      exit 1
    fi
    if [[ -z "$VOXEL" ]]; then
      echo "ERROR: CELLMAP_MODEL_TYPE=bioimage requires CELLMAP_VOXEL_SIZE." >&2
      echo "  e.g. CELLMAP_VOXEL_SIZE=8,8,8   (nm per voxel, comma-separated)" >&2
      exit 1
    fi

    echo "[cellmap-hf-space] starting cellmap_flow_server (bioimage)"
    echo "  bmz:     $BMZ"
    echo "  name:    $BMZ_NAME"
    echo "  voxel:   $VOXEL"
    echo "  data:    $DATA"
    echo "  port:    $PORT"

    exec cellmap_flow_server bioimage \
      --model-name "$BMZ" \
      --voxel-size "$VOXEL" \
      --name "$BMZ_NAME" \
      -d "$DATA" \
      --port "$PORT"
    ;;

  *)
    echo "ERROR: unknown CELLMAP_MODEL_TYPE='$MODE'." >&2
    echo "  Supported: 'huggingface' (default), 'bioimage'." >&2
    exit 1
    ;;
esac
