# cellmap-flow inference backend, deployable as a Hugging Face Space
# (Docker SDK).
#
# Hosts one cellmap-flow virtual-zarr server, parameterized by env vars:
#   CELLMAP_HF_REPO     — HF repo to load (e.g. cellmap/jrc_mus-livers_16nm_to_8nm_mito)
#   CELLMAP_HF_NAME     — display name (defaults to last segment of repo)
#   CELLMAP_DATASET     — input zarr URL (e.g. s3://janelia-cosem-datasets/.../*.zarr/recon-1/em/fibsem-uint8)
#
# To use a different model + dataset, fork the Space and change the secrets
# in HF Settings -> Variables and secrets.

FROM python:3.11-slim

# Build deps for the scientific stack (most ship manylinux wheels but a few
# do not; keep gcc/g++ around to be safe).
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ libgomp1 git \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces expect uid 1000 with a writable $HOME.
RUN useradd -m -u 1000 cellmap
USER cellmap
ENV HOME=/home/cellmap \
    PATH=/home/cellmap/.local/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install torch CPU first (much smaller than the default GPU build) and
# pin a recent stable. Cellmap-flow uses torch for HF model loading.
RUN pip install --no-cache-dir --user --index-url https://download.pytorch.org/whl/cpu torch

# cellmap-flow itself plus huggingface_hub. Install from PyPI; the dashboard
# templates aren't needed because we serve the virtual-zarr only.
RUN pip install --no-cache-dir --user \
    cellmap-flow \
    huggingface_hub \
    s3fs

# A tiny entrypoint that reads the env vars and starts cellmap_flow_server.
COPY --chown=cellmap:cellmap entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# HF Spaces routes public traffic to whatever port is in README's `app_port`.
EXPOSE 7860

CMD ["/app/entrypoint.sh"]
