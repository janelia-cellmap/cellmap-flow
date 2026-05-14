# cellmap-flow inference backend, deployable as a Hugging Face Space
# (Docker SDK).
#
# Hosts one cellmap-flow virtual-zarr server. Mode is selected by
# CELLMAP_MODEL_TYPE; see entrypoint.sh for the required env vars.
#
# Quick examples (set in HF Settings → Variables and secrets):
#
#   # cellmap HF model (default)
#   CELLMAP_HF_REPO=cellmap/jrc_mus-livers_16nm_to_8nm_mito
#   CELLMAP_DATASET=s3://janelia-cosem-datasets/jrc_mus-liver/jrc_mus-liver.zarr/recon-1/em/fibsem-uint8/
#
#   # BioImage Model Zoo model
#   CELLMAP_MODEL_TYPE=bioimage
#   CELLMAP_BMZ_MODEL=hiding-blowfish
#   CELLMAP_VOXEL_SIZE=8,8,8
#   CELLMAP_DATASET=s3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.zarr/recon-1/em/fibsem-uint8/s1

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

# cellmap-flow with the BMZ extra (pulls bioimageio.core + xarray) so the
# `cellmap_flow_server bioimage` subcommand works in addition to the
# `huggingface` subcommand. The dashboard templates aren't strictly needed
# (we serve the virtual-zarr only) but they come with the install.
RUN pip install --no-cache-dir --user \
    "cellmap-flow[bioimageio]" \
    huggingface_hub \
    s3fs

# A tiny entrypoint that reads the env vars and starts cellmap_flow_server.
COPY --chown=cellmap:cellmap entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# HF Spaces routes public traffic to whatever port is in README's `app_port`.
EXPOSE 7860

CMD ["/app/entrypoint.sh"]
