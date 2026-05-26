# ECTIL inference image.
#
# Build:
#   docker build -t ectil-inference .
#
# Run (mount the WSI, the weights, and an output directory):
#   docker run --rm \
#     -v /path/to/slides:/input:ro \
#     -v /path/to/weights:/weights:ro \
#     -v /path/to/output:/output \
#     ectil-inference \
#       --wsi /input/slide.svs \
#       --classifier-weights /weights/ectil_fold_0_weights_only.ckpt \
#       --retccl-weights /weights/retccl_best_ckpt.pth \
#       --output /output
#
# Add `--gpus all` to `docker run` and `--device cuda` to the command for GPU.
#
# Weights are NOT bundled in the image; mount them at runtime.
#   - ECTIL classifier: https://files.aiforoncology.nl/ectil  (see model_zoo/ectil/tcga/readme.md)
#   - RetCCL encoder:   see model_zoo/retccl/readme.md
# If --retccl-weights is omitted it defaults to /app/model_zoo/retccl/retccl_best_ckpt.pth
# or the RETCCL_WEIGHTS environment variable.

FROM continuumio/miniconda3:latest

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python 3.10.9 plus the WSI system libraries via conda-forge (mirrors README install).
RUN conda create -y -n ectil python=3.10.9 \
    && conda install -y -n ectil -c conda-forge openslide pixman libvips \
    && conda clean -afy

ENV PATH=/opt/conda/envs/ectil/bin:$PATH
ENV CONDA_DEFAULT_ENV=ectil

WORKDIR /app

# Install Python dependencies first for better layer caching.
COPY requirements.txt setup.py ./
RUN python -m pip install --no-cache-dir pip==23.3.2 \
    && python -m pip install --no-cache-dir -r requirements.txt

# Install the ectil package itself.
COPY . .
RUN python -m pip install --no-cache-dir --no-deps -e .

ENTRYPOINT ["python", "-m", "ectil.inference"]
CMD ["--help"]
