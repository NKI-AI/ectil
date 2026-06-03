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

# Pinned (not :latest) so the build is reproducible. The :latest tag moved to
# conda 26.x in April 2026, where the conda-forge solve for the WSI libs below
# can drag GraalPy into the env and break the pip install of torch==2.4.1
# ("Could not find a version that satisfies the requirement torch==2.4.1").
FROM continuumio/miniconda3:24.11.1-0

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python 3.10.9 plus the WSI system libraries via conda-forge (mirrors README install).
# Force the CPython build of python and keep it pinned across the second install
# so newer conda solvers can't swap it for graalpy when resolving conda-forge deps
# (that swap silently breaks the torch==2.4.1 pip install in the next layer).
RUN conda create -y -n ectil -c conda-forge "python=3.10.9=*_cpython" \
    && echo "python 3.10.9" > /opt/conda/envs/ectil/conda-meta/pinned \
    && conda install -y -n ectil -c conda-forge openslide pixman libvips \
    && conda clean -afy

ENV PATH=/opt/conda/envs/ectil/bin:$PATH
ENV CONDA_DEFAULT_ENV=ectil

WORKDIR /app

# Install Python dependencies first for better layer caching.
# Pin the build toolchain as a matched set: an old pip (23.3.2) paired with a
# newer setuptools whose `_core_metadata` calls `canonicalize_version(..., strip_trailing_zero=)`
# needs a `packaging` >= 23.2 that actually has that kwarg, otherwise the editable
# install of this package below dies with
#   TypeError: canonicalize_version() got an unexpected keyword argument 'strip_trailing_zero'
# Pinning setuptools/wheel/packaging together keeps the toolchain self-consistent.
COPY requirements.txt setup.py ./
RUN python -m pip install --no-cache-dir \
        pip==23.3.2 setuptools==69.5.1 wheel==0.43.0 packaging==24.0 \
    && python -m pip install --no-cache-dir -r requirements.txt

# Install the ectil package itself.
COPY . .
RUN python -m pip install --no-cache-dir --no-deps -e .

ENTRYPOINT ["python", "-m", "ectil.inference"]
CMD ["--help"]
