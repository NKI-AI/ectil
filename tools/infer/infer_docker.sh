#!/bin/bash
# Example: run ECTIL inference on a single WSI inside the Docker container.
#
# 1. Build the image (from the repo root):
#       docker build -t ectil-inference .
#
# 2. Download the weights (not bundled in the image):
#       - ECTIL classifier: https://files.aiforoncology.nl/ectil  (model_zoo/ectil/tcga/readme.md)
#       - RetCCL encoder:   see model_zoo/retccl/readme.md
#
# 3. Set the paths below and run this script.
#
# The container writes a per-slide directory under $OUTPUT containing:
#   tils_score.json, tile_predictions.csv, features.h5,
#   thumbnail.png, mask.png, mask_overlay.png,
#   attention_heatmap.png, til_heatmap.png

set -euo pipefail

WSI="/path/to/slide.svs"
CLASSIFIER_WEIGHTS="/path/to/ectil_fold_0_weights_only.ckpt"
RETCCL_WEIGHTS="/path/to/retccl_best_ckpt.pth"
OUTPUT="/path/to/output"

docker run --rm \
    -v "$(dirname "$WSI")":/input:ro \
    -v "$(dirname "$CLASSIFIER_WEIGHTS")":/weights/ectil:ro \
    -v "$(dirname "$RETCCL_WEIGHTS")":/weights/retccl:ro \
    -v "$OUTPUT":/output \
    ectil-inference \
        --wsi "/input/$(basename "$WSI")" \
        --classifier-weights "/weights/ectil/$(basename "$CLASSIFIER_WEIGHTS")" \
        --retccl-weights "/weights/retccl/$(basename "$RETCCL_WEIGHTS")" \
        --output /output

# For GPU, add `--gpus all` to `docker run` and `--device cuda` to the command above.
