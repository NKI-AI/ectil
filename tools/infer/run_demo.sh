#!/bin/bash
# One-command ECTIL WSI inference smoke test.
#
# After `git clone`, just run (from anywhere):
#
#     ./tools/infer/run_demo.sh
#
# It will, idempotently (skip-if-exists everywhere):
#   1. Pre-flight check the host tools (docker, curl, gdown)
#   2. Download the RetCCL encoder weights      -> model_zoo/retccl/retccl_best_ckpt.pth
#   3. Download the ECTIL classifier weights     -> model_zoo/ectil/tcga/fold_0/...ckpt
#   4. Download 5 small TCGA-BRCA slides          -> data/wsi/*.svs
#   5. Build the Docker image (`docker build -t ectil-inference .`)
#   6. Run the pipeline in the container, CPU, in TWO modes:
#         - single-slide:  --wsi <one .svs>   (expect 1 result row)
#         - directory:     --wsi data/wsi     (expect 5 result rows)
#   7. Validate the outputs of both runs and print SMOKE TEST PASSED / FAILED.
#
# Heavy C deps (openslide, libvips, dlup, torch) live inside the image, NOT on
# your host. The only host requirements are Docker (daemon running), curl, and
# python3+pip (used solely to bootstrap `gdown` for the Google-Drive download
# and to parse the result JSON). Everything downloaded lands under data/wsi and
# model_zoo, both of which are gitignored.

set -euo pipefail

# Resolve repo root from this script's location so it works from any cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

IMAGE="ectil-inference"

RETCCL_DIR="model_zoo/retccl"
RETCCL_WEIGHTS="$RETCCL_DIR/retccl_best_ckpt.pth"
RETCCL_GDRIVE_FOLDER="https://drive.google.com/drive/folders/1AhstAFVqtTqxeS9WlBpU41BV08LYFUnL"

ECTIL_DIR="model_zoo/ectil/tcga/fold_0"
ECTIL_WEIGHTS="$ECTIL_DIR/epoch_065_step_858_weights_only.ckpt"
ECTIL_URL="https://files.aiforoncology.nl/ectil/tcga/fold_0/epoch_065_step_858_weights_only.ckpt"

# The demo owns its own slide subdir so the directory-mode row count is exactly
# the 5 slides we download, regardless of anything else under data/wsi.
WSI_DIR="data/wsi/demo"
OUTPUT_DIR="data/inference_output"

# 5 verified small TCGA-BRCA diagnostic slides served by the GDC data endpoint.
GDC_FILE_IDS=(
    "f2d5aa37-d9ce-4264-a447-fc69dd0d7d85"
    "a5b0148b-afba-4cc6-9cb4-c346966d73e3"
    "ba1d2e38-fd12-478a-976a-e6701ed784e2"
    "2449ff02-6925-4f25-9074-7c5fbeab0bd2"
    "acc47852-3f83-4ed3-a85e-ecba28c06aa9"
)
NUM_SLIDES=${#GDC_FILE_IDS[@]}

echo "==> ECTIL WSI inference smoke test (repo: $REPO_ROOT)"

# ---------------------------------------------------------------------------
# 1. Pre-flight: host tools
# ---------------------------------------------------------------------------
echo "==> [1/7] Pre-flight checks ..."
if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker not found on PATH. Install Docker (https://docs.docker.com/get-docker/) and retry." >&2
    exit 1
fi
if ! docker info >/dev/null 2>&1; then
    echo "ERROR: the Docker daemon is not running. Start Docker Desktop / dockerd and retry." >&2
    exit 1
fi
if ! command -v curl >/dev/null 2>&1; then
    echo "ERROR: curl not found on PATH. Install curl and retry." >&2
    exit 1
fi
# gdown is only needed if the RetCCL weights are not already present.
if [ ! -f "$RETCCL_WEIGHTS" ] && ! command -v gdown >/dev/null 2>&1; then
    if command -v python3 >/dev/null 2>&1 && python3 -m pip --version >/dev/null 2>&1; then
        echo "    gdown not found; will install it via pip for the Google-Drive download."
    else
        echo "ERROR: RetCCL weights are missing and neither 'gdown' nor 'python3 -m pip' is available." >&2
        echo "       Install gdown (pip install gdown) or place the weights at $RETCCL_WEIGHTS." >&2
        exit 1
    fi
fi
echo "    docker, curl present; daemon running."

# ---------------------------------------------------------------------------
# 2. RetCCL encoder weights (Google Drive, ~94 MB)
# ---------------------------------------------------------------------------
echo "==> [2/7] RetCCL encoder weights ..."
if [ -f "$RETCCL_WEIGHTS" ]; then
    echo "    already present, skipping: $RETCCL_WEIGHTS"
else
    mkdir -p "$RETCCL_DIR"
    if ! command -v gdown >/dev/null 2>&1; then
        echo "    installing gdown ..."
        python3 -m pip install --user --quiet gdown
        # Make sure the freshly user-installed gdown is on PATH for this shell.
        export PATH="$PATH:$(python3 -m site --user-base)/bin"
    fi
    # The Drive folder contains a single best_ckpt.pth; pull it into a temp dir,
    # then rename to what inference.py expects.
    TMP_RETCCL="$(mktemp -d)"
    gdown --folder "$RETCCL_GDRIVE_FOLDER" -O "$TMP_RETCCL"
    SRC="$(find "$TMP_RETCCL" -name 'best_ckpt.pth' -type f | head -n1)"
    if [ -z "$SRC" ]; then
        echo "ERROR: could not find best_ckpt.pth in the downloaded RetCCL folder." >&2
        rm -rf "$TMP_RETCCL"
        exit 1
    fi
    mv "$SRC" "$RETCCL_WEIGHTS"
    rm -rf "$TMP_RETCCL"
    echo "    -> $RETCCL_WEIGHTS"
fi

# ---------------------------------------------------------------------------
# 3. ECTIL classifier weights (~4.7 MB)
# ---------------------------------------------------------------------------
echo "==> [3/7] ECTIL classifier weights ..."
if [ -f "$ECTIL_WEIGHTS" ]; then
    echo "    already present, skipping: $ECTIL_WEIGHTS"
else
    mkdir -p "$ECTIL_DIR"
    curl -fL "$ECTIL_URL" -o "$ECTIL_WEIGHTS"
    echo "    -> $ECTIL_WEIGHTS"
fi

# ---------------------------------------------------------------------------
# 4. 5 whole-slide images (TCGA-BRCA, GDC data endpoint)
# ---------------------------------------------------------------------------
echo "==> [4/7] TCGA-BRCA slides ($NUM_SLIDES total) ..."
mkdir -p "$WSI_DIR"
for FID in "${GDC_FILE_IDS[@]}"; do
    # GDC rejects HEAD, so we cannot probe for the Content-Disposition filename
    # cheaply; instead key skip-if-exists on a per-id marker, and download into a
    # private temp dir so a re-run never collides with an already-present .svs.
    MARKER="$WSI_DIR/.$FID.done"
    if [ -f "$MARKER" ]; then
        echo "    [$FID] already present, skipping."
        continue
    fi
    echo "    [$FID] downloading ..."
    TMP_DL="$(mktemp -d)"
    # -J -O preserves the server's Content-Disposition .svs filename.
    ( cd "$TMP_DL" && curl -fL -J -O "https://api.gdc.cancer.gov/data/$FID" )
    DL_SVS="$(find "$TMP_DL" -name '*.svs' -type f | head -n1)"
    if [ -z "$DL_SVS" ]; then
        echo "ERROR: GDC id $FID did not yield an .svs file." >&2
        rm -rf "$TMP_DL"
        exit 1
    fi
    mv -f "$DL_SVS" "$WSI_DIR/"
    rm -rf "$TMP_DL"
    touch "$MARKER"
done

# Count without bash-4 builtins (macOS ships bash 3.2, which lacks mapfile).
HAVE_SLIDES="$(find "$WSI_DIR" -maxdepth 1 -name '*.svs' -type f | wc -l | tr -d ' ')"
if [ "$HAVE_SLIDES" -ne "$NUM_SLIDES" ]; then
    echo "ERROR: expected exactly $NUM_SLIDES slides in $WSI_DIR, found $HAVE_SLIDES." >&2
    echo "       (Directory mode asserts a $NUM_SLIDES-row result, so $WSI_DIR must hold only these slides.)" >&2
    exit 1
fi
echo "    have $HAVE_SLIDES slides in $WSI_DIR."

# Pick the smallest slide for the single-slide run to keep that run fast.
SINGLE_SLIDE="$(find "$WSI_DIR" -maxdepth 1 -name '*.svs' -type f -exec ls -S {} + | tail -n1)"
echo "    single-slide run will use: $(basename "$SINGLE_SLIDE")"

# ---------------------------------------------------------------------------
# 5. Build the Docker image (cached after the first build)
# ---------------------------------------------------------------------------
echo "==> [5/7] Building Docker image '$IMAGE' (cached after first build) ..."
docker build -t "$IMAGE" .

# ---------------------------------------------------------------------------
# 6. Run the pipeline in the container, CPU, in two modes.
#    Volume-mount pattern mirrors tools/infer/infer_docker.sh.
# ---------------------------------------------------------------------------
mkdir -p "$OUTPUT_DIR"

run_ectil () {  # $1 = run name, $2 = --wsi target inside the container
    local run_name="$1" wsi_arg="$2"
    # TCGA-BRCA diagnostic SVS are scanned at 40x but often ship without an
    # embedded micron-per-pixel; --overwrite-mpp 0.25 supplies the native spacing
    # so dlup can tile them (otherwise it raises UnsupportedSlideError).
    docker run --rm \
        -v "$REPO_ROOT/$WSI_DIR":/input:ro \
        -v "$REPO_ROOT/$ECTIL_DIR":/weights/ectil:ro \
        -v "$REPO_ROOT/$RETCCL_DIR":/weights/retccl:ro \
        -v "$REPO_ROOT/$OUTPUT_DIR":/output \
        "$IMAGE" \
            --wsi "$wsi_arg" \
            --classifier-weights "/weights/ectil/$(basename "$ECTIL_WEIGHTS")" \
            --retccl-weights "/weights/retccl/$(basename "$RETCCL_WEIGHTS")" \
            --output /output \
            --run-name "$run_name" \
            --overwrite-mpp 0.25 \
            --device cpu
}

SINGLE_RUN="demo_single"
DIR_RUN="demo_dir"

# These run names are fixed (not timestamped), so clear any leftovers from a
# previous invocation to keep the validation counts deterministic.
rm -rf "$OUTPUT_DIR/$SINGLE_RUN" "$OUTPUT_DIR/$DIR_RUN"

echo "==> [6/7] Running SINGLE-SLIDE mode ($(basename "$SINGLE_SLIDE")) ..."
run_ectil "$SINGLE_RUN" "/input/$(basename "$SINGLE_SLIDE")"

echo "==> [6/7] Running DIRECTORY mode (all $NUM_SLIDES slides in $WSI_DIR) ..."
run_ectil "$DIR_RUN" "/input"

# ---------------------------------------------------------------------------
# 7. Validate the outputs of both runs (the actual smoke test).
# ---------------------------------------------------------------------------
echo "==> [7/7] Validating outputs ..."

FAIL=0
fail () { echo "  [FAIL] $1" >&2; FAIL=1; }
ok ()   { echo "  [ok]   $1"; }

# Count data rows (everything after the header) in a CSV.
csv_data_rows () { [ -f "$1" ] && { local n; n=$(($(wc -l < "$1") - 1)); [ "$n" -lt 0 ] && n=0; echo "$n"; } || echo 0; }

# Parse + range-check the til_score from a per-slide tils_score.json. Echoes
# "slide_id\tscore" on success; returns non-zero on any problem.
parse_score () {
    python3 - "$1" <<'PY'
import json, sys
p = sys.argv[1]
try:
    d = json.load(open(p))
    s = float(d["til_score"])
except Exception as e:
    print(f"parse-error: {e}", file=sys.stderr); sys.exit(1)
if not (0.0 <= s <= 1.0):
    print(f"out-of-range: {s}", file=sys.stderr); sys.exit(1)
print(f"{d.get('slide_id','?')}\t{s:.4f}")
PY
}

# Required per-slide artifacts.
PER_SLIDE_FILES=(
    tils_score.json tile_predictions.csv features.h5
    thumbnail.png mask.png mask_overlay.png
    attention_heatmap.png til_heatmap.png
)

validate_run () {  # $1 = run name, $2 = expected number of slide rows/subdirs
    local run_name="$1" expect="$2"
    local run_dir="$OUTPUT_DIR/$run_name"
    echo "  -- run '$run_name' (expecting $expect slide(s)) --"

    [ -d "$run_dir" ] && ok "run dir exists: $run_dir" || { fail "run dir missing: $run_dir"; return; }

    local scores_csv="$run_dir/tils_scores.csv"
    if [ -s "$scores_csv" ]; then ok "tils_scores.csv exists and is non-empty"
    else fail "tils_scores.csv missing or empty: $scores_csv"; return; fi

    local rows; rows=$(csv_data_rows "$scores_csv")
    [ "$rows" -eq "$expect" ] && ok "tils_scores.csv has $rows data row(s)" \
        || fail "tils_scores.csv has $rows data row(s), expected $expect"

    # Per-slide subdirs (exclude the top-level config.json / tils_scores.csv).
    local subdirs; subdirs=$(find "$run_dir" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')
    [ "$subdirs" -eq "$expect" ] && ok "$subdirs per-slide subdir(s)" \
        || fail "$subdirs per-slide subdir(s), expected $expect"

    # Each subdir must hold all required artifacts and a sane, parseable score.
    local d f
    for d in "$run_dir"/*/; do
        [ -d "$d" ] || continue
        for f in "${PER_SLIDE_FILES[@]}"; do
            [ -s "$d$f" ] || fail "missing/empty artifact: $d$f"
        done
        local line
        if line=$(parse_score "$d/tils_score.json"); then
            ok "score sane for $(basename "$d"): $(echo "$line" | cut -f2)"
        else
            fail "bad TIL score in $d/tils_score.json"
        fi
    done
}

validate_run "$SINGLE_RUN" 1
validate_run "$DIR_RUN" "$NUM_SLIDES"

echo ""
echo "============================================================"
echo " Per-slide TIL scores (directory run):"
DIR_RUN_DIR="$OUTPUT_DIR/$DIR_RUN"
for d in "$DIR_RUN_DIR"/*/; do
    [ -d "$d" ] || continue
    line=$(parse_score "$d/tils_score.json" 2>/dev/null || true)
    if [ -n "$line" ]; then
        sid=$(echo "$line" | cut -f1); sc=$(echo "$line" | cut -f2)
        printf "   %-60s TIL = %s (%.1f%%)\n" "$sid" "$sc" "$(python3 -c "print($sc*100)")"
    fi
done

# Single, headline TIL score = the single-slide run.
SINGLE_ID="$(basename "$SINGLE_SLIDE" .svs)"
SINGLE_JSON="$OUTPUT_DIR/$SINGLE_RUN/$SINGLE_ID/tils_score.json"
SINGLE_SCORE="$(python3 -c "import json;print(f\"{json.load(open('$SINGLE_JSON'))['til_score']:.4f}\")" 2>/dev/null || echo '?')"
echo "============================================================"

if [ "$FAIL" -eq 0 ]; then
    echo "SMOKE TEST PASSED"
    echo "TIL score (single-slide $SINGLE_ID): $SINGLE_SCORE   |   run dir: $OUTPUT_DIR/$SINGLE_RUN"
    exit 0
else
    echo "SMOKE TEST FAILED  (see [FAIL] lines above)"
    exit 1
fi
