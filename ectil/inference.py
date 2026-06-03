"""End-to-end WSI inference for ECTIL.

Given a whole-slide image (WSI) -- or a directory of WSIs -- and a path to
ECTIL classifier weights, this runs the full pipeline in one command:

    tissue mask  ->  foreground tiling  ->  RetCCL features  ->  ECTIL

and writes everything needed for practical/clinical use into a timestamped run
directory, with one subdir per slide and an aggregate scores table:

    <output>/<run_name>/
        config.json             full run configuration
        tils_scores.csv         one row per slide (score, status) for easy analysis
        <slide_id>/
            tils_score.json     slide-level TIL score + full config
            tile_predictions.csv per-tile TIL score, attention weight, region
            features.h5         the generated dataset (RetCCL features + tile meta)
            thumbnail.png       plain slide thumbnail
            mask.png            tissue mask used for tiling
            mask_overlay.png    mask drawn on the thumbnail (sanity check)
            attention_heatmap.png per-tile attention painted on the thumbnail
            til_heatmap.png     per-tile TIL score painted on the thumbnail

`<run_name>` defaults to a timestamp (override with --run-name). RetCCL is
loaded automatically; only the ECTIL classifier weights have to be provided
explicitly. This reuses the same components as the training/extraction pipeline
(DLUP tiling + FESI mask, RetCCL encoder, MeanMIL + GatedAttention), so results
match `extract.py` + `eval.py`.

Batch mode skips slides it cannot process and records the failure in
tils_scores.csv rather than aborting the whole run.

Examples:
    # single slide
    python -m ectil.inference \
        --wsi /input/slide.svs \
        --classifier-weights /weights/ectil_fold_0_weights_only.ckpt \
        --output /output

    # a directory of slides (recursively globbed by extension)
    python -m ectil.inference \
        --wsi /input/cohort \
        --classifier-weights /weights/ectil_fold_0_weights_only.ckpt \
        --output /output
"""

import argparse
import csv
import json
import logging
import os
from pathlib import Path
from typing import Optional

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from dlup import SlideImage
from dlup.tiling import GridOrder, TilingMode
from PIL import Image
from torch.nn import Identity, Linear, ReLU, Sequential, Sigmoid
from torch.utils.data import DataLoader

from ectil.datamodules.components.dlup_dataset import (
    DLUPDatasetWrapper,
    compute_mask,
    save_overlay,
    transform_factory,
)
from ectil.models.components import GatedAttention, MeanMIL, RetCCL
from ectil.models.extraction_module import H5Writer
from ectil.utils.background import AvailableMaskFunctions

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("ectil.inference")

# Default location of the RetCCL weights inside the repo / container.
# Can be overridden with --retccl-weights or the RETCCL_WEIGHTS env var.
DEFAULT_RETCCL_WEIGHTS = (
    Path(__file__).resolve().parent.parent
    / "model_zoo"
    / "retccl"
    / "retccl_best_ckpt.pth"
)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def build_ectil(
    in_features: int,
    hidden_features: int,
    attention_hidden_features: int,
) -> MeanMIL:
    """Instantiate the ECTIL MeanMIL model.

    The Identity() layers stand in for the Dropout layers used during training so
    that the indices in the Sequential modules match the saved state dict keys.
    """
    return MeanMIL(
        post_encoder=Sequential(
            Identity(),
            Identity(),
            Linear(in_features=in_features, out_features=hidden_features, bias=True),
            ReLU(),
        ),
        classifier=Sequential(
            Identity(),
            Identity(),
            Linear(in_features=hidden_features, out_features=1, bias=True),
            Sigmoid(),
        ),
        attention=GatedAttention(
            in_features=hidden_features, hidden_features=attention_hidden_features
        ),
    ).eval()


def load_ectil_weights(model: MeanMIL, ckpt_path: Path, device: torch.device) -> None:
    weights = torch.load(ckpt_path, map_location=device, weights_only=True)
    # Saved checkpoints are the `net` state dict of the LightningModule and keep a
    # `net.` prefix; a plain torch model expects the prefix stripped.
    weights = {k.replace("net.", "", 1): v for k, v in weights.items()}
    model.load_state_dict(weights)


def save_mask_images(slide: SlideImage, mask: np.ndarray, out_dir: Path) -> None:
    """Write a viewable tissue mask and a mask-on-thumbnail overlay."""
    mask_path = out_dir / "mask.png"
    Image.fromarray((mask.astype(np.uint8) * 255)).save(mask_path)
    # save_overlay derives `<stem>_overlay.png` next to mask_path, i.e. mask_overlay.png
    save_overlay(mask_path=mask_path, mask=mask, slide=slide)


def extract_features(
    slide_path: Path,
    mask: np.ndarray,
    encoder: RetCCL,
    device: torch.device,
    mpp: float,
    tile_size: int,
    mask_threshold: float,
    batch_size: int,
    num_workers: int,
    overwrite_mpp: Optional[float] = None,
):
    """Tile the foreground and extract RetCCL features.

    Returns the stacked per-tile outputs (numpy), the list of tile regions
    (x, y, w, h, mpp) aligned with the features, and the dataset.
    """
    transform = transform_factory("imagenet_normalization")
    # `overwrite_mpp` is forwarded to dlup for slides that lack an embedded
    # spacing (common for TCGA SVS); it sets the native micron-per-pixel so the
    # requested tiling `mpp` can be resolved. Tiling is unaffected for slides
    # that already carry a spacing.
    extra = {"overwrite_mpp": (overwrite_mpp, overwrite_mpp)} if overwrite_mpp else {}
    dataset = DLUPDatasetWrapper.from_standard_tiling(
        path=slide_path,
        mpp=mpp,
        tile_size=(tile_size, tile_size),
        tile_overlap=(0, 0),
        tile_mode=TilingMode.skip,
        grid_order=GridOrder.C,
        crop=False,
        transform=transform,
        mask=mask,
        mask_threshold=mask_threshold,
        limit_bounds=True,
        **extra,
    )
    if len(dataset) == 0:
        raise RuntimeError(
            "No foreground tiles were selected. The tissue mask may be empty; "
            "try a different --mask-function or check the slide."
        )

    loader = DataLoader(
        dataset=dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    outputs = []
    encoder = encoder.to(device).eval()
    log.info(f"Extracting RetCCL features for {len(dataset)} tiles")
    with torch.no_grad():
        for batch in loader:
            batch["image"] = encoder(batch["image"].to(device)).cpu()
            outputs.append(batch)

    stacked = H5Writer(h5_root_dir="").stack_output(outputs)

    # Regions (x, y, w, h, mpp) aligned with the masked tiles, same as extract.py.
    regions = [
        region
        for idx, region in enumerate(dataset.regions)
        if idx in set(dataset.masked_indices)
    ]
    return stacked, regions, dataset


def run_ectil(model: MeanMIL, features: np.ndarray, device: torch.device):
    model = model.to(device).eval()
    x = torch.from_numpy(features).float().unsqueeze(0).to(device)  # 1 x n_tiles x dim
    with torch.no_grad():
        out = model(x)
    score = float(out["out"].reshape(-1)[0].item())
    til = out["meta"]["out_per_instance"].reshape(-1).cpu().numpy()
    attention = out["meta"]["attention_weights"].reshape(-1).cpu().numpy()
    return score, til, attention


def save_features_h5(
    out_path: Path,
    stacked: dict,
    regions: np.ndarray,
    slide_id: str,
    slide_path: Path,
    til: np.ndarray,
    attention: np.ndarray,
) -> None:
    with h5py.File(out_path, "w") as hf:
        hf.create_dataset("features", data=stacked["image"])
        if regions is not None and len(regions):
            hf.create_dataset("regions", data=np.asarray(regions, dtype=float))
        for key in [
            "coordinates",
            "mpp",
            "region_index",
            "grid_local_coordinates",
            "grid_index",
        ]:
            if key in stacked:
                hf.create_dataset(key, data=stacked[key])
        hf.create_dataset("tile_level_output", data=til)
        hf.create_dataset("attention_weights", data=attention)
        hf.attrs["slide_id"] = slide_id
        hf.attrs["path"] = str(slide_path)


def tile_regions_for_output(stacked: dict, regions: list, n_tiles: int, mpp: float, tile_size: int):
    """Return regions (x, y, w, h, mpp) guaranteed aligned with the n_tiles outputs.

    Prefer the DLUP regions; if their count does not match (defensive), rebuild
    them from the per-tile coordinates, which are always aligned with the features.
    """
    if len(regions) == n_tiles:
        return np.asarray(regions, dtype=float)
    coords = np.asarray(stacked["coordinates"], dtype=float)  # n x 2
    rebuilt = np.zeros((n_tiles, 5), dtype=float)
    rebuilt[:, 0] = coords[:, 0]
    rebuilt[:, 1] = coords[:, 1]
    rebuilt[:, 2] = tile_size
    rebuilt[:, 3] = tile_size
    rebuilt[:, 4] = mpp
    return rebuilt


def save_tile_csv(out_path: Path, regions: np.ndarray, til: np.ndarray, attention: np.ndarray) -> None:
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "w", "h", "mpp", "tile_level_output", "attention_weights"])
        for r, t, a in zip(regions, til, attention):
            writer.writerow([r[0], r[1], r[2], r[3], r[4], float(t), float(a)])


def make_heatmap(
    slide: SlideImage,
    thumb: Image.Image,
    mpp: float,
    regions: np.ndarray,
    values: np.ndarray,
    title: str,
    out_path: Path,
    cmap: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cbar_label: str = "",
) -> None:
    """Paint per-tile `values` onto the (pre-decoded) slide thumbnail and save as a PNG overlay."""
    scaling = slide.get_scaling(mpp)
    scaled_w, scaled_h = (np.asarray(slide.size, dtype=float) * scaling)
    tw, th = thumb.size
    sx = tw / scaled_w
    sy = th / scaled_h

    heat = np.full((th, tw), np.nan, dtype=float)
    for (x, y, w, h, _), v in zip(regions, values):
        x0, y0 = int(round(x * sx)), int(round(y * sy))
        x1, y1 = int(round((x + w) * sx)), int(round((y + h) * sy))
        x0, x1 = max(0, x0), min(tw, x1)
        y0, y1 = max(0, y0), min(th, y1)
        if x1 > x0 and y1 > y0:
            heat[y0:y1, x0:x1] = v

    fig, ax = plt.subplots(figsize=(tw / 100.0, th / 100.0), dpi=100)
    ax.imshow(np.asarray(thumb))
    im = ax.imshow(
        np.ma.masked_invalid(heat), cmap=cmap, alpha=0.5, vmin=vmin, vmax=vmax
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if cbar_label:
        cbar.set_label(cbar_label)
    ax.set_title(title)
    ax.axis("off")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# Whole-slide formats globbed by default when --wsi points to a directory.
# Note on MRXS: a `.mrxs` slide is the single file you open; it is accompanied by
# a same-named directory of raw data. Globbing by extension matches the `.mrxs`
# file (which dlup/openslide opens) and never the companion directory.
DEFAULT_SLIDE_GLOB = "*.svs,*.tif,*.tiff,*.ndpi,*.mrxs,*.scn,*.svslide,*.bif"


def discover_slides(input_path: Path, glob_patterns: str) -> list:
    """Return the list of slide files for `input_path` (a file or a directory)."""
    p = input_path.expanduser().resolve()
    if p.is_file():
        return [p]
    if not p.is_dir():
        raise FileNotFoundError(f"WSI path not found: {p}")

    patterns = [pat.strip() for pat in glob_patterns.split(",") if pat.strip()]
    slides: list = []
    seen = set()
    for pattern in patterns:
        for match in sorted(p.rglob(pattern)):
            if match.is_file() and match not in seen:
                seen.add(match)
                slides.append(match)
    return slides


def build_config(args: argparse.Namespace, device: torch.device, run_name: str) -> dict:
    """All run configuration, embedded in each slide's JSON and the run config.json."""
    return {
        "run_name": run_name,
        "device": str(device),
        "mpp": args.mpp,
        "overwrite_mpp": args.overwrite_mpp,
        "tile_size": args.tile_size,
        "mask_function": args.mask_function,
        "mask_threshold": args.mask_threshold,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "heatmap_size": args.heatmap_size,
        "in_features": args.in_features,
        "hidden_features": args.hidden_features,
        "attention_hidden_features": args.attention_hidden_features,
        "classifier_weights": str(Path(args.classifier_weights).expanduser()),
        "retccl_weights": str(Path(args.retccl_weights).expanduser()),
    }


def _unique_slide_dir(run_dir: Path, slide_id: str) -> Path:
    """Per-slide output dir, suffixed if two input slides share a filename stem."""
    out_dir = run_dir / slide_id
    suffix = 1
    while out_dir.exists():
        out_dir = run_dir / f"{slide_id}_{suffix}"
        suffix += 1
    out_dir.mkdir(parents=True)
    return out_dir


def process_slide(
    wsi_path: Path,
    run_dir: Path,
    encoder: RetCCL,
    model: MeanMIL,
    device: torch.device,
    config: dict,
    args: argparse.Namespace,
) -> dict:
    """Run the full pipeline for one slide and write its outputs. Returns a summary row."""
    slide_id = wsi_path.stem
    out_dir = _unique_slide_dir(run_dir, slide_id)
    log.info(f"[{slide_id}] writing outputs to {out_dir}")

    # 1. Tissue mask + thumbnail. Decode the thumbnail once and reuse it for both
    # the saved PNG and the two heatmap overlays below (a thumbnail decode reads and
    # resamples a pyramid level, so it is the expensive part to avoid repeating).
    # overwrite_mpp lets slides without an embedded spacing (e.g. many TCGA SVS)
    # still be opened/tiled; dlup otherwise raises UnsupportedSlideError.
    open_kwargs = (
        {"overwrite_mpp": (args.overwrite_mpp, args.overwrite_mpp)}
        if args.overwrite_mpp
        else {}
    )
    slide = SlideImage.from_file_path(wsi_path, **open_kwargs)
    log.info(f"[{slide_id}] computing tissue mask with '{args.mask_function}'")
    mask = compute_mask(slide=slide, mask_function=args.mask_function)
    thumb = slide.get_thumbnail(size=(args.heatmap_size, args.heatmap_size)).convert("RGB")
    thumb.save(out_dir / "thumbnail.png")
    save_mask_images(slide=slide, mask=mask, out_dir=out_dir)

    # 2. Foreground tiling + RetCCL feature extraction
    stacked, regions, _ = extract_features(
        slide_path=wsi_path,
        mask=mask,
        encoder=encoder,
        device=device,
        mpp=args.mpp,
        tile_size=args.tile_size,
        mask_threshold=args.mask_threshold,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        overwrite_mpp=args.overwrite_mpp,
    )
    features = stacked["image"]
    n_tiles = features.shape[0]

    # 3. ECTIL classifier
    score, til, attention = run_ectil(model, features, device)
    log.info(f"[{slide_id}] TIL score: {score:.4f} ({score * 100:.1f}%) over {n_tiles} tiles")

    # 4. Persist results
    out_regions = tile_regions_for_output(stacked, regions, n_tiles, args.mpp, args.tile_size)
    save_features_h5(
        out_dir / "features.h5", stacked, out_regions, slide_id, wsi_path, til, attention
    )
    save_tile_csv(out_dir / "tile_predictions.csv", out_regions, til, attention)

    summary = {
        "slide_id": slide_id,
        "slide_path": str(wsi_path),
        "til_score": score,
        "til_score_percent": score * 100.0,
        "num_tiles": int(n_tiles),
        "output_dir": str(out_dir),
        **config,
    }
    with open(out_dir / "tils_score.json", "w") as f:
        json.dump(summary, f, indent=2)

    # 5. Heatmaps
    make_heatmap(
        slide=slide,
        thumb=thumb,
        mpp=args.mpp,
        regions=out_regions,
        values=attention,
        title=f"{slide_id} - attention",
        out_path=out_dir / "attention_heatmap.png",
        cmap="viridis",
        cbar_label="attention weight",
    )
    make_heatmap(
        slide=slide,
        thumb=thumb,
        mpp=args.mpp,
        regions=out_regions,
        values=til,
        title=f"{slide_id} - tile-level TIL (slide score {score * 100:.1f}%)",
        out_path=out_dir / "til_heatmap.png",
        cmap="jet",
        vmin=0.0,
        vmax=1.0,
        cbar_label="tile TIL score",
    )
    return summary


def run_inference(args: argparse.Namespace) -> dict:
    """Discover one or more slides, run ECTIL on each, and aggregate the results."""
    import csv
    from datetime import datetime

    device = resolve_device(args.device)

    retccl_weights = Path(args.retccl_weights).expanduser()
    if not retccl_weights.is_file():
        raise FileNotFoundError(
            f"RetCCL weights not found at {retccl_weights}. Provide them via "
            "--retccl-weights or the RETCCL_WEIGHTS env var (see model_zoo/retccl/readme.md)."
        )
    classifier_weights = Path(args.classifier_weights).expanduser()
    if not classifier_weights.is_file():
        raise FileNotFoundError(f"ECTIL classifier weights not found: {classifier_weights}")

    slides = discover_slides(Path(args.wsi), args.glob)
    if not slides:
        raise FileNotFoundError(
            f"No slides matching '{args.glob}' found under {args.wsi}"
        )

    run_name = args.run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(args.output).expanduser() / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    config = build_config(args, device, run_name)
    with open(run_dir / "config.json", "w") as f:
        json.dump({**config, "num_slides": len(slides)}, f, indent=2)
    log.info(f"Found {len(slides)} slide(s); writing run to {run_dir} (device={device})")

    # Load the encoder and classifier once and reuse them across all slides.
    encoder = RetCCL(project_root_dir="", weights_path=str(retccl_weights)).to(device).eval()
    model = build_ectil(
        in_features=args.in_features,
        hidden_features=args.hidden_features,
        attention_hidden_features=args.attention_hidden_features,
    )
    load_ectil_weights(model, classifier_weights, device)
    model = model.to(device).eval()

    summary_path = run_dir / "tils_scores.csv"
    fieldnames = [
        "slide_id",
        "slide_path",
        "til_score",
        "til_score_percent",
        "num_tiles",
        "status",
        "error",
        "output_dir",
    ]
    n_ok = 0
    # Write the summary incrementally so partial results survive an interrupted batch.
    with open(summary_path, "w", newline="") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for i, wsi_path in enumerate(slides, start=1):
            log.info(f"({i}/{len(slides)}) {wsi_path}")
            try:
                row = process_slide(wsi_path, run_dir, encoder, model, device, config, args)
                row.update({"status": "ok", "error": ""})
                n_ok += 1
            except Exception as exc:  # keep going on per-slide failures
                log.exception(f"Failed to process {wsi_path}: {exc}")
                row = {
                    "slide_id": wsi_path.stem,
                    "slide_path": str(wsi_path),
                    "status": "failed",
                    "error": str(exc),
                }
            writer.writerow(row)
            summary_file.flush()

    log.info(f"Done. {n_ok}/{len(slides)} slide(s) succeeded. Summary: {summary_path}")
    return {"run_dir": str(run_dir), "summary": str(summary_path), "n_ok": n_ok, "n_total": len(slides)}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end ECTIL TIL inference on a WSI or a directory of WSIs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--wsi", required=True, help="Path to a WSI file or a directory of WSIs."
    )
    parser.add_argument(
        "--glob",
        default=DEFAULT_SLIDE_GLOB,
        help="Comma-separated glob patterns used (recursively) when --wsi is a directory.",
    )
    parser.add_argument(
        "--classifier-weights",
        required=True,
        help="Path to the ECTIL classifier weights (e.g. *_weights_only.ckpt).",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output directory; a timestamped run subdir with per-slide subdirs is created.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Name of the run subdir under --output. Defaults to a timestamp.",
    )
    parser.add_argument(
        "--retccl-weights",
        default=os.environ.get("RETCCL_WEIGHTS", str(DEFAULT_RETCCL_WEIGHTS)),
        help="Path to RetCCL weights. Auto-loaded from model_zoo / RETCCL_WEIGHTS by default.",
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda"], help="Compute device."
    )
    parser.add_argument("--mpp", type=float, default=0.5, help="Microns per pixel for tiling.")
    parser.add_argument(
        "--overwrite-mpp",
        type=float,
        default=None,
        help="Native microns-per-pixel to assume when a slide has no embedded spacing "
        "(e.g. many TCGA SVS). For TCGA-BRCA 40x diagnostic slides this is 0.25. "
        "Leave unset to use the slide's own spacing.",
    )
    parser.add_argument("--tile-size", type=int, default=512, help="Tile size in pixels.")
    parser.add_argument(
        "--mask-function",
        default="fesi",
        choices=list(AvailableMaskFunctions.__members__),
        help="Tissue foreground segmentation function.",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.1,
        help="Minimum foreground fraction for a tile to be kept.",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="RetCCL extraction batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument(
        "--heatmap-size", type=int, default=2048, help="Long-edge size of thumbnails/heatmaps."
    )
    parser.add_argument("--in-features", type=int, default=2048, help="RetCCL feature dimension.")
    parser.add_argument("--hidden-features", type=int, default=512, help="ECTIL hidden dimension.")
    parser.add_argument(
        "--attention-hidden-features", type=int, default=128, help="Attention hidden dimension."
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    run_inference(parse_args(argv))


if __name__ == "__main__":
    main()
