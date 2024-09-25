"""Exact script used to compute the detailed metrics

Returns:
    No return. Saves the xlsx file with results to a location.

Usage:
    ~/ectil$ conda activate ectil && python -m tools.analysis.metrics.compute_metrics
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchmetrics.functional import auroc, average_precision, concordance_corrcoef
from torchmetrics.functional import mean_squared_error as mse
from torchmetrics.functional import pearson_corrcoef, spearman_corrcoef


def create_dataset_detailed_metrics(
    df: pd.DataFrame, current_time: str, log_dir: Path
) -> None:
    """Generates metrics as presented in Table S2

    Args:
        df (pd.DataFrame): dataframe with `preds`, `gtruth` and `study` column
        current_time (str): timestamp to add to the log_directory under which to save the xlsx file
        log_dir (Path): where to save the resulting xlsx file
    """
    log_dir = log_dir / current_time
    log_dir.mkdir(parents=True, exist_ok=False)
    log_file = log_dir / "metrics.xlsx"
    print(f"Saving to {log_file}")

    df.groupby("study").apply(
        lambda x: pd.Series(
            {
                "pearson r": pearson_corrcoef(
                    preds=torch.tensor(x["preds"].values),
                    target=torch.tensor(x["gtruth"].values).float(),
                ).item(),
                "spearman r": spearman_corrcoef(
                    preds=torch.tensor(x["preds"].values).float(),
                    target=torch.tensor(x["gtruth"].values).float(),
                ).item(),
                "concordance r": concordance_corrcoef(
                    preds=torch.tensor(x["preds"].values),
                    target=torch.tensor(x["gtruth"].values).float(),
                ).item(),
                "mse": [
                    int(x) if not np.isnan(x) else x
                    for x in [
                        mse(
                            preds=torch.tensor(x["preds"].values),
                            target=torch.tensor(x["gtruth"].values).float(),
                        ).item()
                    ]
                ][0],
                "AUROC@10": auroc(
                    preds=torch.tensor(x["preds"].values),
                    target=torch.tensor(x["gtruth"].values > 10),
                ).item(),
                "AUROC@30": auroc(
                    preds=torch.tensor(x["preds"].values),
                    target=torch.tensor(x["gtruth"].values > 30),
                ).item(),
                "AUROC@50": auroc(
                    preds=torch.tensor(x["preds"].values),
                    target=torch.tensor(x["gtruth"].values > 50),
                ).item(),
                "AUROC@75": auroc(
                    preds=torch.tensor(x["preds"].values),
                    target=torch.tensor(x["gtruth"].values > 75),
                ).item(),
                "random AP@10": (x["gtruth"].values > 10).mean(),
                "random AP@30": (x["gtruth"].values > 30).mean(),
                "random AP@50": (x["gtruth"].values > 50).mean(),
                "random AP@75": (x["gtruth"].values > 75).mean(),
                "AP@10": average_precision(
                    preds=torch.tensor(x["preds"].values),
                    target=torch.tensor(x["gtruth"].values > 10),
                ).item(),
                "AP@30": average_precision(
                    preds=torch.tensor(x["preds"].values),
                    target=torch.tensor(x["gtruth"].values > 30),
                ).item(),
                "AP@50": average_precision(
                    preds=torch.tensor(x["preds"].values),
                    target=torch.tensor(x["gtruth"].values > 50),
                ).item(),
                "AP@75": average_precision(
                    preds=torch.tensor(x["preds"].values),
                    target=torch.tensor(x["gtruth"].values > 75),
                ).item(),
            }
        )
    ).transpose().round(2).to_excel(log_file)


if __name__ == "__main__":
    pth = lambda fold: f"logs/tcga_output/slide-level-output_test_fold_{fold}.csv"
    df = pd.read_csv(pth(0))
    for fold in [1, 2, 3, 4]:
        df = pd.concat([df, pd.read_csv(pth(fold))])

    df["preds"] = df["preds"] * 100
    df["gtruth"] = df["targets"] * 100
    df["study"] = "TCGA"  # Set study name, only for TCGA now
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    create_dataset_detailed_metrics(
        df, current_time=current_time, log_dir=Path("logs/metrics")
    )
