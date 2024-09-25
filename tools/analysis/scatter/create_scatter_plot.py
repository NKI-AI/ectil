"""Generates scatter plots from Figure S2

Returns:
    No return. Saves the pdf file with scatter to a location.

Usage:
    ~/ectil$ conda activate ectil && python -m tools.analysis.scatter.create_scatter_plot
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_scatter(df: pd.DataFrame, x: str, y: str, log_name: Path):
    log_name.parent.mkdir(parents=True, exist_ok=False)
    ax1 = df.plot.scatter(
        x=x,
        y=y,
        color=(0.686, 0.933, 0.933, 1.0),  # PastelBlue
        s=50,
        alpha=0.1,
        edgecolors=(0.255, 0.412, 0.882, 0.6),  # RoyalBlue
        linewidths=2,
    )

    ax1.set_xlim(-5, 105)
    ax1.set_ylim(-5, 105)
    ax1.set_xlabel("ECTIL TILs")
    ax1.set_ylabel("Pathologist TILs")

    ax1.plot(
        [0, 100],
        [0, 100],
        color="LightGray",
        linestyle="--",
        linewidth=1.5,
        label="Perfect model",
    )
    coefficients = np.polyfit(df[x], df[y], 1)
    r = np.corrcoef(df[x], df[y])[0, 1]
    polynomial = np.poly1d(coefficients)
    fit_x = np.linspace(df[x].min(), df[x].max(), 100)
    plt.plot(
        fit_x,
        polynomial(fit_x),
        color="SandyBrown",
        linewidth=2,
        label=f"Fit Line (r = {r:.2f})",
    )

    plt.legend(loc="lower right")

    fig = plt.gcf()
    print(f"Saving scatter plot figure to {log_name}")
    fig.savefig(log_name)


if __name__ == "__main__":
    pth = lambda fold: f"logs/tcga_output/slide-level-output_test_fold_{fold}.csv"
    df = pd.read_csv(pth(0))
    for fold in [1, 2, 3, 4]:
        df = pd.concat([df, pd.read_csv(pth(fold))])

    df["preds"] = df["preds"] * 100
    df["gtruth"] = df["targets"] * 100
    df["study"] = "TCGA"  # Set study name, only for TCGA now
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    plot_scatter(
        df=df,
        x="preds",
        y="gtruth",
        log_name=Path(f"logs/scatter/{current_time}/scatter.pdf"),
    )
