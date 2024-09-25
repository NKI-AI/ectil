"""Creates a calibration curve for the results on a single cohort.

Returns:
    No return. Saves the image to a location.

Usage:
    ~/ectil$ conda activate ectil && python -m tools.analysis.calibration_curve.create_calibration_curve
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import NaN


def get_current_bin_targets(df, bin_min, bin_max) -> pd.Series:
    return df.loc[  # select...
        df["preds"].between(
            left=bin_min, right=bin_max, inclusive="left"
        ),  # ... only those rows where the predicted value is 0-0.05
        "targets",  # ... and look at the targets
    ]


def main(
    scores_csv: str, log_dir: str | Path, filename: str = "calibration_curve.pdf"
) -> None:
    """Creates the calibration curve.

     Read the csv of predicted scores and targets

     Bin them into 20 bins

    compute p(y | y_hat)
    i.e, given that the predicted bin is 0-0.05, what is the probability that it's actually 0-0.05
    i.e. p(y | y_hat ) = p(y_hat | y) * p(y) / p(y_hat)
    So let's say we have 20 samples to predicted to be 0-0.05, we see how many of those samples are actually 0-0.0

    Args:
         scores_csv (str): _description_
         log_dir (str): _description_

    Returns:
        None: It will save the file into the log_dir
    """

    # set the log dir with experiment name and current time.
    experiment_name = Path(
        scores_csv
    ).parent.parent.stem  # experiments are saved as experiment_name/runs/ensemble.csv
    current_time = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path(log_dir) / experiment_name / current_time
    log_dir.mkdir(exist_ok=False, parents=True)

    df = pd.read_csv(scores_csv)

    Es = []  # Expected values
    stds = []  # standard deviations
    cis = []  # confidence intervals

    # Set bins
    num_bins = 20
    bins_range = range(0, num_bins)
    bins = [(i / num_bins, (i + 1) / num_bins) for i in bins_range]

    for bin_min, bin_max in bins:
        # p(y = bin | y_hat = bin), i.e., we check all instances where y_hat is bin, and check how many of those predictions are actually bin

        # Get all samples where the predictions are 0-0.05
        # Get those targets
        # Compute the mean
        # Which is E[y] given that y_hat is in a specific bin.

        current_bin_targets = get_current_bin_targets(df, bin_min, bin_max)

        Es.append(
            # ... to compute the mean of them
            current_bin_targets.mean()
        )

        cis.append(
            (current_bin_targets.quantile(0.1), current_bin_targets.quantile(0.9))
        )

        stds.append(current_bin_targets.std())

    # plot the conditional probabilities

    fig, ax = plt.subplots()

    ax.plot([0, 1], [0, 1], color="grey", linestyle="dashed")

    num_samples_in_bins = [
        df["preds"].between(left=bin[0], right=bin[1], inclusive="left").sum()
        for bin in bins
    ]  # True / False, when we sum this gives number of True which is within the bin.

    max_samples_in_bins = np.max(num_samples_in_bins)  # set the darkest grey here.
    min_samples_in_bins = np.min(num_samples_in_bins)  # Set the lightest grey here.

    for bin, E, num_samples_in_bin, std, (ci_10, ci_90) in zip(
        bins, Es, num_samples_in_bins, stds, cis
    ):
        grey_val = 1 - (
            (num_samples_in_bin - min_samples_in_bins)
            / (max_samples_in_bins - min_samples_in_bins)
        )  # 0 to 1. High number = black, low number = white
        ax.bar(
            x=bin[0],
            height=E,
            width=(1 / num_bins),
            color=str(grey_val),
            align="edge",
            label=num_samples_in_bin,
        )

        current_bin_targets = get_current_bin_targets(df, bin[0], bin[1])
        if np.isfinite(E) and np.isfinite(bin[0]):
            x = bin[0] + (1 / num_bins) / 2
            ax.text(
                x=x,
                y=E,
                s=str(num_samples_in_bin),
                ha="center",
                va="bottom",
            )

            vp = ax.violinplot(
                current_bin_targets,
                positions=[x],
                widths=[0.04],
                quantiles=[[0.25, 0.75]],
            )
            vp["bodies"][0].set_facecolor("blue")
            vp["bodies"][0].set_linewidth(0.1)

    ax.stairs(values=Es, edges=[bin[0] for bin in bins] + [1], color="red")

    ax.set_xlim(left=0, right=1)

    ax.set_xlabel("Binned ECTIL predicted TILs")

    ax.set_ylabel("Pathologist TILs distribution")

    ax.get_figure().savefig(log_dir / filename)

    print(f"Figure has been saved to {log_dir / filename}")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--scores_csv",
        type=str,
        help="Path to the csv file with predictions and targets",
        default="logs/tcga_output/slide-level-output_test_fold_0.csv",
    )
    argparser.add_argument(
        "--log_dir",
        type=str,
        help="Path to the directory where the calibration curve will be saved",
        default="logs/calibration_curve",
    )

    args = argparser.parse_args()

    main(scores_csv=args.scores_csv, log_dir=args.log_dir)
