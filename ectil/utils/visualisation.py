import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from torch import Tensor
from torchmetrics.functional.regression import (
    explained_variance,
    mean_squared_error,
    pearson_corrcoef,
)

from ectil import utils

log = utils.get_pylogger(__name__)


def log_plot(trainer: Trainer, fig: Figure, tag: str) -> bool:
    """
    Obtain the tensorboard loggers from the trainer.
    Throws a warning when more than one tensorboard logger is found.
    """
    filtered_loggers = [
        logger for logger in trainer.loggers if isinstance(logger, TensorBoardLogger)
    ]
    if len(filtered_loggers) != 1:
        log.warning(
            f"You have {len(filtered_loggers)} TensorBoardLoggers, while we expect at least and at most 1. Skipping logging of plots"
        )
        return False
    else:
        tb = filtered_loggers[0].experiment
        # We set global step to trainer.global_step-1, because it appears that
        #  at validation_end the global step is increased by 1. We would like it to match the
        #  scalar logging in tensorboard
        tb.add_figure(tag=f"{tag}", figure=fig, global_step=trainer.global_step - 1)
        return True


def plot_scatter(preds: Tensor, targets: Tensor) -> Figure:
    """
    Plot a scatter plot continuous (in [0,1]) predictions and targets (in [0,1])
    with MSE, explained variance, and a pearson's r.
    """
    sns.set_style("darkgrid")
    mse = mean_squared_error(preds=preds, target=targets)
    r2 = explained_variance(preds=preds, target=targets)
    pearson_r = pearson_corrcoef(preds=preds, target=targets)
    g = sns.regplot(x=targets.numpy(), y=preds.numpy(), fit_reg=True)
    g.set(
        xlabel="True labels",
        ylabel="Predicted Scores",
        xlim=(targets.min() - 0.05, targets.max() + 0.05),
        ylim=(targets.min() - 0.05, targets.max() + 0.05),
        title=f"n={len(targets)}",
    )
    plt.legend(
        labels=[f"R^2={r2:.2f}, MSE={mse:.2f}, pearson={pearson_r:.2f}"],
        fontsize="x-small",
    )
    g.figure.set_size_inches(4, 4)
    sns.despine()
    return plt.gcf()
