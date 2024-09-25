from os import mkdir
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl

from ectil.callbacks import ActOnEvaluationOutputCallback
from ectil.utils.utils import flatten
from ectil.utils.visualisation import log_plot, plot_scatter


class LogTileLevelPrediction(ActOnEvaluationOutputCallback):
    def __init__(self, on_validation: bool = True):
        super().__init__()
        self.on_validation = on_validation

    def custom_function(self, trainer, pl_module):
        """
        Log the tile-level output (attention & score) for the validation or test set.

        Saves a .csv for each slide with the attention score (attention_weights), target score (tile_level_output),  the region (x,y,h,w) of the slide. It is saved under the relative h5_path which can be used to find the
        original h5 and original wsi for further visualization.

        The file is saved in log_dir / [val/test] / tile_level_output

        Only saves during validation if on_validation=True

        Only saves during validation if it is the best model so far, hence the saved output is only
        from the best model.
        """
        out_dir = (
            Path(trainer.log_dir)
            / "out"
            / f"{trainer.state.stage}"
            / f"tile_level_output"
        )
        out_dir.mkdir(exist_ok=True, parents=True)

        best_model_so_far = (
            trainer.testing
        )  # If we're testing, we're already using the best model. If we're in any other stage, we may not have the best model so far.
        if self.monitor is not None and not best_model_so_far:
            best_model_so_far = (
                trainer.callback_metrics[self.monitor]
                == trainer.callback_metrics[f"{self.monitor}_best"]
            )

        if self.monitor is None or best_model_so_far:
            for output in self.outputs:
                for region, model_meta, h5_path in zip(
                    output["regions"], output["model_meta"], output["h5_paths"]
                ):
                    tmp_df = pd.DataFrame(region, columns=["x", "y", "w", "h", "mpp"])
                    tmp_df["tile_level_output"] = model_meta[
                        "out_per_instance"
                    ].squeeze(
                        0
                    )  # the out_per_instance is 1 (batch) * variable(bag_size) * 1 (attention)
                    tmp_df["attention_weights"] = model_meta[
                        "attention_weights"
                    ].squeeze(0)
                    out_file = out_dir / h5_path / "tile_level_output.csv"
                    out_file.parent.mkdir(parents=True, exist_ok=True)
                    tmp_df.to_csv(out_file, index=False)

    def on_validation_end(self, *args, **kwargs):
        if self.on_validation:
            self.custom_function(*args, **kwargs)
        else:
            pass


class LogOutput(ActOnEvaluationOutputCallback):
    def __init__(self, on_validation: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.on_validation = on_validation

    def custom_function(self, trainer, pl_module):
        """
        Log the slide-level output for the validation or test set.

        Saves a .csv with the predictions, targets, path to the h5 file used, and the original filename.
        The file is saved in log_dir / [val/test] / slide_level_output

        Only saves during validation if on_validation=True

        Only saves during validation if it is the best model so far, hence the saved output is only
        from the best model.
        """
        # Log the output of the model for the val/test set
        preds = self.preds
        targets = self.targets
        h5_paths = flatten([batch_output["h5_paths"] for batch_output in self.outputs])
        filenames = [Path(h5_path).stem for h5_path in h5_paths]
        df = pd.DataFrame(
            {
                "preds": preds.tolist(),
                "targets": targets.tolist(),
                "h5_paths": h5_paths,
                "filenames": filenames,
            }
        )
        out_dir = (
            Path(trainer.log_dir)
            / "out"
            / f"{trainer.state.stage}"
            / "slide_level_output"
        )
        out_dir.mkdir(exist_ok=True, parents=True)

        best_model_so_far = (
            trainer.testing
        )  # If we're testing, we're already using the best model. If we're in any other stage, we may not have the best model so far.
        if self.monitor is not None and not best_model_so_far:
            best_model_so_far = (
                trainer.callback_metrics[self.monitor]
                == trainer.callback_metrics[f"{self.monitor}_best"]
            )

        if self.monitor is None or best_model_so_far:
            df.to_csv(out_dir / "slide-level-output.csv", index=False, header=True)

    def on_validation_end(self, *args, **kwargs):
        if self.on_validation:
            self.custom_function(*args, **kwargs)
        else:
            pass


class LogScatterPlot(ActOnEvaluationOutputCallback):
    def __init__(self):
        super().__init__()

    def custom_function(self, trainer, pl_module):
        """
        Plot a scatter plot with the current predictions for each epoch on tensorboard.
        """

        scatter_fig = plot_scatter(preds=self.preds, targets=self.targets)
        log_plot(
            fig=scatter_fig,
            trainer=trainer,
            tag=f"Scatter plot on {trainer.state.fn.value} (stage={trainer.state.stage.value}) set",
        )
