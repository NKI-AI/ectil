from typing import List

from pytorch_lightning import Callback
from torch import Tensor

from ectil.utils.utils import flatten


class ActOnEvaluationMetricsCallback(Callback):
    def __init__(
        self,
        metrics_of_interest: List[
            str
        ],  # These should match the attribute name in trainer.model
    ):
        super().__init__()
        self.metrics = {
            metric_of_interest: [] for metric_of_interest in metrics_of_interest
        }

    def append_metrics(self, trainer):
        for metric_of_interest in self.metrics.keys():
            self.metrics[metric_of_interest].append(
                trainer.model.__dict__["_modules"][metric_of_interest].compute().item()
            )

    def on_validation_epoch_end(self, trainer, *args, **kwargs) -> None:
        self.append_metrics(trainer)

    def on_test_epoch_end(self, trainer, *args, **kwargs) -> None:
        self.append_metrics(trainer)

    def on_predict_epoch_end(self, trainer, *args, **kwargs) -> None:
        self.append_metrics(trainer)


class ActOnEvaluationOutputCallback(Callback):
    def __init__(self, monitor="val/mse"):
        """Do stuff with predictions

        Args:
            task (str, optional), either of 'binary', 'multiclass': Influences how the 'preds' are handled. Defaults to 'binary', which will only save the probability of class 1. For multiclass, the entire list of probabilities will be saved.

            monitor will take care, at least for slidelevle predictions, that
            it will only be saved for the best model.
        """
        super().__init__()
        self.monitor = monitor
        self.outputs = []
        self.size = []

    def custom_function(self, *args, **kwargs):
        raise NotImplementedError

    def append_outputs(self, outputs):
        self.outputs.append(outputs)

    def reset_outputs(self):
        self.outputs = []

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self.append_outputs(outputs)

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self.append_outputs(outputs)

    def on_predict_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self.append_outputs(outputs)

    def on_validation_epoch_start(self, *args, **kwargs):
        self.reset_outputs()

    def on_test_epoch_start(self, *args, **kwargs):
        self.reset_outputs()

    def on_predict_epoch_start(self, *args, **kwargs):
        self.reset_outputs()

    def on_validation_end(self, *args, **kwargs):
        self.custom_function(*args, **kwargs)

    def on_test_end(self, *args, **kwargs):
        self.custom_function(*args, **kwargs)

    def on_predict_end(self, *args, **kwargs):
        self.custom_function(*args, **kwargs)

    @property
    def preds(self):
        return self._preds

    @property
    def _preds(self):
        return Tensor(
            flatten([batch_outputs["preds"].tolist() for batch_outputs in self.outputs])
        )

    @property
    def targets(self) -> Tensor:
        return Tensor(flatten([output["targets"].tolist() for output in self.outputs]))
