from typing import Any, List

import hydra
import torch
from pytorch_lightning import LightningModule
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torchmetrics import MaxMetric, MeanMetric, MinMetric
from torchmetrics.regression.explained_variance import ExplainedVariance
from torchmetrics.regression.mse import MeanSquaredError
from torchmetrics.regression.pearson import PearsonCorrCoef

from ectil import utils

log = utils.get_pylogger(__name__)


class H5LitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: Optimizer,
        criterion: _Loss,
        **kwargs,
    ):
        super().__init__()
        """Main base class to train a neural network.

        Set the net through hydra configs, e.g. in model/your_config.yaml.
        See, e.g., configs.model.net.h5_mean_mil
        """
        self.net = net
        self.criterion = criterion  # loss function

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])

    def on_train_epoch_end(self, *args, **kwargs):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def forward(self, x: torch.Tensor, **kwargs):
        """kwargs may contain e.g. region, which may be used in the attention using the distance_weighted_mean"""
        return self.net(x, **kwargs)

    def on_test_epoch_end(self, *args, **kwargs):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """

        return {
            "optimizer": self.hparams.optimizer(
                params=[params for (name, params) in self.named_parameters()]
            )
        }


class H5LitRegressionModule(H5LitModule):
    def __init__(self, *args, **kwargs):
        """Main class to train a regression model.
        Logged metrics are hardcoded for now.
        """
        super().__init__(*args, **kwargs)

        # metric objects for calculating and averaging accuracy across batches
        self.train_mse = MeanSquaredError()
        self.val_mse = MeanSquaredError()
        self.test_mse = MeanSquaredError()

        self.train_explained_variance = ExplainedVariance()
        self.val_explained_variance = ExplainedVariance()
        self.test_explained_variance = ExplainedVariance()

        self.train_pearson_r = PearsonCorrCoef()
        self.val_pearson_r = PearsonCorrCoef()
        self.test_pearson_r = PearsonCorrCoef()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation mse/r/explained_variance
        self.val_mse_best = MinMetric()
        self.val_pearson_r_best = MaxMetric()
        self.val_explained_variance_best = MaxMetric()

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_mse_best.reset()
        self.val_pearson_r_best.reset()
        self.val_explained_variance_best.reset()

    def step(self, batch: Any):
        # Expects a batch from EctilH5DataModule.
        # Since y may be a float64, we cast it to float() which returns a float32, required by backprop

        # Due to the varying input size, we return a custom-collated object that
        # returns a list of inputs instead of a tensor with stacked inputs.
        if isinstance(batch["x"], list):
            preds = []
            target = []
            model_meta = []
            h5_paths = []
            regions = []
            # We expect batch['x'] to be a [batch_size * torch.Tensor(1 * bag_size * feature_size)]
            for x, y, h5_path, region in zip(
                batch["x"], batch["y"], batch["h5_path"], batch["meta"]["regions"]
            ):
                model_out = self.forward(
                    x
                )  # The custom collate doesn't add the batch dimension.
                preds.append(model_out["out"].squeeze())
                # An aggregated batch gives us [batch_size * 1], wheres the target is [batch_size], so we flatten()
                model_meta.append(model_out["meta"])
                target.append(y.float())
                h5_paths.append(h5_path)
                regions.append(region)
            preds = torch.stack(preds)
            target = torch.stack(target)
        elif isinstance(batch["x"], torch.Tensor):
            # This may be achieved with another custom collate that subsamples
            # or appends to make each input similar size.
            model_out = self.forward(batch["x"])
            preds = model_out["out"].flatten()
            target = batch["y"].float()
            model_meta = model_out["meta"]
            h5_paths = batch["h5_path"]
            regions = batch["regions"]
        else:
            raise ValueError(
                f"the batch of input images is {type(batch['x'])}, we can handle a list of variable size"
                f"samples as given by VariableSizeCollator (a list with tensors of shape "
                f"1*bag_size*feature_dim), "
                f"or a Tensor of similarly sized samples with size batch_size*bag_size*feature_dim ]"
            )

        loss = self.criterion(preds, target)
        return loss, target, preds, model_meta, h5_paths, regions

    def training_step(self, batch: Any, batch_idx: int):
        loss, targets, preds, model_meta, h5_paths, regions = self.step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_mse(preds, targets)
        self.train_explained_variance(preds, targets)
        self.train_pearson_r(preds, targets)
        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "train/mse", self.train_mse, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "train/explained_variance",
            self.train_explained_variance,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/pearson_r",
            self.train_pearson_r,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return {
            "loss": loss,
            "preds": preds,
            "targets": targets,
            "model_meta": model_meta,
            "h5_paths": h5_paths,
            "regions": regions,
        }

    def validation_step(self, batch: Any, batch_idx: int):
        loss, targets, preds, model_meta, h5_paths, regions = self.step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_mse(preds, targets)
        self.val_explained_variance(preds, targets)
        self.val_pearson_r(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/mse", self.val_mse, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val/explained_variance",
            self.val_explained_variance,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/pearson_r",
            self.val_pearson_r,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return {
            "loss": loss,
            "preds": preds,
            "targets": targets,
            "model_meta": model_meta,
            "h5_paths": h5_paths,
            "regions": regions,
        }

    def validation_epoch_end(self, outputs: List[Any]):
        mse = self.val_mse.compute()  # get current val mse
        self.val_mse_best(mse)
        self.log("val/mse_best", self.val_mse_best.compute(), prog_bar=True)

        explained_variance = (
            self.val_explained_variance.compute()
        )  # get current val explained variance
        self.val_explained_variance_best(explained_variance)
        self.log(
            "val/explained_variance_best",
            self.val_explained_variance_best.compute(),
            prog_bar=True,
        )

        pearson_r = self.val_pearson_r.compute()  # get current val pearson r
        self.val_pearson_r_best(pearson_r)
        self.log("val/pearson_r_best", self.val_pearson_r_best.compute(), prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, targets, preds, model_meta, h5_paths, regions = self.step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_mse(preds, targets)
        self.test_explained_variance(preds, targets)
        self.test_pearson_r(preds, targets)
        self.log(
            "test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("test/mse", self.test_mse, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "test/explained_variance",
            self.test_explained_variance,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test/pearson_r",
            self.test_pearson_r,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return {
            "loss": loss,
            "preds": preds,
            "targets": targets,
            "model_meta": model_meta,
            "h5_paths": h5_paths,
            "regions": regions,
        }


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "h5_regression.yaml")
    _ = hydra.utils.instantiate(cfg)
