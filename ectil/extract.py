import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

# ------------------------------------------------------------------------------------ #
# `pyrootutils.setup_root(...)` is recommended at the top of each start file
# to make the environment more robust and consistent
#
# the line above searches for ".git" or "pyproject.toml" in present and parent dirs
# to determine the project root dir
#
# adds root dir to the PYTHONPATH (if `pythonpath=True`)
# so this file can be run from any place without installing project as a package
#
# sets PROJECT_ROOT environment variable which is used in "configs/paths/default.yaml"
# this makes all paths relative to the project root
#
# additionally loads environment variables from ".env" file (if `dotenv=True`)
#
# you can get away without using `pyrootutils.setup_root(...)` if you:
# 1. move this file to the project root dir or install project as a package
# 2. modify paths in "configs/paths/default.yaml" to not use PROJECT_ROOT
# 3. always run this file from the project root dir
#
# https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

from multiprocessing import set_start_method
from typing import List, Optional, Sequence, Tuple

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.loggers import Logger
from torch.utils.data import DataLoader

from ectil import utils

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def extract(cfg: DictConfig) -> Tuple[dict, dict]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        log.info(f"Setting global seed to {cfg.seed}")
        pl.seed_everything(cfg.seed, workers=True)

    # This needs to return a list of dataloaders, 1 for each WSI.
    # Should be able to make something etiher for WSIs or for TCGA-CRCk tiles. However this is best implemented.
    #
    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    wsi_dataloaders: Sequence[DataLoader] = hydra.utils.instantiate(cfg.datamodule)

    # This should have a .predict function that extracts features and metadata,
    # It stores the features and metadata in memory, I'd say.
    # and maybe on_predict_end or something saves everything
    # Also we should have a modelfactory that can return any model based on the type and path.
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    # Nothing to change...
    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    # Nothing to change?
    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    # Nothing ot change?
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": wsi_dataloaders,
        "model": model,
        "logger": logger,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    # Filter out those WSI loaders for which the h5 already exists
    wsi_dataloaders.dataloaders = model.h5_writer.remove_existing_h5(wsi_dataloaders)

    # This option allows to only do the preprocessing, which will compute the masks and set up and cache the datasets.
    # Thus we can easiliy run this step on a CPU node and only do the relatively quick extract step on a GPU node.
    if cfg.get("extract"):
        trainer.predict(
            model=model, dataloaders=wsi_dataloaders(), return_predictions=True
        )

    return {}, {}


@hydra.main(
    version_base="1.2", config_path=root / "configs", config_name="extract.yaml"
)
def main(cfg: DictConfig) -> Optional[float]:
    # extract the features
    extract(cfg)


if __name__ == "__main__":
    main()
