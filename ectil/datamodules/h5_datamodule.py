from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from ectil import utils
from ectil.datamodules.components.h5_features import (
    CustomTargetTransform,
    EctilH5Dataset,
    EctilH5DatasetEval,
)

log = utils.get_pylogger(__name__)


class EctilH5DataModule(LightningDataModule):

    def __init__(
        self,
        root_dir: str,  # Where to look for files
        train_paths: Optional[
            str
        ],  # csv with paths,${label_column} columns with relative filepaths and identifier to match the clini file
        val_paths: Optional[str],
        test_paths: Optional[str],
        clini_file: str,  # clini file with `patient_column` and `label_column`
        patient_column: Optional[str],  # identifier name for the patient label
        label_column: Optional[str | list],  # column that holds the label in clini file
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        store_dataset_in_memory: bool = False,  # whether or not to store entire dataset in memory
        target_transform: Optional[
            CustomTargetTransform
        ] = None,  # function to transform the label value
        collate_fn: Callable = default_collate,  # defaulte collate function for samples of varying size
        h5_type: str = "ectil",
        **kwargs,
    ):
        super().__init__()
        self.h5: Dataset = self._h5_type_factory(h5_type)
        self.kwargs = kwargs
        self.root_dir = root_dir

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        if store_dataset_in_memory:
            log.warning(
                "store_dataset_in_memory is set to True, which can lead to memory overload."
            )

        self.data_train = None  # Dataset
        self.data_val = None  # Dataset
        self.data_test = None  # Dataset

    def _h5_type_factory(self, h5_type: str) -> Dataset:
        """Factory for datasets

        Args:
            h5_type (str): For now, only 'ectil' is allowed

        Returns:
            Dataset: Dataset class
        """
        if h5_type == "ectil":
            return EctilH5Dataset
        elif h5_type == "ectil_eval":
            return EctilH5DatasetEval
        else:
            raise ValueError(f"Invalid h5_type: {h5_type}. Only 'ectil' is allowed.")

    def _setup_h5_dataset(self, paths):
        return (
            self.h5(
                input_path=paths,
                root_dir=self.root_dir,
                clini_file=self.hparams.clini_file,
                patient_column=self.hparams.patient_column,
                label_column=self.hparams.label_column,
                store_dataset_in_memory=self.hparams.store_dataset_in_memory,
                target_transform=self.hparams.target_transform,
                **self.kwargs,
            )
            if paths is not None and paths != ""
            else None
        )

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = self._setup_h5_dataset(self.hparams.train_paths)
            self.data_val = self._setup_h5_dataset(self.hparams.val_paths)
            self.data_test = self._setup_h5_dataset(self.hparams.test_paths)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=self.hparams.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.hparams.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.hparams.collate_fn,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "h5.yaml")
    cfg.data_dir = str(root / "data")
    _ = hydra.utils.instantiate(cfg)
