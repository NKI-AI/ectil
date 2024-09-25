# coding=utf-8
import copy
from abc import ABC
from pathlib import Path
from typing import Optional, Union

import h5py
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from ectil import utils

log = utils.get_pylogger(__name__)


class CustomTargetTransform(ABC):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(
        self, y: Union[int, float], dataset, *args, **kwargs
    ) -> Union[int, float]:
        raise NotImplementedError


class DivideTargetTransform(CustomTargetTransform):
    def __init__(self, divide_factor: int = 100):
        self.divide_factor = divide_factor

    def __call__(self, y, *args, **kwargs):
        return y / self.divide_factor


class EctilH5Dataset(Dataset):

    def __init__(
        self,
        input_path: Path,
        root_dir: str,
        clini_file: Path,
        patient_column: str,
        label_column: str,
        target_transform: Optional[CustomTargetTransform] = None,
        store_dataset_in_memory: bool = False,
    ):
        """Dataset class reading features from h5 files that are generated using the ectil.models.extraction_module.H5Writer

        Args:
            input_path (Path): absolute path to a .txt file with relative or absolute paths to the filenames for this dataset
            root_dir (str): root dir from which to read the relative paths to the filenames in input_path
            clini_file (Path): absolute path to a clini file with patient IDs
            patient_column (str): describing the column that holds the patient ID
            label_column (str): _description_
            target_transform (Optional[CustomTargetTransform], optional): Function to apply to transform the target. Defaults to None.
            store_dataset_in_memory (bool, optional): If set to True will save the entire dataset in memory that may speed up training if i/o is the bottleneck. Defaults to False.
        """

        self.paths = pd.read_csv(input_path)
        self.root_dir = root_dir
        self.clini_file = pd.read_csv(clini_file).set_index(patient_column)
        self.patient_column = patient_column
        self.label_column = label_column
        self.target_transform = target_transform
        self.store_dataset_in_memory = store_dataset_in_memory
        self.in_memory_dataset = {}

    def __len__(self):
        return len(self.paths)

    @staticmethod
    def read_file(hf: h5py.File):
        features = hf["features"][()]
        meta_dict = {k: v for k, v in hf.attrs.items()}
        for key in hf.keys():
            if key != "features":
                meta_dict[key] = hf[key][()]
        return {"x": features, "meta": meta_dict}

    def create_data_obj(
        self, hf: h5py.File, h5_path: str, target: Union[float, int], *args, **kwargs
    ):
        data_obj = self.read_file(hf)
        data_obj["y"] = target
        data_obj["h5_path"] = h5_path
        return data_obj

    def __getitem__(self, idx):
        case_id = self.paths.loc[idx, self.patient_column]
        slide_id = self.paths.loc[idx, "paths"]
        h5_path = self.paths.loc[idx, "paths"]

        target = float(self.clini_file.loc[case_id, self.label_column])

        if self.target_transform:
            target = self.target_transform(y=target, dataset=self)

        if idx in self.in_memory_dataset.keys() and self.store_dataset_in_memory:
            data_obj = self.in_memory_dataset[idx]
        else:
            hf = h5py.File(f"{self.root_dir}/{h5_path}", "r")
            data_obj = self.create_data_obj(
                hf=hf,
                h5_path=h5_path,
                case_id=case_id,
                slide_id=slide_id,
                target=target,
            )
            if self.store_dataset_in_memory:
                self.in_memory_dataset[idx] = copy.deepcopy(data_obj)
            hf.close()

        return data_obj


class EctilH5DatasetEval(EctilH5Dataset):
    def __init__(self, input_path: str, root_dir: str, *args, **kwargs):
        self.paths = pd.read_csv(input_path)
        self.root_dir = root_dir

    def __getitem__(self, idx):
        path = self.paths.loc[idx, "paths"]
        hf = h5py.File(f"{self.root_dir}/{path}", "r")
        data_obj = self.read_file(hf)
        data_obj["y"] = np.nan
        data_obj["h5_path"] = path
        hf.close()
        return data_obj

    def __len__(self):
        return len(self.paths)
