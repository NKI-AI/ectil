from typing import Any, List, Optional

import numpy as np
import torch
from torch.utils.data.dataloader import default_collate

from ectil import utils

log = utils.get_pylogger(__name__)


class CustomCollator:
    def __init__(self):
        pass

    def __call__(self, batch: List):
        return default_collate(batch)


class VariableSizeCollator:
    def __init__(
        self,
        variable_size_keys: List[str] = ["x", "meta"],
        method: str = "list",
    ):
        """Allows collating objects of variable size. Since each WSI has varying number of tiles,
        we can not simply concat the object and place it in a single tensor.

        Args:
            variable_size_keys (Optional[List[str]], optional): _description_. Defaults to ["x", "meta"].
            output_type (str, optional): a list uses variable_size_keys and puts tensors in a list. This is currently the only available and used method.
        """
        self.variable_size_keys = variable_size_keys
        self.method = method

    def _get_value(self, sample: dict, keys: list):
        """Recursive sequential calling of keys for dict"""
        if isinstance(keys, str):
            keys = [keys]

        def recursive_get(current_dict, keys):
            if not keys:
                return current_dict
            if not isinstance(current_dict, dict):
                return None
            return recursive_get(current_dict.get(keys[0]), keys[1:])

        return recursive_get(sample, keys)

    def _list_call(self, batch: List):
        """
        Custom collate function designed for h5_datamodule that loads variable-size compressed WSIs and returns a
        list of tensors

        Transforms a list of dictionaries into a dictionary of (mostly) tensorized values except for those keys
        that are said to be of variable size.
        """
        custom_collated_object = {}

        for variable_size_key in self.variable_size_keys:
            try:
                elem = batch[0][variable_size_key]
            except KeyError as e:
                log.info(f"Key {variable_size_key} not found in batch")
                raise e
            if isinstance(elem, dict):
                custom_collated_object[variable_size_key] = {}
                dict_keys = elem.keys()
                for dict_key in dict_keys:
                    custom_collated_object[variable_size_key][dict_key] = [
                        default_collate(obj[variable_size_key][dict_key])
                        for obj in batch
                    ]
            else:
                custom_collated_object[variable_size_key] = [
                    torch.from_numpy(obj[variable_size_key]).unsqueeze(0)
                    for obj in batch
                ]
        default_collated_object = default_collate(
            [
                {k: v for k, v in obj.items() if k not in self.variable_size_keys}
                for obj in batch
            ]
        )

        return {**custom_collated_object, **default_collated_object}

    def _call_factory(self):
        """
        Factory for various collator functions. For now, only collate for a list of dicts returned from
        the dataloader is implemented.
        """
        return {
            "list": self._list_call,
        }[self.method]

    def __call__(self, batch: List):
        call_function = self._call_factory()
        out = call_function(batch)
        return out
