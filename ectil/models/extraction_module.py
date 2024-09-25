import pathlib
from typing import Any, Dict, List

import h5py
import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

from ectil.utils import get_pylogger
from ectil.utils.utils import flatten

log = get_pylogger(__name__)


class H5Writer:
    def __init__(self, h5_root_dir: str):
        """Base class for writing h5 files"""
        self.h5_root_dir = h5_root_dir
        self.output = {}

    def stack_output(self, output: List[Dict]) -> Dict:
        """Essentially a collate function
        The List[Dict] for e.g. a DLUP WSI dataset is
        [{
            'image': Tensor(batch_size * features),
            'coordinates: [Tensor(batch_size)] * 2,
            'mpp': Tensor(batch_size),
            'path': [str] * batch_size,
            'region_index': Tensor(batch_size)
            'grid_local_coordinates': [Tensor(batch_size)] * 2
            'grid_index': Tensor(batch_size)
        }
        ] * number_of_batches_in_dataloader

        And we return
        {
            'image': ndarray(len(dataset) * features),
            'coordinates: ndarray(len(dataset) * 2)
            'mpp': ndarray(len(dataset)),
            'path': [str] * len(dataset),
            'region_index': ndarray(len(dataset))
            'grid_local_coordinates': ndarray(len(dataset) * 2)
            'grid_index': ndarray(len(dataset)
        }
        """
        stacked_output = {}
        for key in output[0].keys():
            stacked_output[key] = [batch[key] for batch in output]
        for key, value in stacked_output.items():
            if isinstance(value[0], Tensor):
                stacked_output[key] = torch.cat(value, dim=0).cpu().numpy()
            elif isinstance(value[0], list):
                if isinstance(value[0][0], str):
                    stacked_output[key] = flatten(value)
                elif isinstance(value[0][0], torch.Tensor):
                    size = len(value[0])
                    stacked_output[key] = (
                        torch.cat(
                            [
                                torch.cat([val[idx] for val in value], dim=0).unsqueeze(
                                    dim=1
                                )
                                for idx in range(size)
                            ],
                            dim=1,
                        )
                        .cpu()
                        .numpy()
                    )
            else:
                raise ValueError(
                    f"The batch consists of a {type(value[0])}, "
                    f"while we only check for Tensor, or list of strings, or list of tensors."
                )

        return stacked_output

    def save_output(
        self, output: Dict, regions: List[tuple[int, int, int, int, float] | None]
    ) -> None:
        """Saves the output of the extractor to an h5 file.

        Args:
            output (Dict): the output as given by the lightningmodule
            regions (List[tuple[int, int, int, int, float]]): explicitly added from the dataset. Will give the regions
                that generated the dataset with (x,y,w,h,mpp)
        """
        self.output = output
        self.output["slide_id"] = self.get_slide_id(self.output["path"])
        self.output["case_id"] = self.get_case_id(self.output["path"])
        self.set_relative_tile_paths(self.output["path"])
        self.output["path"] = self.get_relative_slide_path(self.output["path"])
        out_dir = self.get_out_dir(self.output["path"])
        out_dir.mkdir(parents=True)
        h5_filepath = self.get_h5_filepath(out_dir, self.output["slide_id"])
        with h5py.File(h5_filepath, "w") as hf:
            log.info(f"Writing to {h5_filepath}")
            hf.create_dataset("features", data=self.output["image"])
            del self.output["image"]

            if regions is not None:
                hf.create_dataset("regions", data=regions)

            if "relative_tile_paths" in self.output.keys():
                hf.create_dataset(
                    "relative_tile_paths", data=self.output["relative_tile_paths"]
                )
                del self.output["relative_tile_paths"]

            for key in [
                "coordinates",
                "mpp",
                "region_index",
                "grid_local_coordinates",
                "grid_index",
            ]:
                hf.create_dataset(key, data=self.output[key])
                del self.output[key]

            # slide_id, case_id, path are left and put to attrs
            hf.attrs.update(self.output)

    def set_relative_tile_paths(self, filepaths: List[str]):
        raise NotImplementedError

    def get_slide_id(self, filepaths: List[str]):
        # Generally called with only a single slide
        # If there's no official slide id, we will instead just use the filename, like
        slide_ids = [str(pathlib.Path(filepath).stem) for filepath in filepaths]
        assert (
            len(set(slide_ids)) == 1
        ), f"Not one unique slide ID for all tiles in one slide: {slide_ids}"
        return slide_ids[0]

    def get_case_id(self, filepaths: List[str]):
        raise NotImplementedError

    def get_relative_slide_path(self, filepaths: List[str]):
        raise NotImplementedError

    @staticmethod
    def get_h5_filepath(out_dir, slide_id):
        return f"{out_dir / slide_id}.h5"

    def get_out_dir(self, relative_slide_path: str) -> pathlib.Path:
        return pathlib.Path(self.h5_root_dir) / relative_slide_path

    def remove_existing_h5(self, dataloaders: List[DataLoader]) -> List[DataLoader]:
        """Filter dataloaders for which an h5 exists, called from extract.py

        Args:
            dataloaders (List[DataLoader]): list of all dataloaders of all WSIs

        Returns:
            List[DataLoader]: filtered list of dataloaders with dataloaders removed for which an h5 already exists
        """
        filtered_dataloaders = []
        for dataloader in (pbar := tqdm(dataloaders)):
            pbar.set_description(f"Checking if {dataloader} is already done")
            file_path = str(dataloader.dataset._path)
            slide_id = self.get_slide_id([file_path])
            relative_slide_path = self.get_relative_slide_path([file_path])
            out_dir = self.get_out_dir(relative_slide_path)
            h5_filepath = self.get_h5_filepath(out_dir, slide_id)
            if pathlib.Path(h5_filepath).exists():
                log.info(f"{h5_filepath} already exists. Skipping this dataloader")
                continue
            else:
                filtered_dataloaders.append(dataloader)
        return filtered_dataloaders


class TCGADLUPH5Writer(H5Writer):
    def __init__(self, *args, **kwargs):
        """
        Specific writer for the DLUP dataset for TCGA. Implements specific functions to
        get the slide and case id from the filename structure of TCGA
        """
        super().__init__(*args, **kwargs)

    def get_case_id(self, filepaths: List[str]) -> str:
        # e.g. TCGA-AO-A03R
        case_ids = [str(pathlib.Path(filepath).stem[:12]) for filepath in filepaths]
        assert (
            len(set(case_ids)) == 1
        ), f"Not one unique case ID for all tiles in one slide: {case_ids}"
        return case_ids[0]

    def get_relative_slide_path(self, filepaths: List[str]) -> str:
        slide_paths = [
            str(
                pathlib.Path(filepath).relative_to(pathlib.Path(filepath).parent.parent)
            )
            for filepath in filepaths
        ]
        assert (
            len(set(slide_paths)) == 1
        ), f"Not one unique slide path for all tiles in one slide: {slide_paths}"
        return slide_paths[0]

    def set_relative_tile_paths(self, filepaths: List[str]):
        # We do not have tile paths for DLUP WSIs.
        pass


class ExtractionModule(LightningModule):
    def __init__(self, encoder: Module, h5_writer: H5Writer):
        """
        Parameters
        encoder: Module
            a torch module with .forward() that maps (batch_size * channels * height * width) -> (batch_size * features)
        h5_writer: H5Writer
            a class that aggregates all batch outputs into a single object and writes it to an h5 file. This writer
            may differ for varying dataset classes, since they mey have different metadata that needs to be saved.
        """
        super().__init__()
        self.encoder = encoder
        self.h5_writer = h5_writer
        self.slide_features = []

    def reset_features_in_memory(self):
        self.slide_features = []

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Dict:
        """
        Transforms image to feature vector and returns the entire batch with vectors instead of images
        """
        batch["image"] = self.encoder(batch["image"])
        return batch

    def on_predict_batch_end(
        self, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        self.store_features_in_memory(outputs)
        # Below seems hacky but is done since we have many dataloaders and wish to do this at the end of each dataloader, not `on_predict_end`
        if batch_idx + 1 == len(self.trainer.predict_dataloaders[dataloader_idx]):
            current_dataset = self.trainer.predict_dataloaders[dataloader_idx].dataset
            self.h5_writer.save_output(
                output=self.h5_writer.stack_output(output=self.slide_features),
                regions=[
                    region
                    for idx, region in enumerate(current_dataset.regions)
                    if idx in set(current_dataset.masked_indices)
                ],
            )
            self.reset_features_in_memory()

    def store_features_in_memory(self, outputs=Any):
        self.slide_features.append(outputs)
