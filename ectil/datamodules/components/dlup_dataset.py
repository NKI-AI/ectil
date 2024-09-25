import argparse
import copy
import hashlib
import json
import pathlib
import pickle
from typing import Callable, Optional, Union

import dlup
import numpy as np
import pyvips
from dlup import SlideImage, UnsupportedSlideError
from dlup.annotations import WsiAnnotations
from dlup.data.dataset import TiledROIsSlideImageDataset as DLUPDataset
from dlup.tiling import GridOrder, TilingMode
from dlup.viz.plotting import plot_2d
from packaging import version
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from ectil.utils import pylogger
from ectil.utils.background import AvailableMaskFunctions
from ectil.utils.background import get_mask as dlup_get_mask

assert version.parse(dlup.__version__) >= version.parse(
    "0.3.23"
), f"At least dlup 0.3.23 is required, but {dlup.__version__} is installed"

log = pylogger.get_pylogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def transform_factory(
    transform_set: str = "imagenet_normalization",
) -> transforms.Compose:
    """Barebone transform factory that is only used for imagenet normalization which
    is the standard for many pre-trained models (including RetCCL)

    Args:
        transform_set (str): imagenet_normalization implements totensor() + imagenet mean/std normalization

    Raises:
        ValueError: If anything other than imagenet_normalization is given

    Returns:
        transforms.Compose: a Compose object that can be passed
    """

    if transform_set == "imagenet_normalization":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

        log.info(f"Using transformer_set '{transform_set}':  {transform}")
        return transform
    else:
        raise ValueError(
            f"requested transform_set={transform_set}, only 'imagenet_normalization' is currently available"
        )


def compute_mask(slide: SlideImage, mask_function: str) -> np.ndarray:
    return dlup_get_mask(slide=slide, mask_func=AvailableMaskFunctions[mask_function])


def cache_mask(mask_path: pathlib.Path, mask: np.ndarray) -> None:
    if not mask_path.parent.is_dir():
        mask_path.parent.mkdir(parents=True)
    Image.fromarray(mask).save(fp=mask_path)


def load_mask(mask_path: pathlib.Path) -> np.ndarray:
    img = Image.open(mask_path)
    try:
        mask = np.asarray(img, dtype="uint8")
    except SystemError:
        mask = np.asarray(img.getdata(), dtype="uint8")
    log.debug(f"Loaded {mask_path}")
    return mask


def get_thumbnail_mask_overlay_fp(mask_path: pathlib.Path):
    return mask_path.parent / f"{mask_path.stem}_overlay{mask_path.suffix}"


def save_overlay(mask_path: pathlib.Path, mask: np.ndarray, slide: SlideImage) -> None:
    """Save mask overlay on a thumbnail to show how well the chosen tissue mask looks

    Args:
        mask_path (pathlib.Path): path to the mask. will be used to set the name and path of the overlay
        mask (np.ndarray): boolean ndarray of the mask. This is relatively large so we want to resize it
        slide (SlideImage): the SlideImage object that the mask belongs to. Used to create a thumbnail.
    """
    current_mask_size = np.array(mask.shape)
    scaling = 512 / max(current_mask_size)
    new_mask_size = tuple((scaling * current_mask_size).astype(int))
    mask = np.array(
        Image.fromarray(mask).resize(
            new_mask_size[::-1], resample=Image.Resampling.NEAREST
        )
    )
    thumb = slide.get_thumbnail(
        size=mask.shape[::-1]
    )  # ndarray and PIL Image have inverted dimensions
    # These can be 1 pixel off.
    mask = mask[: thumb.size[1], : thumb.size[0]]
    thumb = thumb.crop(box=(0, 0, mask.shape[1], mask.shape[0]))

    overlay = plot_2d(image=thumb, mask=mask, mask_colors={1: "#ff0000"})
    overlay.save(fp=get_thumbnail_mask_overlay_fp(mask_path))


def get_mask(
    image_root_dir: pathlib.Path,
    mask_cache_dir: pathlib.Path,
    slide: pathlib.Path,
    mask_function: str,
) -> np.ndarray:
    slide_path = image_root_dir / slide
    mask_path = (
        mask_cache_dir / slide_path.relative_to("/") / mask_function / "mask.png"
    )
    if mask_path.is_file():
        mask = load_mask(mask_path=mask_path)
        if not get_thumbnail_mask_overlay_fp(mask_path).is_file():
            save_overlay(
                mask_path=mask_path,
                mask=mask,
                slide=SlideImage.from_file_path(slide_path),
            )
    else:
        slide = SlideImage.from_file_path(slide_path)
        log.debug(f"Computing mask for {slide}")
        mask = compute_mask(slide=slide, mask_function=mask_function)
        cache_mask(mask_path=mask_path, mask=mask)
        save_overlay(mask_path=mask_path, mask=mask, slide=slide)

    return mask, mask_path


def get_dataset_factory(
    mpp: float,
    tile_size: int,
    transform: Callable,
    limit_bounds: bool,
    dataset_cache_dir: Optional[pathlib.Path],
):
    def dataset_factory(
        slide_path: pathlib.Path,
        mask: Optional[Union[SlideImage, np.ndarray, WsiAnnotations]] = None,
        mask_file: Optional[pathlib.Path] = None,
    ) -> DLUPDatasetWrapper:

        dataset_args = {
            "path": slide_path,
            "mpp": mpp,
            "tile_size": (tile_size, tile_size),
            "tile_overlap": (0, 0),  # default value
            "tile_mode": TilingMode.skip,  # default value
            "grid_order": GridOrder.C,  # default value
            "crop": False,  # default value
            "transform": transform,
            "mask": mask,
            "mask_threshold": 0.1,  # default value
            "limit_bounds": limit_bounds,
        }
        saveable_dataset_args = copy.deepcopy(dataset_args)
        saveable_dataset_args["mask"] = str(
            mask_file
        )  # Don't want to save the mask itself, just the path
        if dataset_cache_dir is None:
            return DLUPDatasetWrapper.from_standard_tiling(**dataset_args)
        else:
            out_dir = dataset_cache_dir / slide_path.relative_to("/")
            if not out_dir.is_dir():
                out_dir.mkdir(parents=True)

            cleaned_args = {
                k: (
                    v.tolist()
                    if isinstance(v, np.ndarray)
                    else (
                        str(v)
                        if isinstance(v, pathlib.Path)
                        else (
                            str(v.transforms)
                            if isinstance(v, transforms.Compose)
                            else v
                        )
                    )
                )
                for k, v in saveable_dataset_args.items()
            }
            dataset_args_serialized = pickle.dumps(cleaned_args)
            # Hashing can't handle numpy arrays, so we convert them to lists
            dataset_args_hash = hashlib.sha256(dataset_args_serialized).hexdigest()

            dataset_pickle_path = out_dir / f"{dataset_args_hash}.pickle"
            dataset_args_path = out_dir / f"{dataset_args_hash}.json"

            if dataset_pickle_path.is_file():
                log.info(f"Loading pickled dataset from {dataset_pickle_path}")
                with open(dataset_pickle_path, "rb") as f:
                    dataset = pickle.load(f)
            else:
                dataset = DLUPDatasetWrapper.from_standard_tiling(**dataset_args)
                log.info(f"Pickling dataset to {dataset_pickle_path}")
                log.info(f"Saving dataset arguments to {dataset_args_path}")
                with open(dataset_pickle_path, "wb") as f:
                    pickle.dump(dataset, f)
                with open(dataset_args_path, "w") as f:
                    json.dump(cleaned_args, f)

            return dataset

    return dataset_factory


class DLUPDatasetWrapper(DLUPDataset):
    """Thin wrapper to return plain RGB images."""

    def __init__(self, *args, transform: Callable | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.transform = transform

    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        out["image"] = out["image"].convert("RGB")
        if self.transform:
            out["image"] = self.transform(out["image"])

        out["path"] = str(out["path"])
        return out


class DLUPDataLoaders:
    def __init__(
        self,
        image_root_dir: str,
        image_glob: str,
        mpp: float,
        tile_size: int,
        transform_set: str | None = None,
        image_paths_file: str | None = None,
        mask_cache_dir: pathlib.Path | None = None,
        mask_function: str | None = "fesi",
        num_workers: int = 0,
        batch_size: int = 32,
        limit_bounds: bool = True,  # dlup 0.3.23 can use the openslide.bounds-{} property
        backend: str = "PYVIPS",
    ):
        """Initialization of the data loader that loads WSIs using DLUP

        Args:
            image_root_dir (str): the root directory of the images to glob from
            image_glob (str): the glob pattern to use to find the images from image_root_dir. Will not be used if image_paths_file is provided
            mpp (float): the mpp to use for the patches
            tile_size (int): the size of the patches
            transform_set (str | None, optional): the transform set to use. Default to None. See transform_factory for options
            image_paths_file (str | None, optional): an absolute path to a file containing a list of relative image paths to the image_root_dir. Defaults to None. If provided, will disable the use of image_glob
            mask_cache_dir (Path | None, optional): the directory to cache masks to.
            mask_function (str | None, optional): Defaults to "fesi". See DLUP for options
            num_workers (int, optional): Defaults to 0.
            batch_size (int, optional): Defaults to 32.
        """

        use_mask = mask_cache_dir and mask_function
        if use_mask:
            log.info(
                f"Checking {mask_cache_dir}/{image_root_dir} for cached {mask_function} masks for entire dataset. "
                f"If not cached, masks will be computed"
            )
        else:
            log.warning(
                "Not using background masks for the data. This is likely very inefficient."
            )

        if image_paths_file:
            log.info(f"Using image paths file {image_paths_file}")
            with open(image_paths_file) as f:
                all_image_paths = [pathlib.Path(i.strip()) for i in f.readlines()]
        else:
            all_image_paths = [i for i in pathlib.Path(image_root_dir).glob(image_glob)]

        images = []
        for image_path in tqdm(
            all_image_paths,
            total=len(all_image_paths),
            desc="Checking slides that can be opened",
        ):
            relative_image_path = image_path.relative_to(image_root_dir)
            try:
                SlideImage.from_file_path(image_path, backend=backend)
                images.append(relative_image_path)
            except (UnsupportedSlideError, pyvips.error.Error) as e:
                log.warning(f"{image_path} skipped: {e}")

        transform = (
            transform_factory(transform_set)
            if transform_set
            else transforms.Compose([transforms.ToTensor()])
        )
        dataset_factory = get_dataset_factory(
            mpp=mpp,
            tile_size=tile_size,
            transform=transform,
            limit_bounds=limit_bounds,
            dataset_cache_dir=mask_cache_dir,
        )
        self.dataloaders = []
        for image in tqdm(
            images,
            desc="Getting/computing masks and creating dataloaders for each openable WSI",
        ):
            mask, mask_file = (
                get_mask(
                    image_root_dir=pathlib.Path(
                        image_root_dir
                    ).expanduser(),  # Allow giving ~ paths
                    mask_cache_dir=pathlib.Path(
                        mask_cache_dir
                    ).expanduser(),  # Allow giving ~ paths
                    slide=image,
                    mask_function=mask_function,
                )
                if use_mask
                else None
            )
            slide_dataset = dataset_factory(
                slide_path=pathlib.Path(image_root_dir) / image,
                mask=mask,
                mask_file=mask_file,
            )
            loader = DataLoader(
                dataset=slide_dataset, num_workers=num_workers, batch_size=batch_size
            )
            self.dataloaders.append(loader)

    def __iter__(self):
        return iter(self.dataloaders)

    def __call__(self):
        return self.dataloaders


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arguments for testing the DLUP Dataset. If the cache dir does not "
        "contain masks, they will be computed and cached."
    )

    parser.add_argument(
        "--image_root_dir",
        type=pathlib.Path,
        default="",
        help="Root dir that contains images. the will be also be used asa relative path to the cache"
        "root dir.",
    )
    parser.add_argument(
        "--image_glob",
        type=str,
        default="",
        help="Glob pattern used to find the WSIs. E.g. '*/*.svs' for TCGA.",
    )

    args = parser.parse_args()

    loaders = DLUPDataLoaders(
        image_root_dir=args.image_root_dir,
        image_glob=args.image_glob,
        mpp=0.5,
        tile_size=512,
        mask_cache_dir="~/wimildon/logs/preprocessing/masks",
        mask_function="fesi",
        num_workers=0,
        batch_size=16,
    )

    first_wsi = [wsi for wsi in loaders][0]

    batches = [tiles for tiles in first_wsi]
