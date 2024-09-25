#!/bin/bash

# Perform foreground selection with FESI on all .svs WSIs from 
# TCGA_BRCA_IMAGES_ROOT as stated in the .env file.
# Extract patches with RetCCL and saves the features in .h5 format
# in the h5_dump_dir as defined in configs/paths/default.yaml
# If you want to only extract for specific WSIs within the directory,
# you can overwrite the image_glob with 
# +datamodule.image_paths_file which should contain absolute paths within
# the TCGA_BRCA_IMAGES_ROOT directory.

# To test this, you can, e.g., download a single WSI into ~/ectil/data/wsi, set `TCGA_BRCA_IMAGES_ROOT` in `.env` to `~/ectil/data/wsi`, and this will run the pipeline for a single WSI, saving it to `h5_dump_dir` defined in `paths/default.yaml`

# This will save a `.h5` file with the following structure, assuming `hf` is the loaded file
# hf.attrs contains `case_id: str`, `path: str`, `slide_id: str`
# hf has the following datasets: `coordinates` (n*2), `features` (n*num_features), `grid_index` (n*1), `grid_local_coordinates` (n*2), `mpp` (n*1), `region_index` (n*1), `regions` (n*5). All but the features are tile metadata that are defined in DLUP, and may be used to relate the feature to a specific image region. 

# num_workers, batch_size, and trainer may be increased and set to gpu 
# if better hardware is available. With 0 workers on a cpu this may take 20 minutes.
python ectil/extract.py \
    experiment=ectil/extract/tcga_retccl \
    datamodule.num_workers=0 \
    datamodule.batch_size=16 \
    trainer=cpu


# To run ECTIL on this slide, see `tools/infer/infer_tcga_retccl_external.sh`

    