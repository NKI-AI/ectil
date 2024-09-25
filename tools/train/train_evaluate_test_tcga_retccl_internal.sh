#!/bin/bash

# Script to perform training, validation, and testing.

# This expect that features have been extracted with `extract_retccl_tcga_bc.sh` 
# on the slides for all patients in `data/clini/tcga_bc_folds.csv`, and that 
# H5_ROOT_DIR points to the dir where these are saved, with datamodule_train_paths, 
# datamodule_val_paths, and datamodule_test_paths in `configs/paths/default.yaml` 
# to be pointing to files that has a `paths` and 
# `{your_id_matching_to_id_in_clini_file}` column with a relative path to the h5, 
# and the identifier that is found in the clini file to retrieve labels. 

for i in 0 1 2 3 4; do
    python ectil/train.py \
        variables.fold=$i \
        experiment=ectil/train/tcga/run_internal_validation_tcga.yaml \
        datamodule.num_workers=0 \
        task_name=train_eval_test \
        trainer.max_epochs=1 \
        trainer=cpu
done