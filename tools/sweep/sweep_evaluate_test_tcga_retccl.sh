#!/bin/bash

# Perform a hyperparameter sweep on train-val data of fold 0 of TCGA

# Currently, these are performed sequentially on the machine that it is run from. This may take a long time depending on the hardware. Is it bayesian, however, and will find good combinations of hyperparameters quickly. 

python ectil/train.py \
    experiment=ectil/train/tcga/run_internal_validation_tcga.yaml \
    test=False \
    datamodule.num_workers=0 \
    task_name=sweep \
    hparams_search=mil_sweep_regression_stil \
    trainer=cpu