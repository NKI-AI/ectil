# The txt file for the paths for pure evaluation should be a csv with
# a `paths` column that holds relative filepaths to the h5s of interest with respect to the `datamodule.root_dir`
# datamodule.root_dir for this script is set to be `paths.h5_dump_dir`.
# E.g. go to `paths.h5_dump_dir` and find all relative paths to `h5s` with `find -type f -name "*.h5"`
# this may be a file with only 1 h5 path, or many. 

# output is logged in the standard log directory

# File with relative paths wrt datamodule.root_dir has to be set here (datamodule.test_paths)
python ectil/eval.py \
    ckpt_path=model_zoo/ectil/tcga/fold_0/epoch_065_step_858_weights_only.ckpt \
    trainer=cpu \
    datamodule.num_workers=0 \
    datamodule.test_paths=/home/y.schirris/ectil/tmp/test_paths.txt
    