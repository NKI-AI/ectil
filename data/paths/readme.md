These paths are generated with a simple python script that searches for the WSIs related to the `T-number`, and places the relative paths for each train/val/test fold in a separate file with an identifier that matches it to the clini file that holds the label of interest.

The `.csv` files should have a `paths` column with the relative path with respect to the `datamodule.root_dir`.

The other column should be a column with an identifier that is also available in the clini file which holds the label of interest. In our case, this is `T-number`. 

The absolute paths to these files are then given to `datamodule.train_paths`, `datamodule.val_paths`, and `datamodule.test_paths`, which are used to generate the train, val, and test datasets for each fold.

