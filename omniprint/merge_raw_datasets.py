"""
This script assumes that each image instance has a unique name. 

If the datasets were synthesized sequentially at different timestamps, 
there will be no problems.

Otherwise, if the datasets were synthesized in parallel, 
one should verify that the timestamp (dataset ID) for each 
dataset is different.

# for datasets (e.g. one multilingual datasets)
python3 merge_raw_datasets.py --input_dir out --output_path merged_raw_dataset

# for datasets of datasets (e.g. several multilingual datasets)
python3 merge_raw_datasets.py --input_dir out --output_path merged_raw_dataset --recursive
"""

import argparse 
import os
import time
import glob
import pandas as pd
import shutil


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Merge multiple raw datasets into one single raw dataset")
    parser.add_argument("--input_dir", type=str, default="out", help="""The directory which 
        contains all the raw datasets to be merged.""")
    parser.add_argument("--output_path", type=str, default="merged_raw_dataset", help="""
        The path of the merged dataset. WARNING: If this path already exists, it will be deleted.""")
    parser.add_argument("--only_copy", action="store_true", default=False, help="""
        If True, then image instances will be copied. By default (False), image instances will be moved, 
        which is more efficient.""")
    parser.add_argument("--extension", type=str, default="png", help="""This must match the image extension, 
        otherwise no images will be moved""")
    parser.add_argument("--recursive", action="store_true", default=False, help="""Whether recursively search 
        raw datasets, this is useful if input_dir contains datasets of datasets.""")
    args = parser.parse_args()

    t0 = time.time()

    assert os.path.exists(args.input_dir), "Input directory not found!"

    # create (new) output directory
    if os.path.exists(args.output_path):
        shutil.rmtree(args.output_path) 
    os.makedirs(os.path.join(args.output_path, "data"))
    os.makedirs(os.path.join(args.output_path, "label"))

    if args.recursive:
        pattern = os.path.join(args.input_dir, "**", "label", "raw_labels.csv") 
        paths = sorted(glob.glob(pattern, recursive=True))
    else:
        pattern = os.path.join(args.input_dir, "*", "label", "raw_labels.csv")
        paths = sorted(glob.glob(pattern))

    dfs = []
    for path in paths:
        dfs.append(pd.read_csv(path, sep="\t", encoding="utf-8", index_col=0))
    df = pd.concat(dfs, ignore_index=True)

    df.to_csv(os.path.join(args.output_path, "label", "raw_labels.csv"), sep="\t", encoding="utf-8")

    if args.recursive:
        pattern = os.path.join(args.input_dir, "**", "data", "*.{}".format(args.extension)) 
        paths = sorted(glob.glob(pattern, recursive=True))
    else:
        pattern = os.path.join(args.input_dir, "*", "data", "*.{}".format(args.extension))
        paths = sorted(glob.glob(pattern))

    if args.only_copy:
        for path in paths:
            file_name = os.path.basename(path)
            shutil.copy(path, os.path.join(args.output_path, "data", file_name)) 
    else:
        for path in paths:
            file_name = os.path.basename(path)
            shutil.move(path, os.path.join(args.output_path, "data", file_name))
        shutil.rmtree(os.path.join(args.input_dir)) 

    print("Done in {:.2f} s.".format(time.time() - t0))









