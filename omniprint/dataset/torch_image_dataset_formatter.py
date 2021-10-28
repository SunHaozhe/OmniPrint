"""
Transforms OmniPrint raw data to the format of torchvision.datasets.ImageFolder
* https://pytorch.org/vision/stable/datasets.html#torchvision.datasets.ImageFolder
"""

import os 
import glob 
import numpy as np 
import pandas as pd 
from PIL import Image
import shutil


def ImageFolder_format(dataset_name, raw_dataset_path, label_name, image_format="png", 
                       output_dir="datasets", is_multilingual=False):
    """
    ImageFolder

    https://pytorch.org/vision/stable/datasets.html#torchvision.datasets.ImageFolder

    root/dog/xxx.png
    root/dog/xxy.png
    root/dog/[...]/xxz.png

    root/cat/123.png
    root/cat/nsdf3.png
    root/cat/[...]/asd932_.png

    raw_dataset_path:
        For example, out/20201203_161302_297288 
    image_format:
        png, jpg, etc. 
    """ 
    if not os.path.exists(raw_dataset_path):
        raise Exception("Raw dataset not found, please check the path.")

    output_dir = os.path.join(output_dir, dataset_name)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir) 
    os.makedirs(output_dir) 
    
    
    df = _read_raw_labels_csv(raw_dataset_path, label_name, is_multilingual)
    labels_ = sorted(list(set(df.loc[:, label_name].tolist())), key=_to_int_if_possible)

    labels_complete = ["{}_{}".format(label_name, xx) for xx in labels_]
    label2destination = dict()
    for i, label_complete in enumerate(labels_complete):
        path = os.path.join(output_dir, label_complete)
        os.makedirs(path)
        label2destination[labels_[i]] = path
    
    if is_multilingual:
        search_pattern = os.path.join(raw_dataset_path, "*", "data", "*.{}".format(image_format)) 
    else:
        search_pattern = os.path.join(raw_dataset_path, "data", "*.{}".format(image_format))
    for path in sorted(glob.glob(search_pattern)):
        current_label = df.loc[df["image_name"] == os.path.basename(path), label_name].iloc[0]
        shutil.copy(path, label2destination[current_label]) 


def _read_raw_labels_csv(raw_dataset_path, label_name, is_multilingual):
    """
    raw_dataset_path: str
        If is_multilingual==True, this directory should contain only one level 
        of subdirectories, each subdirectory contains a data subdirectory and a 
        label subdirectory.
    """
    if is_multilingual:
        dfs = []
        for path in sorted(glob.glob(os.path.join(raw_dataset_path, "*", "label", "raw_labels.csv"))):
            dfs.append(pd.read_csv(path, sep="\t", encoding="utf-8").loc[:, ["image_name", label_name]])
        return pd.concat(dfs, ignore_index=True)
    else:
        df = pd.read_csv(os.path.join(raw_dataset_path, "label", "raw_labels.csv"), 
                         sep="\t", encoding="utf-8")
        return df.loc[:, ["image_name", label_name]]  



def _to_int_if_possible(x):
    if str(x).isdigit():
        return int(x)
    else:
        return x


if __name__ == "__main__":
    import time 
    import argparse
    
    t0 = time.time()

    # read command line arguments
    parser = argparse.ArgumentParser() 
    parser.add_argument("-n", "--dataset_name", type=str, default="hello_world_classification_dataset")
    parser.add_argument("-r", "--raw_dataset_path", type=str, default="../out/20201222_232645_378905")
    parser.add_argument("-l", "--label_name", type=str, default="unicode_code_point")
    parser.add_argument("--is_multilingual", action="store_true", default=False)
    parser.add_argument("--output_dir", type=str, default="OmniPrint_ImageFolder")
    args = parser.parse_args() 

    ImageFolder_format(dataset_name=args.dataset_name, 
                       raw_dataset_path=args.raw_dataset_path, 
                       label_name=args.label_name, 
                       output_dir=args.output_dir,
                       is_multilingual=args.is_multilingual)


    


    print("Done in {:.4f} s.".format(time.time() - t0))














