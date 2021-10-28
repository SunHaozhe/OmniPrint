"""
Inspired by 
https://github.com/dragen1860/MAML-Pytorch/blob/master/omniglotNShot.py
https://github.com/dragen1860/MAML-Pytorch/blob/master/omniglot.py 
"""

import os
import glob
import numpy as np
import pandas as pd
from PIL import Image 
import torch



class MultilingualDataset(torch.utils.data.Dataset):
    """
    data_path: str
        e.g. ../out/multilingual/20210502_172424_943937
        Under data_path, there should be several directories, each of which 
        corresponds to one alphabet. In each alphabet directory, one should have 
        two subdirectories: data and label. The subdirectory data contains the images 
        e.g. arabic_0.png. The subdirectory label contains a single file raw_labels.csv.
    transform: how to transform the input (X)
    target_transform: how to transform the target variable (y)
    label: str
        choose one column from raw_labels.csv to be used as the target
    image_extension: str
        png, jpeg, etc. 
    """
    def __init__(self, data_path, transform=None, target_transform=None, 
                 label="unicode_code_point", image_extension=".png"):
        super().__init__()
        self.data_path = data_path
        self.check_path_exists()
        self.name = _get_dataset_name(self.data_path)

        self.transform = transform 
        self.target_transform = target_transform
        self.label = label

        self.image_extension = image_extension
        if not self.image_extension.startswith("."):
            self.image_extension = "." + self.image_extension
        
        self.construct_items()
    
    def check_path_exists(self):
        if not os.path.exists(self.data_path):
            raise Exception("Directory not found: {}".format(self.data_path))

    def consume_raw_labels_csv(self):
        dfs = dict()

        search_pattern = os.path.join(self.data_path, "*", "label", "raw_labels.csv")
        for path in sorted(glob.glob(search_pattern)):
            alphabet_name = path.split(os.sep)[-3]
            df = pd.read_csv(path, sep="\t", encoding="utf-8")
            dfs[alphabet_name] = df.loc[:, ["image_name", self.label]]
        
        return dfs

    def construct_items(self):
        """
        Builds self.items, self.idx2raw_label
        
        item: tuple
            (path, raw_label, label/target)

        idx2raw_label: dict 
            unique ID of raw_label ==> the value of raw_label
        """
        # dict, raw labels for each alphabet
        dfs = self.consume_raw_labels_csv()

        self.items = []

        search_pattern = os.path.join(self.data_path, "*", "data", "*" + self.image_extension)
        for path in sorted(glob.glob(search_pattern)):
            file_name = os.path.basename(path)
            alphabet_name = path.split(os.sep)[-3]
            df = dfs[alphabet_name]
            raw_label = df.loc[df["image_name"] == file_name, self.label].iloc[0]
            self.items.append([path, raw_label])

        idx = dict() 
        for item in self.items:
            if item[1] not in idx:
                idx[item[1]] = len(idx)
            item.append(idx[item[1]])

        self.idx2raw_label = {v: k for k, v in idx.items()} 

    def __getitem__(self, index):
        img_path = self.items[index][0]
        target = self.items[index][2]

        if self.transform is not None:
            img = self.transform(img_path) 
        else:
            img = Image.open(img_path).convert("RGB")
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.items)


def _get_dataset_name(path):
    return path.split(os.sep)[-1]


class NonEpisodicDataset(torch.utils.data.Dataset):
    """
    expects np.array input, e.g. (900, 20, 3, 28, 28)
    """
    def __init__(self, data, device):
        super().__init__()
        self.device = device
        self.n_classes = data.shape[0]
        self.y = np.repeat(range(self.n_classes), data.shape[1])
        self.y = torch.from_numpy(self.y).to(self.device)
        X = []
        for c in range(self.n_classes):
            X.append(data[c, :, :, :, :])
        self.X = torch.from_numpy(np.concatenate(X, axis=0)).to(self.device)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        """
        X: (C, H, W) image
        y: int
        """
        return self.X[index, :, :, :], self.y[index]


class ZMultilingualDataset(MultilingualDataset):
    """
    z_metadata: a python list of str
        each str corresponds to a column name in the raw_labels.csv
    """
    def __init__(self, data_path, transform=None, target_transform=None, 
                 label="unicode_code_point", z_metadata=None, image_extension=".png"):
        assert isinstance(z_metadata, list)
        self.z_metadata = z_metadata
        super().__init__(data_path, transform, target_transform, label, image_extension)
    

    def consume_raw_labels_csv(self):
        dfs = dict()

        search_pattern = os.path.join(self.data_path, "*", "label", "raw_labels.csv")
        for path in sorted(glob.glob(search_pattern)):
            alphabet_name = path.split(os.sep)[-3]
            df = pd.read_csv(path, sep="\t", encoding="utf-8")
            dfs[alphabet_name] = df.loc[:, ["image_name", self.label] + self.z_metadata]
        
        return dfs

    def construct_items(self):
        """
        Builds self.items, self.idx2raw_label
        
        item: tuple
            (path, raw_label, metadata, label/target)

        idx2raw_label: dict 
            unique ID of raw_label ==> the value of raw_label
        """
        # dict, raw labels for each alphabet
        dfs = self.consume_raw_labels_csv()

        self.items = []

        search_pattern = os.path.join(self.data_path, "*", "data", "*" + self.image_extension)
        for path in sorted(glob.glob(search_pattern)):
            file_name = os.path.basename(path)
            alphabet_name = path.split(os.sep)[-3]
            df = dfs[alphabet_name]
            raw_label = df.loc[df["image_name"] == file_name, self.label].iloc[0]
            metadata = df.loc[df["image_name"] == file_name, self.z_metadata].iloc[0].tolist()
            self.items.append([path, raw_label, metadata])

        idx = dict() 
        for item in self.items:
            if item[1] not in idx:
                idx[item[1]] = len(idx)
            item.append(idx[item[1]])

        self.idx2raw_label = {v: k for k, v in idx.items()} 


    def __getitem__(self, index):
        img_path = self.items[index][0]
        metadata = self.items[index][2]
        target = self.items[index][3]

        if self.transform is not None:
            img = self.transform(img_path) 
        else:
            img = Image.open(img_path).convert("RGB")
        if self.target_transform is not None:
            target = self.target_transform(target)

        return (img, metadata), target



