"""
Dataloader that loads OmniPrint meta-learning dataset in the same way as 
the standard Omniglot approach 

Adapted from 
https://github.com/dragen1860/MAML-Pytorch/blob/master/omniglotNShot.py
https://github.com/dragen1860/MAML-Pytorch/blob/master/omniglot.py 
"""

import os
import glob
import collections
import numpy as np 
import pandas as pd 
import torch 
import torchvision
from PIL import Image 

from torch_datasets import MultilingualDataset, NonEpisodicDataset


class OmniglotLikeDataloader:
    """
    msplit: str of comma 3 separated integers
        Meta split.
        Number of alphabets (character sets) used for meta-train, meta-validation and meta-test. 
        The sum of the 3 integers must be equal to the number of alphabets, which means that no 
        overlapping is allowed.
    nb_batches_to_preload: int
        How many batches (of episodes/tasks) to preload. 
        One batch can consist of multiple episodes/tasks. 
    batch_size: int
        Meta batch size, the number of episodes/tasks in one (meta) batch. 
    n_way: int
    n_way_mtrain: 
        if None, n_way_mtrain = n_way
    k_support: int
        k for k-shot 
    k_query: int

    For Omniglot, Vinyals split considers the split of 1028/172/423 characters (1623 in total), 
        33/5/12 alphabets respectively (one alphabet is presented in both meta-train and meta-test).
    For OmniPrint, we consider the split of 900/149/360 characters (1409 in total),
        32/8/14 alphabets respectively (no overlapping).

    mtrain:
        1028 / 1623 = 0.633 (Omniglot, Vinyals split)
        900 / 1409  = 0.639 (OmniPrint)
    mtrain + mval:
        1200 / 1623 = 0.739 (Omniglot, Vinyals split)
        1049 / 1409 = 0.744 (OmniPrint)
    """
    def __init__(self, data_path, batch_size=32, n_way=5, k_support=5, k_query=15, 
                 image_size=32, device="cpu", n_way_mtrain=None, cache_dir="cache", 
                 msplit="32,8,14", nb_batches_to_preload=10):
        self.dataset_name = _get_dataset_name(data_path)
        cache_path = os.path.join(cache_dir, "{}.npy".format(self.dataset_name))
        self.device = device
        self.nb_batches_to_preload = nb_batches_to_preload
        self.batch_size = batch_size

        # meta split specification
        msplit = [int(xx) for xx in msplit.split(",")]
        assert len(msplit) == 3, "Wrong format of meta split, there must be 3 integers."
        assert sum(msplit) == max(list(_get_split_point.keys())), """Wrong format of meta 
        split, the sum of meta-train alphabets, meta-validation alphabets and meta-test 
        alphabets must be equal to the number of alphabets in total."""
        self.msplit = msplit

        # N-way specification
        self.n_way = n_way
        if n_way_mtrain is None:
            # different number of ways between meta-train and meta-test
            self.n_way_mtrain = n_way
        else:
            self.n_way_mtrain = n_way_mtrain

        self.k_support = k_support
        self.k_query = k_query
        self.image_size = image_size

        if not os.path.isfile(cache_path):
            transform = torchvision.transforms.Compose([lambda x: Image.open(x).convert("RGB"),
                                                        lambda x: x.resize((image_size, image_size)), 
                                                        lambda x: np.reshape(x, (image_size, image_size, 3)), 
                                                        lambda x: np.transpose(x, [2, 0, 1]), 
                                                        lambda x: x / 255])
            self.dataset = MultilingualDataset(data_path=data_path, transform=transform)

            # transform OmniglotLikeDataset to a np.array (self.dataset)
            label2imgs = collections.defaultdict(list)
            for img, target in self.dataset:
                label2imgs[target].append(img)
            self.dataset = []
            for target, imgs in label2imgs.items():
                self.dataset.append(imgs)
            # (1409, 20, 3, 28, 28) 
            # 1409 classes/characters, 20 shots (support + query) 
            self.dataset = np.array(self.dataset).astype(np.float32)

            # cache np.array (self.dataset) to disk 
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            np.save(cache_path, self.dataset)
            print("Write cached np.array dataset into {}".format(cache_path))
        else:
            self.dataset = np.load(cache_path)
            print("Load cached np.array dataset from {}".format(cache_path))

        assert self.dataset.shape[1] >= self.k_support + self.k_query 

        # split meta-train, meta-validation and meta-test 
        split_point1 = _get_split_point[self.msplit[0]]
        assert split_point1 < self.dataset.shape[0]
        split_point2 = _get_split_point[self.msplit[0] + self.msplit[1]]
        assert split_point2 < self.dataset.shape[0]
        # (900, 20, 3, 28, 28), (149, 20, 3, 28, 28), (360, 20, 3, 28, 28)
        mtrain = self.dataset[:split_point1]
        mval = self.dataset[split_point1:split_point2]
        mtest = self.dataset[split_point2:]
        
        self.datasets = {"train": mtrain, "val": mval, "test": mtest}

        # load the first epochs
        self.batches = {"train": self.load_batches(self.datasets["train"], self.n_way_mtrain), 
                        "val": self.load_batches(self.datasets["val"], self.n_way),
                        "test": self.load_batches(self.datasets["test"], self.n_way)}
        self.batch_cursor = {"train": 0, "val": 0, "test": 0}


    def load_batches(self, data, n_way):
        batches = []
        for _ in range(self.nb_batches_to_preload):
            # one batch of episodes/tasks
            x_spts, y_spts, x_qrys, y_qrys = [], [], [], []
            for i in range(self.batch_size):
                # one episode/task (one data point for meta-learning)
                x_spt, y_spt, x_qry, y_qry = [], [], [], []

                # sample which classes/characters to use to form the current episode/task
                selected_classes = np.random.choice(data.shape[0], size=n_way, replace=False)
                for j, current_class in enumerate(selected_classes):
                    # sample which images to use to form the current class of the current episode/task
                    # This is just a reshuffle operation when k_support + k_query == data.shape[1]
                    selected_img_indices = np.random.choice(data.shape[1], 
                        size=self.k_support+self.k_query, replace=False)

                    # support set of the current episode/task
                    # (k_support, 3, 28, 28)
                    support_img_indices = selected_img_indices[:self.k_support]
                    x_spt.append(data[current_class, support_img_indices, :, :, :])
                    y_spt.append([j for xx in range(self.k_support)])

                    # query set of the current episode/task
                    # (k_query, 3, 28, 28)
                    query_img_indices = selected_img_indices[self.k_support:self.k_support+self.k_query]
                    x_qry.append(data[current_class, query_img_indices, :, :, :])
                    y_qry.append([j for xx in range(self.k_query)])

                # Shuffle the order of the n_way * (k_support + k_query) instances within the 
                # current episode/task, which prevents the instances of the same class from being 
                # next to each other.

                ## shuffle support set of the current episode/task
                ## (n_way * k_support, 3, 28, 28)
                indices = list(range(n_way * self.k_support))
                np.random.shuffle(indices)
                x_spt = np.array(x_spt).reshape(n_way * self.k_support, 
                    3, self.image_size, self.image_size)[indices]
                y_spt = np.array(y_spt).reshape(n_way * self.k_support)[indices]

                ## shuffle query set of the current episode/task
                ## (n_way * k_query, 3, 28, 28)
                indices = list(range(n_way * self.k_query))
                np.random.shuffle(indices)
                x_qry = np.array(x_qry).reshape(n_way * self.k_query, 
                    3, self.image_size, self.image_size)[indices]
                y_qry = np.array(y_qry).reshape(n_way * self.k_query)[indices]

                # add the newly created episode/task to the current batch
                x_spts.append(x_spt)
                y_spts.append(y_spt)
                x_qrys.append(x_qry)
                y_qrys.append(y_qry)

            # (batch_size, n_way * k_support, 3, 28, 28)
            x_spts = np.array(x_spts).reshape(self.batch_size, n_way * self.k_support, 
                3, self.image_size, self.image_size)
            y_spts = np.array(y_spts).astype(int).reshape(self.batch_size, 
                n_way * self.k_support)

            # (batch_size, n_way * k_query, 3, 28, 28)
            x_qrys = np.array(x_qrys).reshape(self.batch_size, n_way * self.k_query, 
                3, self.image_size, self.image_size)
            y_qrys = np.array(y_qrys).astype(int).reshape(self.batch_size, 
                n_way * self.k_query)

            # np.array to PyTorch tensors 
            x_spts = torch.from_numpy(x_spts).to(self.device)
            y_spts = torch.from_numpy(y_spts).to(self.device)
            x_qrys = torch.from_numpy(x_qrys).to(self.device)
            y_qrys = torch.from_numpy(y_qrys).to(self.device)

            batches.append([x_spts, y_spts, x_qrys, y_qrys])
        return batches


    def next(self, mode="train"):
        """
        returns the next batch of episodes/tasks
        """
        # if the preloaded batches ran out, load the next batches 
        if self.batch_cursor[mode] >= len(self.batches[mode]):
            if mode == "train":
                n_way = self.n_way_mtrain
            else:
                n_way = self.n_way
            self.batches[mode] = self.load_batches(self.datasets[mode], n_way)
            self.batch_cursor[mode] = 0 

        next_batch = self.batches[mode][self.batch_cursor[mode]]
        self.batch_cursor[mode] += 1

        return next_batch

    def get_non_episodic_dataloader(self, mode, batch_size=32):
        """
        This is used for the approach which simply trains a classifier 
        over all of the training classes at once.
        
        mode: str
            "train", "val" or "test"
        batch_size: int
            This is the batch size of usual data points, 
            not batch size of meta-learning episodes/tasks
        
        returns a torch.utils.data.DataLoader and an integer representing the 
        number of classes
        """
        non_episodic_dataset = NonEpisodicDataset(self.datasets[mode], self.device)
        non_episodic_dataloader = torch.utils.data.DataLoader(non_episodic_dataset, 
            batch_size=batch_size, shuffle=True)
        n_classes = non_episodic_dataset.n_classes
        return non_episodic_dataloader, n_classes


def _get_dataset_name(path):
    return path.split(os.sep)[-1]


# _get_split_point is now legacy
# These numbers are computed from meta1, meta2, ..., meta5 (Omniglot-like datasets)
_get_split_point = {1: 29, 2: 68, 3: 106, 4: 116, 5: 156, 6: 166, 7: 180, 8: 206, 
                    9: 232, 10: 264, 11: 274, 12: 286, 13: 302, 14: 339, 15: 349, 
                    16: 366, 17: 386, 18: 472, 19: 521, 20: 555, 21: 565, 22: 579, 
                    23: 606, 24: 680, 25: 758, 26: 793, 27: 803, 28: 818, 29: 845, 
                    30: 855, 31: 890, 32: 900, 33: 933, 34: 943, 35: 953, 36: 963, 
                    37: 993, 38: 1027, 39: 1037, 40: 1049, 41: 1115, 42: 1125, 
                    43: 1166, 44: 1184, 45: 1207, 46: 1220, 47: 1232, 48: 1268, 
                    49: 1278, 50: 1292, 51: 1338, 52: 1348, 53: 1390, 54: 1410}




if __name__ == "__main__":
    dataloader = OmniglotLikeDataloader("../omniglot_like_datasets/meta1")

    """
    The number of batches per epoch in meta-train/meta-test equals the number 
    of classes/characters in meta-train/meta-test divided by the (meta) batch 
    size, keeping the integer part. 
    In this way, batch_size * nb_batches (= nb_episodes) \approx nb_classes. 
    If there were not random class sampling in episode formation process, 
    each class/character should be "traversed" n_way times in one epoch.
    """

    for epoch in range(3):
        nb_batches_per_epoch_mtrain = dataloader.datasets["train"].shape[0] // dataloader.batch_size
        for batch_idx in range(nb_batches_per_epoch_mtrain):
            # sample one batch of support and query images and labels within meta-train 
            x_spt, y_spt, x_qry, y_qry = dataloader.next("train")
            print(type(x_spt), x_spt.dtype)
            print(x_spt.shape, y_spt.shape, x_qry.shape, y_qry.shape)

        nb_batches_per_epoch_mval = dataloader.datasets["val"].shape[0] // dataloader.batch_size
        for batch_idx in range(nb_batches_per_epoch_mval):
            # sample one batch of support and query images and labels within meta-test
            x_spt, y_spt, x_qry, y_qry = dataloader.next("val")
            print(type(x_spt), x_spt.dtype)
            print(x_spt.shape, y_spt.shape, x_qry.shape, y_qry.shape)

        nb_batches_per_epoch_mtest = dataloader.datasets["test"].shape[0] // dataloader.batch_size
        for batch_idx in range(nb_batches_per_epoch_mtest):
            # sample one batch of support and query images and labels within meta-test
            x_spt, y_spt, x_qry, y_qry = dataloader.next("test")
            print(type(x_spt), x_spt.dtype)
            print(x_spt.shape, y_spt.shape, x_qry.shape, y_qry.shape)
    



