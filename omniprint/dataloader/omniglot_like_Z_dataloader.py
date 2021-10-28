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

from sklearn.neighbors import NearestNeighbors
import multiprocessing

from torch_datasets import MultilingualDataset, NonEpisodicDataset, ZMultilingualDataset
from omniglot_like_dataloader import OmniglotLikeDataloader, _get_dataset_name, _get_split_point


class OmniglotLikeZDataloader(OmniglotLikeDataloader):
    """
    Assuming that columns from z_metadata all contain continuous variables

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
        For OmniglotLikeZDataloader, it is recommended that k_support + k_query < the total number of images, 
        otherwise OmniglotLikeZDataloader will produce the same episodes while being less efficient. 
        
        According to small-scaled experiments, OmniglotLikeZDataloader consumes more than 6 times as much time 
        to generate episodes.

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

    n_jobs_knn is only used for K-nearst-neighbor search
    """
    def __init__(self, data_path, z_metadata, batch_size=32, n_way=5, k_support=5, k_query=5, image_size=32, device="cpu", 
                 n_way_mtrain=None, cache_dir="cache", msplit="32,8,14", nb_batches_to_preload=10, n_jobs_knn=None):
        self.dataset_name = _get_dataset_name(data_path)
        cache_path = os.path.join(cache_dir, "{}.npy".format(self.dataset_name))
        cache_metadata_path = os.path.join(cache_dir, "{}_metadata.npy".format(self.dataset_name))
        self.device = device
        self.nb_batches_to_preload = nb_batches_to_preload
        self.batch_size = batch_size

        if n_jobs_knn is None:
            self.n_jobs_knn = multiprocessing.cpu_count()
            print("Using {} processes for K-nearest-neighbor search.".format(self.n_jobs_knn))
        else:
            self.n_jobs_knn = n_jobs_knn
            print("Using {} processes for K-nearest-neighbor search.".format(self.n_jobs_knn))

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

        if not os.path.isfile(cache_metadata_path) or not os.path.isfile(cache_path):
            transform = torchvision.transforms.Compose([lambda x: Image.open(x).convert("RGB"),
                                                        lambda x: x.resize((image_size, image_size)), 
                                                        lambda x: np.reshape(x, (image_size, image_size, 3)), 
                                                        lambda x: np.transpose(x, [2, 0, 1]), 
                                                        lambda x: x / 255])
            self.dataset = ZMultilingualDataset(data_path=data_path, z_metadata=z_metadata, transform=transform)

            # transform OmniglotLikeDataset to a np.array (self.dataset)
            label2imgs = collections.defaultdict(list)
            label2metadata = collections.defaultdict(list)
            for (img, metadata), target in self.dataset:
                label2imgs[target].append(img)
                label2metadata[target].append(metadata)
            self.dataset = []
            self.metadata = []
            for target in label2imgs.keys():
                self.dataset.append(label2imgs[target])
                self.metadata.append(label2metadata[target])
            
            # (1409, 20, 3, 28, 28) 
            # 1409 classes/characters, 20 shots (support + query) 
            self.dataset = np.array(self.dataset).astype(np.float32)

            # (1409, 20, 2)
            self.metadata = np.array(self.metadata).astype(np.float32)

            # cache np.array (self.dataset) to disk 
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            np.save(cache_path, self.dataset)
            print("Write cached np.array dataset into {}".format(cache_path))

            # cache metadata np.array (self.metadata) to disk 
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            np.save(cache_metadata_path, self.metadata)
            print("Write cached np.array metadata into {}".format(cache_metadata_path))
        else:
            self.dataset = np.load(cache_path)
            print("Load cached np.array dataset from {}".format(cache_path))
            self.metadata = np.load(cache_metadata_path)
            print("Load cached np.array metadata from {}".format(cache_metadata_path))

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

        train_metadata = self.metadata[:split_point1]
        val_metadata = self.metadata[split_point1:split_point2]
        test_metadata = self.metadata[split_point2:]
        
        self.datasets = {"train": mtrain, "val": mval, "test": mtest}
        self.metadata_dict = {"train": train_metadata, "val": val_metadata, "test": test_metadata}

        # load the first epochs
        self.batches = {"train": self.load_batches(self.datasets["train"], self.n_way_mtrain, self.metadata_dict["train"]), 
                        "val": self.load_batches(self.datasets["val"], self.n_way, self.metadata_dict["val"]),
                        "test": self.load_batches(self.datasets["test"], self.n_way, self.metadata_dict["test"])}
        self.batch_cursor = {"train": 0, "val": 0, "test": 0}
        

    def sample_images_according_to_metadata(self, metadata, current_class, size):
        """
        Samples image instances by selecting the K-nearest-neighbors in the metadata space, 
        the centroid is a randomly generated vector within the bouding box of metadata points.

        metadata: np.ndarray of size e.g. (900, 20, 2)
            where 900 means the total number of classes
                  20 means 20 image instances
                  2 means 2-D metadata search space
        metadata_candidate_points: np.ndarray (20, 2),

        returns the indices of image instances
        """
        metadata_candidate_points = metadata[current_class]
        random_centroid = []
        for axis_ in range(metadata_candidate_points.shape[1]):
            axis_ = metadata_candidate_points[:, axis_]
            lb, ub = axis_.min(), axis_.max()
            random_centroid.append(np.random.uniform(low=lb, high=ub, size=None))

        nearest_neighbors = NearestNeighbors(n_neighbors=size, algorithm="auto", n_jobs=self.n_jobs_knn)
        nearest_neighbors.fit(metadata_candidate_points)

        # selected_img_indices is a 1-D np.ndarray containing indices of image instances
        selected_img_indices = nearest_neighbors.kneighbors([random_centroid], return_distance=False).ravel()

        return selected_img_indices


    def load_batches(self, data, n_way, metadata):
        """
        example format: metadata (900, 20, 2)
        """
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
                    # here, the metadata Z is used
                    ## Be careful, if self.k_support + self.k_query = total number of image instances, 
                    ## then metadata will have 0 effect, because we decide that support instances and 
                    ## query instances are drawn in the same way.
                    selected_img_indices = self.sample_images_according_to_metadata(metadata, 
                                                                                    current_class, 
                                                                                    size=self.k_support+self.k_query)

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
            self.batches[mode] = self.load_batches(self.datasets[mode], n_way, self.metadata_dict[mode])
            self.batch_cursor[mode] = 0 

        next_batch = self.batches[mode][self.batch_cursor[mode]]
        self.batch_cursor[mode] += 1

        return next_batch
    


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time

    dataset_path = "../OmniPrint-metaX/meta3"
    nb_batches_to_preload = 10


    dataloader = OmniglotLikeDataloader(dataset_path, nb_batches_to_preload=2)
    

    t0 = time.time()

    dataloader = OmniglotLikeDataloader(dataset_path, nb_batches_to_preload=nb_batches_to_preload)

    for _ in range(nb_batches_to_preload - 1):
        x_spt, y_spt, x_qry, y_qry = dataloader.next("train")

    t1 = time.time() - t0

    print("****** OmniglotLikeDataloader uses {:.1f} seconds to generate {} batches ******".format(t1, nb_batches_to_preload - 1))
    
    dataloader = OmniglotLikeZDataloader(dataset_path, z_metadata=["shear_x", "rotation"], nb_batches_to_preload=2)

    t0 = time.time()

    dataloader = OmniglotLikeZDataloader(dataset_path, z_metadata=["shear_x", "rotation"], nb_batches_to_preload=nb_batches_to_preload)

    for _ in range(nb_batches_to_preload - 1):
        x_spt, y_spt, x_qry, y_qry = dataloader.next("train")

    t1 = time.time() - t0

    print("****** OmniglotLikeZDataloader uses {:.1f} seconds to generate {} batches ******".format(t1, nb_batches_to_preload - 1))


    
    
    


    




