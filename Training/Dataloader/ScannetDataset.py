"""
Implements the ScanNET dataset to be used by a torch dataloader.
"""

import h5py
import os
import torch.utils.data.dataset

import numpy as np

from Training.Dataloader.data_transformer import ToTensor


class ScanNetDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, dataset_directory, split, transform=None):
        """Initialised the ScanNET Dataset.

        Parameters:
            dataset_directory (str): path to the dataset directory
            split (str): Either train or val depending on which split is wanted
            transform (Compose): composition of several transformation functions (Default: None)
        """
        self.root = os.path.join(dataset_directory, "Scannet", split)

        for root, dirs, files in os.walk(self.root):
            self.length = int(len(files) / 4)
            break

        self.split = split
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        number = f"{idx}".zfill(6)

        img_data = h5py.File(os.path.join(self.root, f"image_{number}.hdf5"))
        image = np.array(img_data["dataset"]).astype('float32')
        img_data.close()

        depth_data = h5py.File(os.path.join(self.root, f"depth_{number}.hdf5"))
        depth = np.array(depth_data["dataset"]).astype('float32')
        depth_data.close()

        normal_data = h5py.File(os.path.join(self.root, f"normal_{number}.hdf5"))
        normal = np.array(normal_data["dataset"]).astype('float32')
        normal_data.close()

        planes_data = h5py.File(os.path.join(self.root, f"planes_{number}.hdf5"))
        planes = np.array(planes_data["dataset"]).astype("float32")
        planes_data.close()

        mask = np.logical_or(np.isnan(depth), (depth == 0))
        mask = np.logical_or(mask, np.all(normal == 0, axis=2))
        mask = np.logical_not(mask)

        if self.transform is None:
            self.transform = ToTensor()

        sample = {'image': image, 'depth': depth, 'normal': normal, 'mask': mask, 'planes': planes}
        return self.transform(sample)
