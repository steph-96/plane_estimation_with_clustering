"""
Implements the Hypersim dataset to be used by a torch dataloader.
"""

import h5py
import os
import torch

import numpy as np
import pandas as pd
from torchvision.transforms import Compose

from Training.Dataloader.data_transformer import ToTensor
from Utilities.generate_planes import GeneratePlanes
from Utilities.util import Utility


class HypersimDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_directory, split, transform=None):
        """Initialised the Hypersim Dataset.

        Parameters:
            dataset_directory (str): path to the dataset directory
            split (str): Either train or val depending on which split is wanted
            transform (Compose): composition of several transformation functions (Default: None)
        """
        self.root = dataset_directory
        image_information = pd.read_csv(os.path.join(self.root, "hypersim_split.csv"))
        all_files = image_information[image_information["split_partition_name"] == split]

        directories = []
        # check which folders are downloaded
        for root, dirs, files in os.walk(self.root):
            directories = dirs
            break

        self.list = all_files[all_files["scene_name"].isin(directories)].copy().reset_index()
        self.split = split
        self.transform = transform
        self.intrinsics = pd.read_csv(os.path.join(self.root, "camera_intrinsics.csv"))
        self.h, self.w = 768, 1024

        self.image_path = "{}/images/scene_{}_final_hdf5/frame.{}.color.hdf5"
        self.depth_path = "{}/images/scene_{}_geometry_hdf5/frame.{}.depth_meters.hdf5"
        self.normal_path = "{}/images/scene_{}_geometry_hdf5/frame.{}.normal_cam.hdf5"
        self.planes_path = "{}/images/scene_{}_geometry_hdf5/frame.{}.planes.hdf5"

        self.failing_images = ["Dataset/ai_032_005/images/scene_cam_00_final_hdf5/frame.0015.color.hdf5",
                               "Dataset/ai_004_001/images/scene_cam_00_final_hdf5/frame.0067.color.hdf5"]

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        row = self.list.iloc[idx, :]
        # print(f"Scene: {row['scene_name']}, Frame: {row['frame_id']}")
        img_path = os.path.join(self.root, self.image_path.format(row["scene_name"], row["camera_name"],
                                                                  str(row["frame_id"]).zfill(4)))
        depth_path = os.path.join(self.root, self.depth_path.format(row["scene_name"], row["camera_name"],
                                                                    str(row["frame_id"]).zfill(4)))
        normal_path = os.path.join(self.root, self.normal_path.format(row["scene_name"], row["camera_name"],
                                                                      str(row["frame_id"]).zfill(4)))
        planes_path = os.path.join(self.root, self.planes_path.format(row["scene_name"], row["camera_name"],
                                                                      str(row["frame_id"]).zfill(4)))

        img_data = h5py.File(img_path, 'r')
        image = np.array(img_data["dataset"]).astype('float32')
        img_data.close()

        image = self.apply_tonemap(image)
        if np.isinf(image).any():
            print(f"inf in image {idx}")

        depth_data = h5py.File(depth_path, 'r')
        depth = np.array(depth_data["dataset"]).astype('float32')
        depth_data.close()

        mask = np.isnan(depth)
        mask = np.logical_not(mask)

        focal_length = Utility.calculate_hypersim_focal_length()
        if not np.isnan(focal_length):
            depth = Utility.convert_distance_to_depth_map(depth, focal_length)
            pass

        normal_data = h5py.File(normal_path, 'r')
        normal = np.array(normal_data["dataset"]).astype('float32')
        normal_data.close()

        if not os.path.exists(planes_path):
            # creates ground truth plane segmentation on the fly.
            # ATTENTION might take very long depending on how many still have to be created
            planes = self._create_gt_plane(depth, normal, mask, planes_path)
        else:
            planes_data = h5py.File(planes_path, 'r')
            try:
                planes = np.array(planes_data["dataset"].astype('float32'))
                planes_data.close()
            except KeyError:
                planes_data.close()
                os.remove(planes_path)
                planes = self._create_gt_plane(depth, normal, mask, planes_path)

        if self.transform is None:
            self.transform = ToTensor()

        sample = {'image': image, 'depth': depth, 'normal': normal, 'mask': mask, 'planes': planes}

        return self.transform(sample)

    def _create_gt_plane(self, depth, normal, mask, planes_path, cores=1):
        """Creates the ground truth plane segmentation.

        Parameters:
            depth (np.ndarray): Ground truth depth
            normal (np.ndarray): Ground truth surface normals
            mask (np.ndarray): NAN-mask to remove NAN Values
            planes_path (str): Path to save the ground truth plane segmentation to
            cores (int): Number of cpu cores to use (Default: 1)

        Returns
        np.ndarray: Ground Truth plane segmentation
        """
        normal4planes = torch.Tensor(normal).moveaxis(-1, 0)
        normal4planes[:, np.logical_not(mask)] = 0
        depth4planes = torch.Tensor(depth).unsqueeze(0)
        depth4planes[:, np.logical_not(mask)] = 0
        try:
            planes, labels = GeneratePlanes.plane_from_normals_and_depth(normal4planes, depth4planes, "hypersim", 0.15, cores)
            planes = np.moveaxis(np.array(torch.vstack((planes[0, 3:], labels[0]))), 0, -1)
            Utility.save_hdf5_file(planes, planes_path)
        except ValueError:
            print(f"{planes_path} was not able to pass a meanshift with a bandwidth of 0.15")
            height, width = depth.shape
            planes = np.zeros((height, width, 2))
        return planes

    def generate_gt(self, idx, cores):
        """Generates the ground truth plane segmentation for given id

        Parameters:
            idx (int): ID for which the ground truth will be generated.
            cores (int): Number of cores used for the generation.
        """
        row = self.list.iloc[idx, :]
        depth_path = os.path.join(self.root, self.depth_path.format(row["scene_name"], row["camera_name"],
                                                                    str(row["frame_id"]).zfill(4)))
        normal_path = os.path.join(self.root, self.normal_path.format(row["scene_name"], row["camera_name"],
                                                                      str(row["frame_id"]).zfill(4)))
        planes_path = os.path.join(self.root, self.planes_path.format(row["scene_name"], row["camera_name"],
                                                                      str(row["frame_id"]).zfill(4)))

        if os.path.exists(planes_path):
            planes_data = h5py.File(planes_path, 'r')
            try:
                planes = np.array(planes_data["dataset"].astype('float32'))
                planes_data.close()
                if planes.shape != (768, 1024, 2):
                    print("Wrong")
                    os.remove(planes_path)
                else:
                    return
            except KeyError:
                planes_data.close()
                os.remove(planes_path)

        depth_data = h5py.File(depth_path, 'r')
        depth = np.array(depth_data["dataset"]).astype('float32')
        depth_data.close()

        mask = np.isnan(depth)
        mask = np.logical_not(mask)

        focal_length = Utility.calculate_hypersim_focal_length()
        if not np.isnan(focal_length):
            depth = Utility.convert_distance_to_depth_map(depth, focal_length)
            pass

        normal_data = h5py.File(normal_path, 'r')
        normal = np.array(normal_data["dataset"]).astype('float32')
        normal_data.close()

        self._create_gt_plane(depth, normal, mask, planes_path, cores)

    def apply_tonemap(self, image):
        """ This function uses the tone-map creation of the hypersim dataset to ensure that the lighting of the scene is
        correct and not blown out.

        Parameters:
            image (np.ndarray): Numpy array of the unedited image loaded from the database.

        Returns:
            np.ndarray: Image with corrected brightness
        """
        gamma = 1.0 / 2.2  # standard gamma correction exponent
        inv_gamma = 1.0 / gamma
        percentile = 90  # we want this percentile brightness value in the unmodified image...
        brightness_nth_percentile_desired = 0.8  # ...to be this bright after scaling

        # "CCIR601 YIQ" method for computing brightness
        brightness = 0.3 * image[:, :, 0] + 0.59 * image[:, :, 1] + 0.11 * image[:, :, 2]
        brightness_valid = brightness[np.logical_not(np.isinf(brightness))]

        # if the kth percentile brightness value in the unmodified image is less than this, set the scale to 0.0
        # to avoid divide-by-zero
        eps = 0.0001
        brightness_nth_percentile_current = np.percentile(brightness_valid, percentile)

        if brightness_nth_percentile_current < eps:
            scale = 0.0
        else:
            scale = np.power(brightness_nth_percentile_desired, inv_gamma) / brightness_nth_percentile_current

        return np.clip(np.power(np.maximum(scale * image, 0), gamma), 0, 1)
