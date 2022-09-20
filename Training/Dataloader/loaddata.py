"""
Implements all functions to get dataloader for the different datasets.
"""

from torch.utils.data import DataLoader
from torchvision import transforms

from Training.Dataloader.HypersimDataset import HypersimDataset
from Training.Dataloader.ScannetDataset import ScanNetDataset
from Training.Dataloader.data_transformer import *
from Training.settings import SystemSettings
from Utilities.DatasetEnum import DatasetEnum


def get_training_data(dataset_dir, batch_size, dataset):
    """Returns the dataloader for the asked dateset for training.

    Parameters:
        dataset_dir (str): Path to the dataset directory
        batch_size (int): Number of images per batch
        dataset (DatasetEnum): Dataset used to train

    Returns:
        DataLoader:Dataloader of the given dataset
    """
    if dataset == DatasetEnum.HYPERSIM:
        dataloader = get_training_data_hypersim(dataset_dir, batch_size)
    elif dataset == DatasetEnum.SCANNET:
        dataloader = get_training_Data_scannet(dataset_dir, batch_size)
    else:
        raise Exception(f"Dataset {dataset.name} is not supported!")
    return dataloader


def get_training_data_hypersim(dataset_dir, batch_size):
    """Returns the training dataloader for Hypersim.

    Parameters:
        dataset_dir (str): Path to the dataset directory
        batch_size (int): Number of images per batch

    Returns:
        DataLoader:Dataloader for Hypersim
    """
    transformed_training = HypersimDataset(dataset_dir, 'train', transform=transforms.Compose([
            ToTensor(),
            Scale(228),
            SetNANTo(0),
            Grayscale(),
            ColorJitter(0.2, 0.2, 0.2, 0.2)
        ]))

    return DataLoader(transformed_training, batch_size, shuffle=True,
                      num_workers=SystemSettings.num_cpus, pin_memory=False)

def get_training_Data_scannet(dataset_dir, batch_size=10):
    """Returns the training dataloader for ScanNET.

    Parameters:
        dataset_dir (str): Path to the dataset directory
        batch_size (int): Number of images per batch

    Returns:
        DataLoader:Dataloader for ScanNET
    """
    transformed_training = ScanNetDataset(dataset_dir, 'train', transform=transforms.Compose([
        ToTensor(),
        Scale(228),
        SetNANTo(0),
        Grayscale(),
        ColorJitter(0.2, 0.2, 0.2, 0.2)
    ]))

    return DataLoader(transformed_training, batch_size,
                      num_workers=SystemSettings.num_cpus, pin_memory=False, shuffle=True)

def get_testing_data_hypersim(dataset_dir, batch_size=64):
    """Returns the testing dataloader for Hypersim.

    Parameters:
        dataset_dir (str): Path to the dataset directory
        batch_size (int): Number of images per batch

    Returns:
        DataLoader:Dataloader for Hypersim
    """
    transformed_testing = HypersimDataset(dataset_dir, 'val', transform=transforms.Compose([
        ToTensor(),
        Scale(192),
        SetNANTo(0)
    ]))

    return DataLoader(transformed_testing, batch_size, shuffle=False,
                      num_workers=SystemSettings.num_cpus, pin_memory=False)


def get_testing_data_scannet(dataset_dir, batch_size=10):
    """Returns the testing dataloader for ScanNET.

    Parameters:
        dataset_dir (str): Path to the dataset directory
        batch_size (int): Number of images per batch

    Returns:
        DataLoader:Dataloader for ScanNET
    """
    transformed_training = ScanNetDataset(dataset_dir, 'val', transform=transforms.Compose([
        ToTensor(),
        Scale(192),
        SetNANTo(0)
    ]))

    return DataLoader(transformed_training, batch_size,
                      num_workers=SystemSettings.num_cpus, pin_memory=False)
