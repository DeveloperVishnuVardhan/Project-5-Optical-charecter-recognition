"""
1. Jyothi Vishnu Vardhan Kolla
2. Vidya Ganesh

Project-5: CS-5330 -> Spring 2023.

This file contains the code to load the
training and testing data into disk.
"""

import torch
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader


data_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        (0.1307, ), (0.3081,))
])


def create_dataloaders(batch_size: int):
    """
    Creates training and testing DataLoaders.

    Takes in a batch_size as parameter and return an
    train_dataloader, test_dataloader, class_names.

    Args:
     batch_size: Number of samples per batch in each of the DataLoaders.

    Returns:
     A tuple of (train_dataloader, test_dataloader, class_names).
     where class_names is a list of target_classes.
    """
    train_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=data_transform,
    )

    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=data_transform
    )

    class_names = train_data.classes

    train_dataloader = DataLoader(train_data,
                                  batch_size=batch_size,
                                  shuffle=True)

    test_dataloader = DataLoader(test_data,
                                 batch_size=batch_size,
                                 shuffle=False)

    return train_dataloader, test_dataloader, class_names
