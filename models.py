"""
1. Jyothi Vishnu Vardhan Kolla
2. Vidya Ganesh

Project-5: CS-5330 -> Spring 2023.

This file contains the code to create all 
the models used in the project.
"""

import torch
from torch import nn

class LeNet(nn.Module):
    """
    Creates a LenNet based architecture to
    classify digits in Mnist dataset.
    """
    def __init__(self):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(
            in_channels=1,
            out_channels=10,
            kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(
            in_channels=10,
            out_channels=20,
            kernel_size=5,
            ),
            nn.Dropout2d(0.5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=20 * 4 * 4, out_features=50),
            nn.Linear(50, 10)
        )

    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.classifier(x)

        return nn.functional.log_softmax(x)

    