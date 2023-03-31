"""
1. Jyothi Vishnu Vardhan Kolla
2. Vidya Ganesh

Project-5: CS-5330 -> Spring 2023.

This file contains the code to Examine the
network and analyze how it processes the data.
"""
import sys
from models import LeNet
from dataloader import create_dataloaders
import torch
import matplotlib.pyplot as plt
import cv2
import numpy


def get_layer1_weights(model: torch.nn.Module):
    """
    Gets the weights of first layer in the saved model.

    Args:
        model: The saved model with all its weights.

    Returns.
        A torch.Tensor containing all the weights 
        of first convolutioanal layer.
    """
    layer_name = "block_1.0.weight"
    layer_weights = model.state_dict()[layer_name]
    return layer_weights


def plot_filters(filter_weights: torch.Tensor):
    """
    Takes a torch.Tensor with weights of all the
    filters and displays them as a matplotlib plot.

    Args:
        filter_weights: Contains the weights all the filters in
                        convolutional layer.
    """
    fig = plt.figure(figsize=(8, 8))
    rows, cols = 3, 4
    for i in range(len(filter_weights)):
        img = filter_weights[i][0]
        fig.add_subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.title(f"Figure {i}")
        plt.axis('off')

    plt.show()


def plot_filtered_images(conv1_weights: torch.Tensor, image: numpy.ndarray):
    """
    akes a torch.Tensor with weights of all the
    filters and displays the first image with all
    filters applied.

    Args:
        conv1_weights: contains the weights of all filters in
                       convolutional layer.
        image: contains the image as numpy.ndarray
    """
    fig = plt.figure(figsize=(10, 10))
    rows, cols = 3, 4
    for i in range(len(conv1_weights)):
        kernel = conv1_weights[i][0].numpy()
        filter_img = cv2.filter2D(image[0], -1, kernel)
        fig.add_subplot(rows, cols, i + 1)
        plt.imshow(filter_img, cmap='gray')
        plt.axis('off')
    plt.show()


def main(argv):
    # Load the saved model.
    model_path = "Models/base_model.pth"
    state_dict = torch.load(model_path, map_location=torch.device('mps'))
    model = LeNet()
    model.load_state_dict(state_dict)

    # Get the weights of layer-1.
    conv1_weights = get_layer1_weights(model)

    # Plot weights of all filters in layer-1.
    plot_filters(conv1_weights)

    # Plot the image after applying filters.
    train_data, test_data, class_names = create_dataloaders(32)
    image, label = next(iter(train_data))
    image = image[0].numpy()
    plot_filtered_images(conv1_weights, image)


if __name__ == "__main__":
    main(sys.argv)
