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
from torchvision import models


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
    takes a torch.Tensor with weights of all the
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


def plot_resnet_filters(conv1_weights: torch.Tensor):
    """
    takes of the weights of first conv layer
    in resnet18 and displays the first 12 filters
    in it.

    Args:
        conv1_weights: contains the weights of first conv layer.
    """
    fig = plt.figure(figsize=(10, 10))
    rows, cols = 5, 5
    for i in range(len(conv1_weights)):
        if i == 25:
            break

        # This is because matplotlib takes (height, width, channels)
        img = conv1_weights[i].permute(1, 2, 0)
        fig.add_subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.title(f'Figure {i}')
        plt.axis('off')

    plt.show()


def plot_grey_scaled_resnet_filters(conv1_weights):
    """
    takes of the weights of first conv layer
    in resnet18 and displays the first 12 filters
    in it by gray scaling them.

    Args:
        conv1_weights: contains the weights of first conv layer.
    """
    fig = plt.figure(figsize=(10, 10))
    rows, cols = 5, 5
    for i in range(len(conv1_weights)):
        if i == 25:
            break
        img = conv1_weights[i]
        # greyscaling filter.
        grey_scale_img = torch.argmax(img, dim=0, keepdim=True)
        fig.add_subplot(rows, cols, i + 1)
        plt.imshow(grey_scale_img.squeeze())
        plt.title(f"Figure {i}")
        plt.axis('off')

    plt.show()


def plot_resnet_filtered_images(conv1_weights, image):
    """
    takes a torch.Tensor with weights of all the
    filters and displays the first image with all
    filters applied.

    Args:
        conv1_weights: contains the weights of all filters in
                       convolutional layer.
        image: contains the image as numpy.ndarray
    """
    fig = plt.figure(figsize=[10, 10])
    rows, cols = 5, 5
    for i in range(len(conv1_weights)):
        if i == 25:
            break
        filter = conv1_weights[i]
        grey_scale_filter = torch.mean(filter, dim=0, keepdim=True).numpy()
        filter_img = cv2.filter2D(image[0], -1, grey_scale_filter[0])
        fig.add_subplot(rows, cols, i + 1)
        plt.imshow(filter_img, cmap='gray')

    plt.show()

# A main function that runs the examination of layers.
def main(argv):
    examine_custom_model = int(argv[1])
    examine_resnet_model = int(argv[2])

    train_data, test_data, class_names = create_dataloaders(32)
    image, label = next(iter(train_data))
    image = image[0].numpy()

    if examine_custom_model == 1:  # If true examine the custom model.
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
        plot_filtered_images(conv1_weights, image)

    if examine_resnet_model == 1:  # if true examine resnet model.
        model_resnet = models.resnet18(pretrained=True)
        # get weights of first conv layer.
        conv1_weights = model_resnet.state_dict()['conv1.weight']

        plot_resnet_filters(conv1_weights)
        plot_grey_scaled_resnet_filters(conv1_weights)

        plot_resnet_filtered_images(conv1_weights, image)


if __name__ == "__main__":
    main(sys.argv)
