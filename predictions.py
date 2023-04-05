"""
1. Jyothi Vishnu Vardhan Kolla
2. Vidya Ganesh

Project-5: CS-5330 -> Spring 2023.

This file contains the code to load and evaluate
the model with inputs.
"""
import sys
from models import LeNet
from helper_functions import load_model
from dataloader import create_dataloaders, GreekTransform
import matplotlib.pyplot as plt
import torch
import os
import torchvision
from PIL import Image


def plot_predictions(labels: list, pred_labels: list, images: torch.tensor, class_names: list):
    """
    This functions takes original labels and
    predicted labels as list and images and plots
    them to visualize predictions.

    Args:
        labels: A list containing original labels.
        pred_labels: A list containing pred_labels.
        images: torch.tesnor object containing images pixels.
    """
    torch.manual_seed(42)
    figure = plt.figure(figsize=(10, 10))
    rows, cols = 3, 3

    for i in range(1, rows * cols + 1):
        img, label = images[i], labels[i]
        figure.add_subplot(rows, cols, i)
        plt.imshow(img.squeeze(), cmap='gray')
        plt.title(f"Predicted:{class_names[pred_labels[i]]}")
        plt.axis('off')
    plt.show()


def make_custom_predictions(dir_path: str, model: torch.nn.Module):
    """
    A function that reads a directory of custom images and visualizes
    the images with predictions.
    """
    torch.manual_seed(42)
    figure = plt.figure(figsize=(10, 10))
    rows, cols = 3, 4

    data_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.1307, ), (0.3081,))
    ])
    # Get the count of images in the directory.
    files = os.listdir(dir_path)
    i = 0
    for file in files:
        image_ = Image.open(os.path.join(dir_path, file))
        image_ = image_.convert(mode="L")
        image_ = image_.resize((28, 28))
        image_ = data_transform(image_)
        image_ = image_.unsqueeze(0)

        prediction = model(image_)
        prediction_label = int(torch.argmax(prediction, dim=1))
        figure.add_subplot(rows, cols, i + 1)
        plt.imshow(image_.squeeze())
        plt.title(f"Prediction:{prediction_label}")
        plt.axis('off')
        i += 1

    plt.show()


# Main functions that takes cmd args and performs predictions.
def main(argv):
    # load the model
    model = LeNet()
    print(model)
    model_path = "Models/base_model.pth"
    model_ = load_model(target_dir=model_path,
                        model=model)  # Load the model with all weights.

    # load and get data.
    train_data, test_data, class_names = create_dataloaders(32)
    images, labels = next(iter(test_data))
    torch.set_printoptions(precision=2)
    model_.eval()

    # Perfrom predictions.
    pred_labels = []
    with torch.inference_mode():
        for i in range(10):
            image = images[i].unsqueeze(0)
            label = labels[i]

            prediction = model_(image)
            # store the predictins.
            pred_labels.append(int(torch.argmax(prediction, dim=1)))
            print(f"Prediction Probabilities are: {prediction}")
            print(
                f"Original label:{label}, predicted label:{torch.argmax(prediction, dim=1)}")
            print(
                "--------------------------------------------------------------------------")

    make_custom_predictions("custom-data/Mnist", model_)
    # Plot predictions.
    plot_predictions(labels, pred_labels, images, class_names)


if __name__ == "__main__":
    main(sys.argv)
