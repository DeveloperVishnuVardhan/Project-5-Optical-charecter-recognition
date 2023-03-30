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
from dataloader import create_dataloaders
import matplotlib.pyplot as plt
import torch


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

# Main functions that takes cmd args and performs predictions.


def main(argv):
    # load the model
    model = LeNet()
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

    # Plot predictions.
    plot_predictions(labels, pred_labels, images, class_names)


if __name__ == "__main__":
    main(sys.argv)
