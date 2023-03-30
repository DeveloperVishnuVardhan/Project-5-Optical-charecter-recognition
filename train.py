"""
1. Jyothi Vishnu Vardhan Kolla
2. Vidya Ganesh

Project-5: CS-5330 -> Spring 2023.

This file contains the code to load all
the dependencies and train the model.
"""

from dataloader import create_dataloaders
from models import LeNet
import torch
from torch import nn
from tqdm.auto import tqdm
from train_prep import train_step, test_step
from helper_functions import save_model, save_results, plot_loss_curves
import pandas as pd
import matplotlib.pyplot as plt
import sys

# A function that trains the network
def train_network(model: torch.nn.Module,
                  train_dataloader: torch.utils.data.DataLoader,
                  test_dataloader: torch.utils.data.DataLoader,
                  optimizer: torch.optim.Optimizer,
                  loss_fn: torch.nn.Module,
                  epochs: int,
                  device: torch.device="mps"):
    """
    trains the model and saves the model into local directory.

    Args:
        model: Model to use.
        train_dataloader: train_dataloader object with training data.
        test_dataloader: test_dataloader object with testing data.
        optiimizer: Type of optimization to use.
        loss: loss function to use for training the model.
        epochs: Number of epochs to train the model.
        device: device to use for computing.

    Returns: 
        A dictionary of training and testing losses and accuracies.
    """

    # Create a results dictionary.
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)
        
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn)
        # Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results
        
         


# A function that reads cmd arguements and trains the model.
def main(argv):
    batch_size = int(argv[1])
    epochs = int(argv[2])
    train_mode = int(argv[3])
    test_mode = int(argv[4])

    train_dataloader, test_dataloader, class_names = create_dataloaders(
        batch_size=batch_size)
    

    if train_mode == 1:
        # Initialize the model
        model = LeNet().to("mps")
        loss = nn.CrossEntropyLoss()
        optim = torch.optim.SGD(params=model.parameters(), lr=0.1)
        results = train_network(model=model,
                                train_dataloader=train_dataloader,
                                test_dataloader=test_dataloader,
                                optimizer=optim,
                                loss_fn=loss,
                                epochs=epochs)
        
        print(results)
        save_model(model, "Models", "base_model.pth") # Save the model to disk.
        save_results(results, "Models") # Save the results of model in a text file.

    if test_mode == 1:
        # Analyze and plot the results above training mode.
        df1 = pd.read_csv("Models/results.csv")
        df1.drop("Unnamed: 0", axis=1, inplace=True)
        my_dict = df1.to_dict('list')
        plot_loss_curves(my_dict)
        plt.show()




if __name__ == "__main__":
    main(sys.argv)
