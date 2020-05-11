import numpy as np
import torch
from codebase.data_handling import BookingDataset
from codebase.nn_models import ExodiaNet
from codebase import io


def train(model, dataset, epochs, learning_rate, device):
    """Trains the model on the given dataset
        Args:
            - config (?): contains information on hyperparameter settings and such.
            - dataset (Dataset object): dataset with which to train the model.
    """

    # Setup the loss and optimizer
    model.to(device)
    criterion = None  # lage standaarden
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    for epoch in range(epochs):
        for search_id, X, Y, rand_bool in dataset:
            X = X.to(device)
            out = model(X)
            print(out)


        with torch.no_grad():
            for search_id_V, X_V, Y_V, rand_bool_V in dataset.validation_batch_iter():
                X_V = X_V.to(device)


def train_main(hyperparameters, fold_config):

    if fold_config != "k_folds":

        print("\nPreparing Dataset...")
        dataset = BookingDataset(fold_config)
        print("Done")
        model_id = io.add_model(hyperparameters)

        print("Summoning the forbidden one...")
        model = ExodiaNet(model_id,
                          dataset.feature_no,
                          hyperparameters['layer_size'],
                          hyperparameters['layers'],
                          hyperparameters['attention_layer_idx'],
                          hyperparameters['resnet'],
                          hyperparameters['relu_slope'])

        print("Done, It's time to d-d-d-ddd-d-d-d-dduel!")
        train(model,
              dataset,
              hyperparameters["epochs"],
              hyperparameters["learning_rate"],
              hyperparameters["device"])
        return

    K = 10
    for fold_no in range(1, K + 1):
        dataset = BookingDataset(fold_no)
        model_id = io.add_model(hyperparameters)
        model = ExodiaNet(model_id,
                          dataset.feature_no,
                          hyperparameters['layer_size'],
                          hyperparameters['layers'],
                          hyperparameters['attention_layer_idx'],
                          hyperparameters['resnet'],
                          hyperparameters['relu_slope'])
        train(model,
              dataset,
              hyperparameters["epochs"],
              hyperparameters["learning_rate"],
              hyperparameters["device"])
    return
