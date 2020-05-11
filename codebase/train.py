import numpy as np
import torch
from code.data_handling import Dataset


def train_k_fold(config, k=10):
    for fold_no in range(1, k+1):
        fold_dataset = Dataset(fold_no)

        # TODO
        model = None

        # Train a model with the current fold
        train(model, config, fold_dataset)


def train_full(config):
    dataset = Dataset("full")

    # TODO
    model = None

    # Train a model with the full dataset
    train(model, config, dataset)


def train_dummy(config,):
    dataset = Dataset("dummy")

    # TODO
    model = None

    # Train a model with the dummy dataset
    train(model, config, dataset)


def train(model, config, dataset):
    """Trains the model on the given dataset
        Args:
            - config (?): contains information on hyperparameter settings and such.
            - dataset (Dataset object): dataset with which to train the model.
    """

    # Initialize the device which to run the model on
    device = torch.device("cuda:0")#(config.device)

    # Setup the loss and optimizer
    criterion = None
    optimizer = None

    for epoch in range(config.epochs):

        ### NAMES ARE SUBJECT TO CHANGE, THIS IS ONLY FOR THE FORM ###
        # s = srch_id (integer/str)
        # X = train batch (matrix)
        # Y = relevances (vector)
        # rand = random_bool value ()
        for s, X, Y, rand in fold_dataset:
            X = X.to(device)
            # Then = calculat

        # If we would want the validation stats every epoch:
        with torch.no_grad():
            for  s_V, X_V, Y_V, rand_V in fold_dataset.validation_batch_iter():
                X_V = X_V.to(device)
                # Then = calculat
