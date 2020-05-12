import numpy as np
import torch
from codebase.data_handling import BookingDataset
from codebase.nn_models import ExodiaNet
import time
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
    criterion = torch.nn.MSELoss() ################################# WANTED TO CHECK :(
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    for epoch in range(epochs):

        # to keep track of batches/second
        t = time.time()

        train_loss = 0
        for search_id, X, Y, rand_bool in dataset:
            X = X.to(device)
            Y = Y.to(device)

            out = model(X)

####################### NEW ##################
######## Do we want initialization loss?
####### convergence criterium? ######
            batch_loss = criterion(out, Y)########srch_id level might be interesting for performance analysis (what kind of srches are easy to predict etc.)
            train_loss += batch_loss.sum()
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
##############################################

        validation_loss = 0
        with torch.no_grad():
            for search_id_V, X_V, Y_V, rand_bool_V in dataset.validation_batch_iter():
                X_V = X_V.to(device)
                Y_V = Y_V.to(device)

                out_val = model(X_V)
                validation_loss += criterion(out_val, Y_V).sum()
        validation_loss /= dataset.val_len
        print(f"Train Loss: {train_loss/len(dataset)}, Validation Loss: {validation_loss}\nSeconds per epoch: {time.time()-t}")

def train_main(hyperparameters, fold_config):

    if fold_config != "k_folds":

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
