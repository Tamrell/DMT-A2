import numpy as np
import torch
from codebase.data_handling import BookingDataset
from codebase.nn_models import ExodiaNet
from codebase import lambdaCriterion
from codebase import evaluation
import time
from codebase import io


def train(model, dataset, epochs, learning_rate, device):
    """Trains the model on the given dataset
        Args:
            - config (?): contains information on hyperparameter settings and such.
            - dataset (Dataset object): dataset with which to train the model.
    """
    TEST_SIGMA = 1
    print("TESTING WITH SIGMA=1")

    # Setup the loss and optimizer
    model.to(device)
    criterion = lambdaCriterion.DeltaNDCG("pytorch")  # lage standaarden
    # criterion = torch.nn.MSELoss() ################################# WANTED TO CHECK :(
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    gt = evaluation.load_ground_truth() ########### HACKS

    for epoch in range(epochs):

        # to keep track of batches/second
        t = time.time()

        i = 0
        trn_ndcg = list()

        for search_id, X, Y, rand_bool in dataset:
            if not gt[search_id]["iDCG@end"]:
                continue
            i += 1
            X = X.to(device)
            Y = Y.to(device)

            out = model(X)

####################### NEW ##################
######## Do we want initialization loss?
####### convergence criterium? ######
            crit, denominator = criterion.compute_loss_torch(out, Y, gt[search_id]["iDCG@end"], TEST_SIGMA, device)
            with torch.no_grad():
                idx = torch.argsort(denominator.squeeze(), descending=True)[:5]

                trn_ndcg.append(((denominator[idx] @ Y[idx])/gt[search_id]["iDCG@5"]).item())

            # input(crit)
            batch_loss = crit.sum() ########srch_id level might be interesting for performance analysis (what kind of srches are easy to predict etc.)
            if i > 100:
                print(f"{i}: {np.mean(trn_ndcg[-100:])}", end="\r")
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

##############################################
        print("Exodia has gotten even stronger! (hopefully)")

        val_ndcg
        with torch.no_grad():
            kek=0
            for search_id_V, X_V, Y_V, rand_bool_V in dataset.validation_batch_iter():
                if not gt[search_id]["iDCG@end"]:
                    kek+=1
                    continue

                X_V = X_V.to(device)
                Y_V = Y_V.to(device)

                out_val = model(X_V)
                crit, denominator = criterion.compute_loss_torch(out_val, Y_V, gt[search_id_V]["iDCG@end"], TEST_SIGMA, device)
                idx = torch.argsort(denominator.squeeze(), descending=True)[:5]
                val_ndcg.append(((denominator[idx] @ Y_V[idx])/gt[search_id]["iDCG@5"]).item())
        print(f"Train NDCG: {np.mean(trn_ndcg)}, Validation NDCG: {np.mean(val_ndcg)} (Epoch time: {time.time()-t})")

def train_main(hyperparameters, fold_config):

    if fold_config != "k_folds":

        dataset = BookingDataset(fold_config)
        print("Done\nDrawing cards... WAIT! IS IT?!?")
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
        print("ExodiaNet enters the battlefield...")
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
