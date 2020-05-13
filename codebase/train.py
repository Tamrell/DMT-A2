import numpy as np
import torch
from codebase.data_handling import BookingDataset
from codebase.nn_models import ExodiaNet
from codebase.dynamic_hist import DynamicHistogram
from codebase import lambdaCriterion, evaluation
import matplotlib.pyplot as plt
import time
from codebase import io


def train(model, dataset, hyperparameters):
    """Trains the model on the given dataset
        Args:
            - config (?): contains information on hyperparameter settings and such.
            - dataset (Dataset object): dataset with which to train the model.
    """
    TEST_SIGMA = 1e0 ##############################################
    print(f"TESTING WITH SIGMA={TEST_SIGMA}")###################################3

    # Setup the loss and optimizer
    device = hyperparameters['device']
    model.to(device)
    criterion = lambdaCriterion.DeltaNDCG("pytorch")  # lage standaarden
    # criterion = torch.nn.MSELoss() ################################# WANTED TO CHECK :(
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters['learning_rate'])
    gt = evaluation.load_ground_truth() ########### HACKS
    d_hist = DynamicHistogram()

    for epoch in range(hyperparameters['epochs']):

        # to keep track of batches/second
        t = time.time()

        i = 0
        trn_ndcg = list()
        losses = []

        for search_id, X, Y, rand_bool, props in dataset:
            if not gt[search_id]["iDCG@end"]:
                continue
            i += 1
            X = X.to(device)
            Y = Y.to(device)
            optimizer.zero_grad()

            out = model(X)

####################### NEW ##################
######## Do we want initialization loss?
####### convergence criterium? ######
            crit, denominator = criterion.compute_loss_torch(out, Y, gt[search_id]["iDCG@end"], TEST_SIGMA, device)
            with torch.no_grad():
                losses.append(crit.sum().item())

                idx = torch.argsort(denominator.squeeze(), descending=True)[:5]
                trn_ndcg.append(((denominator[idx] @ Y[idx])/gt[search_id]["iDCG@5"]).item())


            # input(crit)
            batch_loss = crit.sum() ########srch_id level might be interesting for performance analysis (what kind of srches are easy to predict etc.)
            if i > 99 and i%100 == 0:
                print(f"{i}: {np.mean(trn_ndcg[-100:])}, loss: {np.mean(losses[-100:])}      ", end="\r")

            batch_loss.backward()
            optimizer.step()


##############################################
        # print("Exodia has gotten even stronger! (hopefully)")

        d_hist.update(trn_ndcg)

        val_ndcg = list()
        pred_string = "srch_id,prop_id\n"
        with torch.no_grad():
            kek=0
            for search_id_V, X_V, Y_V, rand_bool_V, props_V in dataset.validation_batch_iter():
                if not gt[search_id]["iDCG@end"]:
                    kek+=1
                    continue

                X_V = X_V.to(device)
                Y_V = Y_V.to(device)

                out_val = model(X_V)

                ranking_prediction_val = prediction_to_property_ranking(out_val, props_V)
                for prop in ranking_prediction_val:
                    pred_string += f"{search_id}, {prop.item()}\n"

                crit, denominator = criterion.compute_loss_torch(out_val, Y_V, gt[search_id_V]["iDCG@end"], TEST_SIGMA, device)
                idx = torch.argsort(denominator.squeeze(), descending=True)[:5]
                val_ndcg.append(((denominator[idx] @ Y_V[idx])/gt[search_id]["iDCG@5"]).item())


        model_id = io.add_model(hyperparameters)
        io.save_model(model_id, model)
        print(f"Train NDCG: {np.mean(trn_ndcg):5f}, Validation NDCG: {np.mean(val_ndcg):5f}, t loss: {np.mean(losses):5f}, model_id: {model_id}, (Epoch time: {time.time()-t:5f})")


def prediction_to_property_ranking(prediction, properties):
    ranking = properties[torch.argsort(torch.argsort(prediction.squeeze(), descending=True))]
    return ranking.squeeze()

def train_main(hyperparameters, fold_config):

    if fold_config != "k_folds":

        dataset = BookingDataset(fold_config)
        print("Done\nDrawing cards... WAIT! IS IT?!?")
        print("Summoning the forbidden one...")
        model = ExodiaNet(dataset.feature_no,
                          hyperparameters['layer_size'],
                          hyperparameters['layers'],
                          hyperparameters['attention_layer_idx'],
                          hyperparameters['resnet'],
                          hyperparameters['relu_slope'])

        print("Done, It's time to d-d-d-ddd-d-d-d-dduel!")
        print("ExodiaNet enters the battlefield...")
        train(model,
              dataset,
              hyperparameters)
        return

    K = 10
    for fold_no in range(1, K + 1):
        dataset = BookingDataset(fold_no)
        model = ExodiaNet(dataset.feature_no,
                          hyperparameters['layer_size'],
                          hyperparameters['layers'],
                          hyperparameters['attention_layer_idx'],
                          hyperparameters['resnet'],
                          hyperparameters['relu_slope'])
        train(model,
              dataset,
              hyperparameters)
    return
