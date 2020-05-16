""" Handles training of models & contains LambdaRankCriterion class (bottom)

I kept the squeeze operations 100% as they were, not fokkin with those things m8.
"""



import numpy as np
import torch
from codebase.data_handling import BookingDataset
from codebase.nn_models import ExodiaNet
from codebase.dynamic_hist import DynamicHistogram
from codebase import lambdaCriterion as LC
from codebase import evaluation
import matplotlib.pyplot as plt
import time
from codebase import io


def train(model, dataset, hyperparameters, dynamic_hist=False):
    """Trains the model on the given dataset
        Args:
            - config (?): contains information on hyperparameter settings and such.
            - dataset (Dataset object): dataset with which to train the model.
    """
    EXP_VER = hyperparameters["exp_ver"] ####################### bool, determines NDCG type False = like blogpost
    TEST_SIGMA= 1e0 ##############################################
    print(f"TESTING WITH SIGMA={TEST_SIGMA}")###################################3

    # Setup the loss and optimizer
    device = hyperparameters['device']
    model[0].to(device)
    model[1].to(device)
    LRCrit = LC.lambdaRankCriterion(EXP_VER, device, TEST_SIGMA)
    optimizers = []
    optimizers.append(torch.optim.Adam(model[0].parameters(), lr=hyperparameters['learning_rate']))
    optimizers.append(torch.optim.Adam(model[1].parameters(), lr=hyperparameters['learning_rate']))
    gt = evaluation.load_ground_truth() ########### HACKS
    d_hist = DynamicHistogram(dynamic_hist)

    # Iter epochs
    for i in range(hyperparameters['epochs']):
        t = time.time()

        count = 0
        batch_size = hyperparameters['lambda_batch_size']
        ndcg_dict = dict()

        # Prediciton & gradient accumulation
        grad_batch, y_pred_batch = [], []
        # NDCG values for plotting with dynamic hist first is for rand bool off.
        trn_ndcg = [[],[]]


        total_batches = len(dataset)

        # Train loop
        for i, (search_id, X, Y, rand_bool, props) in enumerate(dataset):
            if torch.sum(Y) == 0:
                # no differences in this search_id
                continue
            if i%100 == 0:
                print(f"Exodia is getting stronger!? Batch: {i}/{total_batches}", end='\r')

            X = X.to(device)
            Y = Y.to(device)
            props=props.to(device)

            # Feed input to one of both models, dependent on rand_bool.
            y_pred = model[rand_bool](X)
            y_pred_batch.append(y_pred)

            # Calculate gradients.
            with torch.no_grad():

                # NDCG_train currently unused, maybe we want toi report this as well?
                grad, NDCG_train, NDCG_train_at5 = LRCrit.calculate_gradient_and_NDCG(y_pred, Y)
                grad_batch.append(grad)
                # if dynamic_hist:
                #     dcg_pred_elements = NDCG_relevance_grade.squeeze() / torch.log2(torch.argsort(torch.argsort(out_val.squeeze(), descending=True)).float() + 2)
                #     idx = torch.argsort(out_val, descending=True)[:5]
                trn_ndcg[rand_bool].append(NDCG_train_at5)

            # Apply gradients.
            count += 1
            if count % batch_size == 0:
                for grad, y_pred in zip(grad_batch, y_pred_batch):
                    y_pred.backward(grad / batch_size)

                # todo: check necessary
                torch.nn.utils.clip_grad_norm_(model[rand_bool].parameters(), 10)
                optimizers[rand_bool].step()
                model[rand_bool].zero_grad()

                # reset grad_batch, y_pred_batch used for gradient_acc
                grad_batch, y_pred_batch = [], []


        # Validation below
        with torch.no_grad():
            val_ndcgs = list()

            # [rand bool=0, rand_bool=1]
            val_ndcgs_at5 = [[],[]]
            pred_string = ["srch_id,prop_id"]
            for search_id_V, X_V, Y_V, rand_bool_V, props_V in dataset.validation_batch_iter():
                if torch.sum(Y) == 0:
                    continue

                X_V = X_V.to(device)
                Y_V = Y_V.to(device)
                out_val = model[rand_bool_V](X_V)
                ranking_prediction_val = prediction_to_property_ranking(out_val, props_V)

                for prop in ranking_prediction_val.squeeze():
                    pred_string.append(f"{search_id_V},{prop.item()}")

                val_ndcg, val_ndcg_at5 = LRCrit.calc_NDCG_val(out_val, Y_V)
                val_ndcgs.append(val_ndcg)
                val_ndcgs_at5[rand_bool_V].append(val_ndcg_at5)

        model_id = io.add_model(hyperparameters)
        io.save_val_predictions(model_id, "\n".join(pred_string))
        io.save_model(model_id, model)


        trn_at_end = []

        trn_at_end = []

        # ndcg@5 for train in [total, rand_bool=0, rand_bool=1]
        trn_at_5 = [np.mean(trn_ndcg[0] + trn_ndcg[1]), np.mean(trn_ndcg[0]), np.mean(trn_ndcg[0])]

        # ndcg@5 for validation in [total, rand_bool=0, rand_bool=1]
        val_at_5 = [np.mean(val_ndcgs_at5[0]+val_ndcgs_at5[1]), np.mean(val_ndcgs_at5[0]), np.mean(val_ndcgs_at5[1])]

        # epoch time
        epoch_time = time.time()-t

        print(f" (total/0/1) Trn NDCG@5: {trn_at_5[0]:.3f}/{trn_at_5[1]:.3f}/{trn_at_5[2]:.3f}, Val NDCG@5: {val_at_5[0]:.3f}/{val_at_5[1]:.3f}/{val_at_5[2]:.3f}, model_id: {model_id}, Val NDCG:{np.mean(val_ndcgs):4f}, (Epoch time: {epoch_time:4f})")
        d_hist.update(model_id, trn_ndcg)



def prediction_to_property_ranking(prediction, properties):
    ranking = properties.squeeze()[torch.argsort(prediction.squeeze(), descending=True)]
    return ranking.squeeze()

def train_main(hyperparameters, fold_config):

    if fold_config != "k_folds":
        dataset = BookingDataset(fold_config,
                                 artificial_relevance=hyperparameters["artificial_relevance"],
                                 uniform_relevance=hyperparameters["uniform_relevance"],
                                 use_priors=hyperparameters["use_priors"])
        print("Done\nDrawing cards... WAIT! IS IT?!?")
        print("Summoning the forbidden one...")
        model = []
        model.append(ExodiaNet(dataset.feature_no,
                          hyperparameters['layer_size'],
                          hyperparameters['layers'],
                          hyperparameters['attention_layer_idx'],
                          hyperparameters['resnet'],
                          hyperparameters['relu_slope']))
        model.append(ExodiaNet(dataset.feature_no,
                          hyperparameters['layer_size'],
                          hyperparameters['layers'],
                          hyperparameters['attention_layer_idx'],
                          hyperparameters['resnet'],
                          hyperparameters['relu_slope']))

        print("Done, It's time to d-d-d-ddd-d-d-d-dduel!")
        print("ExodiaNet enters the battlefield...")
        train(model,
              dataset,
              hyperparameters)
        return

    K = 10
    for fold_no in range(1, K + 1):
        dataset = BookingDataset(fold_no, artificial_relevance=hyperparameters["artificial_relevance"])
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
