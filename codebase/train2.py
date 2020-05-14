import numpy as np
import torch
from codebase.data_handling import BookingDataset
from codebase.nn_models import ExodiaNet
from codebase.lambdacriterion2 import train_loop_plug
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
    TEST_SIGMA = 1e1 ##############################################
    print(f"TESTING WITH SIGMA={TEST_SIGMA}")###################################3

    # Setup the loss and optimizer
    device = hyperparameters['device']
    model[0].to(device)
    model[1].to(device)
    # criterion = lambdaCriterion.DeltaNDCG("pytorch")  # lage standaarden
    # criterion = torch.nn.MSELoss() ################################# WANTED TO CHECK :(
    optimizers = []
    optimizers.append(torch.optim.Adam(model[0].parameters(), lr=hyperparameters['learning_rate']))
    optimizers.append(torch.optim.Adam(model[1].parameters(), lr=hyperparameters['learning_rate']))
    # optimizer = torch.optim.SGD(model.parameters(), lr=hyperparameters['learning_rate'], momentum=0.9)
    gt = evaluation.load_ground_truth() ########### HACKS
    # d_hist = DynamicHistogram(dynamic_hist)


    for i in range(hyperparameters['epochs']):
        t = time.time()
        train_loop_plug(model, dataset, optimizers, TEST_SIGMA, device)
        val_ndcg = list()
        val_ndcg_at5 = list()
        with torch.no_grad():
            kek=0
            pred_string = "srch_id,prop_id\n"
            for search_id_V, X_V, Y_V, rand_bool_V, props_V in dataset.validation_batch_iter():
                if not gt[search_id_V]["iDCG@end"]:
                    kek+=1
                    continue

                X_V = X_V.to(device)
                Y_V = Y_V.to(device)

                out_val = model[rand_bool_V](X_V)
                ranking_prediction_val = prediction_to_property_ranking(out_val, props_V)

                for prop in ranking_prediction_val.squeeze():
                    pred_string += f"{search_id_V},{prop.item()}\n"


                val_dcg_pred_at5_idx = torch.argsort(out_val.squeeze(), descending=True)[:5]
                val_dcg_max_at5_idx = torch.argsort(Y_V.squeeze(), descending=True)[:5]


                val_dcg_pred_elements = Y_V.squeeze() / torch.log2(torch.argsort(torch.argsort(out_val.squeeze(), descending=True)).float() + 2)
                val_dcg_max_elements = Y_V.squeeze() / torch.log2(torch.argsort(torch.argsort(Y_V.squeeze(), descending=True)).float() + 2)


                # val_dcg_pred = torch.sum(Y_V.squeeze() / torch.log2(torch.argsort(torch.argsort(out_val.squeeze(), descending=True)) + 1))
                # val_dcg_max = torch.sum(Y_V.squeeze() / torch.log2(torch.argsort(torch.argsort(Y_V.squeeze(), descending=True)) + 1))


                # maxDCG = torch.sum(Y.squeeze() / torch.log2(torch.argsort(torch.argsort(Y.squeeze(), descending=True)) + 1))
                # print("val_dcg_pred_elements")
                # print(val_dcg_pred_elements)
                # print("val_dcg_max_elements")
                # print(val_dcg_max_elements)
                # print("Y_V")
                # print(Y_V)
                # print("torch.log2(torch.argsort(torch.argsort(Y_V.squeeze(), descending=True)).float() + 1)")
                # print(torch.log2(torch.argsort(torch.argsort(Y_V.squeeze(), descending=True)).float() + 2))
                # input()
                val_ndcg.append((torch.sum(val_dcg_pred_elements)/torch.sum(val_dcg_max_elements)).item())
                val_ndcg_at5.append((torch.sum(val_dcg_pred_elements[val_dcg_pred_at5_idx])/torch.sum(val_dcg_max_elements[val_dcg_max_at5_idx])).item())
                # input(val_ndcg_at5[-1])
                # val_ndcg.append(((denominator[idx] @ Y_V[idx])/gt[search_id_V]["iDCG@5"]).item())
        model_id = io.add_model(hyperparameters)
        io.save_val_predictions(model_id, pred_string)
        io.save_model(model_id, model)
        print(f"Validation NDCG: {np.mean(val_ndcg):5f}, Validation NDCG@5: {np.mean(val_ndcg_at5):5f}, model_id: {model_id}, (Epoch time: {time.time()-t:5f})")



def prediction_to_property_ranking(prediction, properties):
    ranking = properties.squeeze()[torch.argsort(prediction.squeeze(), descending=True)]
    return ranking.squeeze()

def train_main(hyperparameters, fold_config):

    if fold_config != "k_folds":

        dataset = BookingDataset(fold_config)
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
