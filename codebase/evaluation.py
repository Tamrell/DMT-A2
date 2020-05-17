import pandas as pd
import numpy as np
import pickle as pk
import torch
import time
from codebase import io
import os
from progressbar import progressbar
from codebase.data_handling import BookingDataset

def ndcg(pred_df, at=5):
    """Calculates the normalized discounted cumulative gain (by default: at 5)
        Args:
            - gt_fg (DataFrame): dataframe with ground truth data
            - pred_df (DataFrame): dataframe containing predictions

        Returns:
            - NDCG (float): normalized discounted cumulative gain as described
                            in blog post
    """
    gt_dict = load_ground_truth()
    ndcg_list = []
    logs = {i:1/np.log2(i+2) for i in range(50)}

    searches = len(pred_df["srch_id"].unique())
    grouped_pred = pred_df.groupby(by="srch_id")

    print("Calculating NDCG...")
    for s, sub_pred in progressbar(grouped_pred, max_value=searches):
        dcg = 0
        for i, p in enumerate(sub_pred["prop_id"]):
            if i < at:
                dcg += gt_dict[s][p] * logs[i]
        ndcg_list.append(dcg / gt_dict[s]["iDCG@5"])
    return np.mean(ndcg_list)

def load_ground_truth(path=os.path.join("data", "ground truth.p")):
    return pk.load(open(path, "rb"))

def prediction_to_property_ranking(prediction, properties):
    ranking = properties.squeeze()[torch.argsort(prediction.squeeze(), descending=True)]
    return ranking.squeeze()

def make_test_predictions(model, model_id, hyperparameters):
    test_data = BookingDataset("test", hyperparameters)
    with torch.no_grad():
        pred_string = ["srch_id,prop_id"]
        for search_id, X, rand_bool, props in test_data:
            out = model.forward(rand_bool, X)
            ranking = prediction_to_property_ranking(out, props)
            for prop in ranking:
                pred_string.append(f"{search_id},{prop.item()}")
        io.save_test_predictions(model_id, "\n".join(pred_string))
