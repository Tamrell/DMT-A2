import pandas as pd
import numpy as np
import pickle as pk
import time
import os
from progressbar import progressbar

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
    logs = {i:1/np.log2(i+2) for i in range(at)}

    searches = len(pred_df["srch_id"].unique())
    grouped_pred = pred_df.groupby(by="srch_id")

    print("Calculating NDCG...")
    for s, sub_pred in progressbar(grouped_pred, max_value=searches):
        dcg = 0
        for i, p in enumerate(sub_pred["prop_id"]):
            # print(gt_dict)[s]
            if i < at:
                dcg += gt_dict[s][p] * logs[i]
        ndcg_list.append(dcg / gt_dict[s]["iDCG@5"])
    return np.mean(ndcg_list)



def load_ground_truth(path=os.path.join("data", "ground truth.p")):
    return pk.load(open(path, "rb"))
