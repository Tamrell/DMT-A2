import pandas as pd
import numpy as np
import pickle as pk
import time
import os
from progressbar import progressbar

def ndcg(gt_dict, pred_df, at=5):
    """Calculates the normalized discounted cumulative gain (by default: at 5)
        Args:
            - gt_fg (DataFrame): dataframe with ground truth ordering
            - pred_df (DataFrame): dataframe containing predictions

        Returns:
            - NDCG (float): normalized discounted cumulative gain as described
                            in blog post
    """
    ndcg_list = []
    logs = {i:np.log2(i+2) for i in range(at)}

    searches = len(pred_df["srch_id"].unique())
    grouped_pred = pred_df.groupby(by="srch_id")

    print("Calculating NDCG...")
    for s, sub_pred in progressbar(grouped_pred, max_value=searches):
        dcg = 0
        for i, p in enumerate(sub_pred["prop_id"]):
            if i == at:
                break
            dcg += gt_dict[s][p] / logs[i]
        ndcg_list.append(dcg / gt_dict[s]["iDCG@5"])
    return np.mean(ndcg_list)


def compare_performance(model_list):
    #do we want this? I think it could be done more easily after the _results_ are saved in the general results csv
    pass


def load_ground_truth(path=os.path.join("data", "ground truth.p")):
    return pk.load(open(path, "rb"))

def load_predictions(path=os.path.join("benchmark_results", "benchmark V0 predictions.csv"), model_name="Benchmark V5"):
    return pd.read_csv(path)


if __name__ == '__main__':
    # Load predictions and ground truth
    pred = load_predictions()
    gt = load_ground_truth()

    # print results for the predictions
    print(ndcg(gt, pred))
    # print(ndcg(gt, pred2))
