import pandas as pd
import numpy as np
import pickle as pk
import time
from progressbar import progressbar


def save_ground_truth(df, filename="ground truth.p", at=5):
    """Saves the ground truth dictionary for a given dataframe containing labels."""

    ground_truth = {}
    total = len(df["srch_id"].unique())
    logs = {i:1/np.log2(i+2) for i in range(at)}

    grouped = df.groupby(by="srch_id")

    print("starting collection for ground truth...")

    for s, sub_df in progressbar(grouped, max_value=total):
        ground_truth[s] = {"properties": {}, "iDCG": 0}
        vals = []

        for _, row in sub_df.iterrows():
            prop = row["prop_id"]
            val = 0

            booked = int(row["booking_bool"])
            clicked = int(row["click_bool"])

            if clicked:
                val = 1

            if booked:
                val =  5

            ground_truth[s][prop] = val
            vals.append(val)

        vals.sort(reverse=True)
        ground_truth[s]["iDCG"] = sum(vals[i] * logs[i] for i in range(at))
    pk.dump(ground_truth, open(filename, 'wb'))
    print(f"Done, ground truth saved as pickle dump in {filename}")


if __name__ == '__main__':

    ROWS = 0
    t = time.time()
    if ROWS:
        df = pd.read_csv("training_set_VU_DM.csv", nrows=ROWS)
    else:
        df = pd.read_csv("training_set_VU_DM.csv")
    shape = df.shape
    print(f"loading {shape[0]} rows took {time.time() - t} seconds...")

    save_ground_truth(df)
