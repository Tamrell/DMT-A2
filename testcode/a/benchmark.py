import pandas as pd
import numpy as np
import pickle as pk
import os
import time
from progressbar import progressbar



def average_position_of_property(df, include_random=False, skip_5_and_friends=False, geometric=False):

    if skip_5_and_friends:
        to_replace = [5, 11, 17, 23]
        lowest = df["position"].max()
        for r in to_replace:
            df["position"] = df["position"].replace(r, lowest)

    if not include_random:
        if geometric:
            benchmark = np.log(df[df["random_bool"]==0][["prop_id", "position"]]).groupby(["prop_id"]).mean()
        else:
            benchmark = df[df["random_bool"]==0].groupby(["prop_id"]).mean()
    else:
        if geometric:
            benchmark = np.log(df[["prop_id", "position"]]).mean()
        else:
            benchmark = df.groupby(["prop_id"]).mean()
    benchmark = benchmark["position"]
    return benchmark


def average_clicks_of_property(df, include_random=True):

    if not include_random:
            benchmark = df[df["random_bool"]==0].groupby(["prop_id"]).mean()
    else:
        benchmark = df.groupby(["prop_id"]).mean()

    benchmark = -1 * benchmark["click_bool"].rename(columns={"click_bool":"position"})
    return benchmark

def average_bookings_of_property(df, include_random=True):

    if not include_random:
            benchmark = df[df["random_bool"]==0].groupby(["prop_id"]).mean()
    else:
        benchmark = df.groupby(["prop_id"]).mean()

    benchmark = -1 * benchmark["booking_bool"].rename(columns={"booking_bool":"position"})
    return benchmark



def save_bench_pred(df, benchmark, filename="benchmark V4 predictions.csv"):
    with open(filename, 'w') as f:
        f.write("srch_id,prop_id\n")
        known = set(benchmark)
        avg = benchmark.mean()

        grouped = df.groupby(by="srch_id")

        print("Doing predictions...")
        for s, sub_df in progressbar(grouped):
            rankings = []
            for p in list(sub_df["prop_id"]):
                if p in benchmark:
                    rankings.append((benchmark[p], p))
                else:
                    rankings.append((avg, p))

            for _, p in sorted(rankings):
                f.write(f"{s},{p}\n")
        print("Done")



def load_train(path=os.path.join("data","training_set_VU_DM.csv")):
    return pd.read_csv(path)

def load_test(path=os.path.join("data", "test_set_VU_DM.csv")):
    return pd.read_csv(path)




if __name__ == '__main__':

    t = time.time()
    df = load_train()
    shape = df.shape
    print(f"loading {shape[0]} rows took {time.time() - t} seconds...")

    t = time.time()
    test_df = load_test()
    # shape = test_df.shape
    print(f"loading {shape[0]} rows took {time.time() - t} seconds...")

    # to_know = test_df["prop_id"].unique()
    # all_known = df[df["random_bool"]==0]["prop_id"].unique()

    # print(len(set(all_known) & set(to_know)), len(set(to_know)))

    benchmark = average_bookings_of_property(df)
    # benchmark = average_clicks_of_property(df)
    # benchmark = average_position_of_property(df)

    # save_bench_pred(test_df, benchmark, filename="test benchmark V0.5 predictions.csv")

    filename = "benchmark V0.5 predictions.csv"
    save_bench_pred(df, benchmark, filename=filename)
