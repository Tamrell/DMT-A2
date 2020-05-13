import pandas as pd
import numpy as np
import os
import sys
import time
import torch
from codebase import benchmark

#===================== Dataset for the model training ==============

class BookingDataset():

    def __init__(self, fold="dummy"):
        """Class for holding a dataset.
            - Loads in segments to form a predefined fold (segment combination).
        """
        not_for_train = ["srch_id", "relevance", "random_bool", "prop_id"]

        path = os.path.join("data", "train_segments", "train_segment_") ###################### I/O
        self.fold=fold
        if fold == "dummy":
            print("\n\n!!! TRAINING ON THE DUMMY FOLD !!!\n\n")
            train_segments = [0]
            val_segment = 1
        elif fold == "full":
            print("\n\n!!! TRAINING ON THE FULL DATA!!!\n\n")
            train_segments = list(range(0, 10))
            val_segment = None
        elif fold == "test":
            print("\n\n!!! LOADED IN TEST DATA!!!\n\n")
            train_segments = None
            val_segment = None
        else:
            options = set(range(1, 11))
            assert (fold in options), f"Received illegal fold '{fold}' as input"

            train_segments = np.array(list(options - {fold})) - 1
            val_segment = fold - 1

        print(f"Fold: {fold}\nTrain segments: {train_segments}\nValidation segment: {val_segment}")
        print("Shuffling the deck...")#"\nPreparing Dataset...")
        if fold == "test":
            test_df = pd.read_csv(os.path.join("data", "test_preprocessed.csv"))
            test_df = shift_rescale_columns(test_df, not_for_train)
            self.search_no = len(test_df["srch_id"].unique())
            self.feature_no = test_df.shape[1] - len(not_for_train) + 1

        else:
            val_df = pd.read_csv(f"{path}{val_segment}.csv")           ###################### I/O
            train_df =  pd.read_csv(f"{path}{train_segments[0]}.csv")  ###################### I/O

            #################### TEST the shift-rescale ################
            val_df = shift_rescale_columns(val_df, not_for_train)
            train_df = shift_rescale_columns(train_df, not_for_train)
            #########################################


            for segment in train_segments[1:]:
                train_df = train_df.append(pd.read_csv(f"{path}{segment}.csv")) ################ I/O

            self.search_no = len(train_df["srch_id"].unique())
            self.feature_no = train_df.shape[1] - len(not_for_train)
        self.batches = {}
        self.props = {}
        self.relevances = {}
        self.rand_bools = {}
        self.val_batches = {}
        self.val_relevances = {}
        self.val_rand_bools = {}
        self.val_props = {}

        if fold == "test":
            # Precompute batches
            for s, sub_df in test_df.groupby("srch_id"):
                self.batches[s] = torch.from_numpy(sub_df.drop(columns=[t for t in not_for_train if t != "relevance" ]).values).float()
                self.rand_bools[s] = sub_df["random_bool"].tolist()[0]
                self.props[s] = torch.from_numpy(sub_df[["prop_id"]].values)
        else:
            # Precompute batches
            for s, sub_df in train_df.groupby("srch_id"):
                self.batches[s] = torch.from_numpy(sub_df.drop(columns=not_for_train).values).float()
                self.relevances[s] = torch.from_numpy(sub_df[["relevance"]].values).float()
                self.rand_bools[s] = sub_df["random_bool"].tolist()[0]
                self.props[s] = torch.from_numpy(sub_df[["prop_id"]].values)

            for s, sub_df in val_df.groupby("srch_id"):
                self.val_batches[s] = torch.from_numpy(sub_df.drop(columns=not_for_train).values).float()
                self.val_relevances[s] = torch.from_numpy(sub_df[["relevance"]].values).float()
                self.val_rand_bools[s] = sub_df["random_bool"].tolist()[0]
                self.val_props[s] = torch.from_numpy(sub_df[["prop_id"]].values)

            self.val_len = len(self.val_batches)

    def get_val(self, key):
        return key, self.val_batches[key], self.val_relevances[key], self.val_rand_bools[key], self.val_props[key]

    def validation_batch_iter(self):
        for i in self.val_batches.keys():
            yield self.get_val(i)

    def __getitem__(self, key):
        if self.fold == "test":
            return key, self.batches[key], self.rand_bools[key], self.props[key]

        # label, I mean we have the key and probably the ground truth... do we want this inside this module or outside?
        return key, self.batches[key], self.relevances[key], self.rand_bools[key], self.props[key]

    def __iter__(self):
        for i in np.random.permutation(list(self.batches.keys())):
            yield self[i]

    def __len__(self):
        """Assumes that each batch will contain all rows for a single search id"""
        return self.search_no

def shift_rescale_columns(df, to_skip):
    for c in df:
        if c not in to_skip:
            df[c] -= df[c].mean()
            df[c] /= df[c].std()
    return df


def preprocessing(train_path="", test_path=""):
    """Preprocesses the data after the first engineered features are added
        Args:
            - train_df (DataFrame): dataframe containing the training data
            - test_df (DataFrame): dataframe containing the test data
    """

    to_drop = ["visitor_hist_starrating"
              ,"visitor_hist_adr_usd"
              ,"srch_query_affinity_score"
              ]

    to_drop_exclusive = ["gross_bookings_usd"
                        ,"click_bool"
                        ,"booking_bool"
                        ,"position"
                        ]

    to_decide = []


    to_convert_on_occurrence = ["prop_country_id"
                               ,"visitor_location_country_id"
                               ,"srch_destination_id"
                               ,"site_id"
                               ,"prop_id"
                               ]

    to_fill = {"prop_review_score": "zero"
              ,"prop_location_score2": "median"
              }

    # to_rescale_and_shift = []

    print("Preprocessing training data...")

    t = time.time()
    train_df = load_train()

    # Add relevance column # NOT TO USE AS FEATURE!!!!!
    # train_df["relevance"] = train_df["click_bool"] + 4 * train_df["booking_bool"]

    prior_dict = get_prior_dict(train_df) # ONLY USE WHEN YOU NEED STRONGER PREDICTIONS IN THE ENDGAME #RELEASEtheBEAST

    # # Drop columns
    # train_df = train_df.drop(to_drop + to_drop_exclusive, axis=1)
    #
    # # Special fill distances
    # train_df = fill_distances(train_df)
    #
    # train_df = add_engineered_features(train_df, prior_dict)
    #
    # # Conversion to occurrence
    # for c in to_convert_on_occurrence:
    #     train_df = occurrence_based_conversion(train_df, c)
    #
    # # Skip to_decide
    # train_df = train_df.drop(to_decide, axis=1)
    #
    # # Fill NaNs
    # train_df["prop_review_score"] = train_df["prop_review_score"].fillna(0)
    # train_df["prop_location_score2"] = train_df["prop_location_score2"].fillna(train_df["prop_location_score2"].median())
    #
    # # Finish up by saving the train data
    # train_df = k_fold_segmentation(train_df)
    print(f"Done in {time.time()-t} seconds")

    ###========= Test Preprocessing =========###

    print("Preprocessing test data...")
    t = time.time()
    test_df = load_test()

    # Drop columns
    test_df = test_df.drop(to_drop, axis=1)

    # Add features
    test_df = add_engineered_features(test_df, prior_dict)

    # Special fill distances
    test_df = fill_distances(test_df)

    # Skip to_decide
    test_df = test_df.drop(to_decide, axis=1)

    # Fill NaNs
    test_df["prop_review_score"] = test_df["prop_review_score"].fillna(0)
    test_df["prop_location_score2"] = test_df["prop_location_score2"].fillna(train_df["prop_location_score2"].median())

    # Handle occurrence conversion
    for c in to_convert_on_occurrence:
        test_df = occurrence_based_conversion(test_df, c)

    print(f"Done in {time.time()-t} seconds")

    # train_df.to_csv(os.path.join("data", "train_preprocessed.csv"))
    test_df.to_csv(os.path.join("data", "test_preprocessed.csv"))

    return train_df, test_df

def add_engineered_features(df, prior_dict):

    # Include benchmarks
    df = assign_prior_information(df, prior_dict)

    # Summarize competitor information
    df = summarize_competitor_information(df)

    # add "travelling within country" ## ==== To discuss!!
    df["travelling_within_country_bool"] = (df["prop_country_id"] == df["visitor_location_country_id"]).astype(int)

    # Sine Cosine transform of date?
    df = add_sine_cosine(df)

    return df

def add_sine_cosine(df):
    sine = []
    cosine = []
    dates = pd.to_datetime(df["date_time"])
    for day in dates:
        total_days = 365
        if day.is_leap_year:
            total_days += 1

        daypos = 2 * np.pi / total_days * day.dayofyear
        sine.append(np.sin(daypos))
        cosine.append(np.cos(daypos))
    df["day sin"] = sine
    df["day cos"] = cosine
    df = df.drop(columns=["date_time"])
    return df

def get_prior_dict(train_df):
    """saves information based on the train data for the features as prior probabilities
        - average clicks per property ?
        - average position per property ?
        - average booking per property ?
    """
    prior_dict = {}

    # Average clicks
    prior_dict["clicks"] = benchmark.average_clicks_of_property(train_df).to_dict()

    # Average bookings
    prior_dict["bookings"] = benchmark.average_bookings_of_property(train_df).to_dict()

    # Average position at non-random
    prior_dict["position"] = benchmark.average_position_of_property(train_df).to_dict()

    return prior_dict

def assign_prior_information(df, prior_dict):

    avgpos = np.nanmean(list(prior_dict["position"].items()))
    for pos in prior_dict["position"]:
        if prior_dict["position"][pos]:
            pass
        else:
            prior_dict["position"][pos] = avgpos

    for p in prior_dict:
        colname = f"prior_information_{p}"
        df[colname] = df["prop_id"].map(prior_dict[p])
        df[colname] = df[colname].fillna(df[colname].mean())
    return df



def k_fold_segmentation(train_df, k=10, save_as_files=True):
    # TODO; shuffle all search_ids and make even-length folds over them.
    folds = {s: np.random.randint(0, k) for s in train_df["srch_id"].unique()}

    folds_list = []
    for _, row in train_df.iterrows():
        folds_list.append(folds[row["srch_id"]])
    train_df["fold_segment"] = folds_list
    #     for f in folds:
    #         print(len(f))

    if save_as_files:

        #### PATH #####
        os.mkdir(os.path.join("data", "train_segments"))
        #### PATH #####

        for i in range(k):
            to_save = train_df[train_df["fold_segment"]==i]

            #### PATH #####
            path = os.path.join("data", "train_segments", f"train_segment_{i}.csv")
            to_save.reset_index().drop(columns=["fold_segment", "index"]).to_csv(path, index=False)
            #### PATH #####
    return train_df


def occurrence_based_conversion(df, column):
    conversion_dict = {}
    count = df[column].value_counts()
    for i, (val, c) in enumerate(zip(count.index, count)):
        conversion_dict[val] = i+1
    if column == "prop_id":
        df["prop_occ"] = df[column].map(conversion_dict)
    else:
        df[column] = df[column].map(conversion_dict)
    return df


# =======================================
# ================ TODO =================
# =======================================

def zero_center(df, column, srch_level=False):
    df[column] -= df[column].mean()

def unit_variance(df, column, srch_level=False):
    df[column] /= df[column].std()

# =======================================


def fill_distances(df):
    """Fills missing distances per srch_id for 2 cases:
        - all distances are missing: mean average distance of the property
        - some distances are missing: mean distance to other properties
    """
    srch_split = df.groupby("srch_id")
    prop_avg_distances = {p: sub_df["orig_destination_distance"].mean() for p, sub_df in df.groupby("prop_id")}
    distances = {}

    for s, sub_df in srch_split:

        nans = sub_df["orig_destination_distance"].isnull()

        # Case: all are missing
        if nans.all():
            fill = np.nanmean([prop_avg_distances[p] for p in sub_df["prop_id"]])

            # Case: NOTHING IS KNOWN: fill mean distance or 0? consult team.
            if np.isnan(fill):
                fill = np.nanmean(list(prop_avg_distances.items()))

        # Case: some are missing
        elif nans.any():
            fill = sub_df["orig_destination_distance"].mean()

        else:
            fill = None

        distances[s] = fill

    distance_column = []
    for (_, r) in df.iterrows():
        s = r["srch_id"]
        if distances[s] is not None:
            distance_column.append(distances[s])
        else:
            distance_column.append(r["orig_destination_distance"])
    df["orig_destination_distance"] = distance_column
    return df


def summarize_competitor_information(df):
    """Summarizes competitor information in
       Min, Mean, Max values.
    """

    for c in range(1, 9):

        # Handle nans in competitor information

        # Combine the sign of rate with rate_percent_diff
        df[f"comp{c}_signed_rate_percent_diff"] = df[f"comp{c}_rate"] * df[f"comp{c}_rate_percent_diff"]

    diff_names = [f"comp{c}_signed_rate_percent_diff" for c in range(1, 9)]

    competitor_diff_df = df[diff_names]

    # If there are no competitors, there is no difference in rate and also no availability.

    # Add minimum of signed_rate_percent_diff between competors
    df["comp_diff_min"] = competitor_diff_df.min(axis=1).fillna(0)

    # Add maximum of signed_rate_percent_diff between competors
    df["comp_diff_max"] = competitor_diff_df.max(axis=1).fillna(0)

    # Add mean of signed_rate_percent_diff between competors
    df["comp_diff_mean"] = competitor_diff_df.mean(axis=1).fillna(0)

    # Add number of competitors that present this property
    df["comp_count"] = competitor_diff_df.count(axis=1).fillna(0)

    # Comp inv categorization + not boolean but count of number of competitors for the value?
    inv_names = [f"comp{c}_inv" for c in range(1, 9)]
    inv_df = df[inv_names].replace("nan", np.nan) # does this work though?
    inv_df = df[inv_names].replace("NULL", np.nan)
    # inv_allnans = inv_df.isnull().all(1)

    min_id = competitor_diff_df.idxmin(axis=1)
    max_id = competitor_diff_df.idxmax(axis=1)

    min_list = []
    max_list = []

    for min_, max_ , (_, vals) in zip(min_id, max_id, inv_df.iterrows()):
        if isinstance(min_, str):
            min_list.append(vals[f"{min_[:5]}_inv"])
            max_list.append(vals[f"{max_[:5]}_inv"])
        else:

            # Current NAN indicator = no room outside
            min_list.append(1)
            max_list.append(1)

    df["comp_min_inv"] = min_list
    df["comp_max_inv"] = max_list
    df["comp_mean_inv"] = inv_df.mean(axis=1).fillna(1)

    comp_names = [f"comp{c}_rate_percent_diff" for c in range(1, 9)]
    comp_rate_names = [f"comp{c}_rate" for c in range(1, 9)]
    df = df.drop(columns=diff_names+comp_names+inv_names+comp_rate_names)

    return df



def load_train(path=os.path.join("data","training_set_VU_DM.csv")):
    return pd.read_csv(path)

def load_test(path=os.path.join("data", "test_set_VU_DM.csv")):
    return pd.read_csv(path)


if __name__ == '__main__':
    preprocessing()
    # dataset = BookingDataset(5)
    #
    # # dataset = BookingDataset("dummy")
    # for i in dataset:
    #     input(i)
