import os
import pandas as pd
import torch

import pickle

# An exhaustive list of directories where result documents are saved.
MODEL_DIR = os.path.join("results", "models")
LOG_DIR = os.path.join("results", "logs")
VALIDATION_DIR = os.path.join("results", "validation_predictions")
TEST_DIR = os.path.join("results", "test_predictions")

# Make sure to add any new directories you make to this loop.
for directory in (MODEL_DIR, LOG_DIR, PREDICTIONS_DIR):
    if not os.path.exists(directory):
        os.makedirs(directory)

# The csv where the dataframe that maps models to hyperparameters is saved.
TRACKING_CSV = os.path.join("results", "io_tracking.csv")
TRACKING_DF = None


def get_tracking_df(init=''):
    if not os.path.exists(TRACKING_CSV):
        user_input = init
        msg = "No tracking csv was found, do you want to initialize one? y/n "
        while user_input not in ('y', 'n'):
            user_input = input(msg).lower()

        if user_input == 'y':
            print("An empty tracking csv was created")
            return pd.DataFrame()

        else:
            print("The program cannot execute without a tracking csv")
            quit()

    return pd.read_csv(TRACKING_CSV)


def add_hyperparam(hyperparameter, default_value=None, save=True):
    """Add a new column to the tracking dataframe."""
    global TRACKING_DF
    TRACKING_DF = TRACKING_DF.assign(hyperparameter=lambda x: default_value)
    if save:
        TRACKING_DF.to_csv(TRACKING_CSV)


def add_model(hyperparam_dict, save=True):
    """Add the hyperparameters of the new model that will be tracked. Return
    the model id."""
    global TRACKING_DF
    TRACKING_DF = TRACKING_DF.append(hyperparam_dict, ignore_index=True)
    if save:
        TRACKING_DF.to_csv(TRACKING_CSV)
    return TRACKING_DF.index[-1]


def get_model_ids(hyperparam_dict):
    """Returns a list of al model ids that fit the input hyperparameters."""

    df = TRACKING_DF
    for hyperparameter, value in hyperparam_dict.items():
        df = df.loc[df[hyperparameter] == value]

    return list(df.index)


def save_model(model_id, model):
    with open(os.path.join(MODEL_DIR, f"{model_id}.pkl"), 'wb') as save_f:
        pickle.dump(model, save_f)


def load_model(model_id):
    with open(os.path.join(MODEL_DIR, f"{model_id}.pkl")) as load_f:
        return pickle.load(load_f)


def save_val_predictions(model_id, pred_str):
    with open(os.path.join(VALIDATION_DIR, f"{model_id}.csv"), 'w') as save_f:
        save_f.write(pred_str)

def save_test_predictions(model_id, pred_str):
    with open(os.path.join(TEST_DIR, f"{model_id}.csv"), 'w') as save_f:
        save_f.write(pred_str)


TRACKING_DF = get_tracking_df()


# Some example code
if __name__ == "__main__":
    hyperparam = {"is_good_model":True, "will_impress_Angelo_n_Reitze":False}
    model_id = add_model(hyperparam)

    # Note that you can have a model id as soon as you know the hyperparameters.
    model = "Very good predictions, also very impressed.csv"
    save_model(model_id, model)

