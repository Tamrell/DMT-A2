import os
import pandas as pd
import numpy as np
import torch
from matplotlib import pyplot as plt
import pickle
import json

# An exhaustive list of directories where result documents are saved.
MODEL_DIR = os.path.join("results", "models")
LOG_DIR = os.path.join("results", "logs")
VALIDATION_DIR = os.path.join("results", "validation_predictions")
TEST_DIR = os.path.join("results", "test_predictions")
HISTOGRAM_VAL_DIR = os.path.join("results", "histograms_validation")
HISTOGRAM_TRN_DIR = os.path.join("results", "histograms_train")
JSON_DIR = os.path.join("results", "jsons")

# Make sure to add any new directories you make to this loop.
for directory in (MODEL_DIR, LOG_DIR, VALIDATION_DIR, TEST_DIR,
                  HISTOGRAM_VAL_DIR, HISTOGRAM_TRN_DIR, JSON_DIR):
    if not os.path.exists(directory):
        os.makedirs(directory)

# The csv where the dataframe that maps models to hyperparameters is saved.
TRACKING_CSV = os.path.join("results", "io_tracking.wouw")
TRACKING_DF = None


def get_tracking_df(init=''):
    if not os.path.exists(TRACKING_CSV):
        user_input = init
        msg = "No tracking csv was found, do you want to initialize one? y/n "
        while user_input not in ('y', 'n'):
            user_input = input(msg).lower()

        if user_input == 'y':
            print("An empty tracking csv was created")
            df = pd.DataFrame({"model_id" : []}).astype({"model_id": "int32"})
            return df.set_index("model_id")

        else:
            print("The program cannot execute without a tracking csv")
            quit()

    return pd.read_csv(TRACKING_CSV, index_col=0)


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

def save_json(model_id, dictionary):
    with open(os.path.join(JSON_DIR, f"{model_id}.json"), 'w') as save_f:
        save_f.write(json.dumps(dictionary))

def load_json(model_id):
    """out: dict"""
    with open(os.path.join(JSON_DIR, f"{model_id}.json")) as load_f:
        return json.loads(load_f.read())

def load_jsons(model_ids=[]):
    """ Return the dataframe comprised of the json files assosiated with the
    input model ids, if no model ids are provided, use all stored json files.
    out: DataFrame"""
    df = pd.DataFrame()
    if len(model_ids) > 0:
        iterator = [f'{model_id}.json' for model_id in model_ids]
    else:
        iterator = os.listdir(JSON_DIR)
    for filename in iterator:
        model_id = int(filename[:-5])
        dictionary = load_json(model_id)
        dictionary['model_id'] = model_id
        df = df.append(dictionary, ignore_index=True)
    df = df.astype({'model_id': 'int32'})
    return df.set_index('model_id')

def save_model(model_id, model):
    with open(os.path.join(MODEL_DIR, f"{model_id}.pkl"), 'wb') as save_f:
        pickle.dump(model, save_f)

def load_model(model_id):
    """out: ExodiaNet"""
    with open(os.path.join(MODEL_DIR, f"{model_id}.pkl"), 'rb') as load_f:
        return pickle.load(load_f)

def save_val_predictions(model_id, pred_str):
    with open(os.path.join(VALIDATION_DIR, f"{model_id}.csv"), 'w') as save_f:
        save_f.write(pred_str)

def load_val_predictions(model_id):
    """out: DataFrame"""
    return pd.read_csv(os.path.join(VALIDATION_DIR, f"{model_id}.csv"))

def save_test_predictions(model_id, pred_str):
    with open(os.path.join(TEST_DIR, f"{model_id}.csv"), 'w') as save_f:
        save_f.write(pred_str)

def load_test_predictions(model_id):
    """out: DataFrame"""
    return pd.read_csv(os.path.join(TEST_DIR, f"{model_id}.csv"))

def save_histogram(model_id, val=False):
    if val:
        plt.savefig(os.path.join(HISTOGRAM_VAL_DIR, f"{model_id}.png"))
    else:
        plt.savefig(os.path.join(HISTOGRAM_TRN_DIR, f"{model_id}.png"))

def save_histogram_trn(model_id):
    save_histogram(model_id, val=False)

def save_histogram_val(model_id):
    save_histogram(model_id, val=True)


TRACKING_DF = get_tracking_df()


# Some example code
if __name__ == "__main__":
    hyperparam = {"is_good_model":True, "will_impress_Angelo_n_Reitze":False}
    model_id = add_model(hyperparam)

    # Note that you can have a model id as soon as you know the hyperparameters.
    model = "Very good predictions, also very impressed.csv"
    save_model(model_id, model)
