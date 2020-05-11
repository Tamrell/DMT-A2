import os
import pandas as pd

# An exhaustive list of directories where result documents are saved.
MODEL_DIR = os.path.join("results", "models")
LOG_DIR = os.path.join("results", "logs")
PREDICTIONS_DIR = os.path.join("results", "predictions")

# Make sure to add any new directories to this tuple.
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


# TODO; Add this function in the init of the model class.
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


def save_predictions(model_id, pred_str):
    with open(os.path.join(PREDICTIONS_DIR, str(model_id) + ".pred"),
              'w') as pred_f:
        pred_f.write(pred_str)


def load_predictions(model_id):
    with open(os.path.join(PREDICTIONS_DIR, str(model_id) + ".pred")) as pred_f:
        return pred_f.readlines()


TRACKING_DF = get_tracking_df()


# Some example code
if __name__ == "__main__":
    hyperparam = {"is_good_model":True, "will_impress_Angelo_n_Reitze":False}
    model_id = add_model(hyperparam)

    # Note that you can have a model id as soon as you know the hyperparameters.
    pred_str = "Very good predictions, also very impressed.csv"
    save_predictions(model_id, pred_str)

