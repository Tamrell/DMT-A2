from copy import copy
from codebase import train

hyperparameters = {
    "learning_rate" : 1e-4,
    "layers" : 4,
    "layer_size" : 80,
    "resnet" : True,
    "attention_layer_idx" : 1,  # -1 denotes no attention layer
    "lambda_batch_size": 150,
    "relu_slope" : 0.01,

    "uniform_relevance": False,
    "split_on_random_bool": False,
    "ndcg@5": False,

    # Do not change these
    "epochs" : 3,
    "artificial_relevance": False,
    "exp_ver": False,
    "device" : None,

    # Feature groups
    "use_priors": True,
    "normalize_per_subset": True,
    "datetime_shenanigans": True,
    "summarize_competitors": True,
    "travelling_within_country_bool": True,
    "occurrence conversion": True

}

hyperparameter_settings = {
    "layer_size" : [20, 320],
    "learning_rate": [1e-3, 1e-5],
    "layers": [2, 8],
    "resnet": [False],
    "attention_layer_idx": [-1, 0, 2, 3],  # -1 denotes no attention layer
    "relu_slope": [0.001, 0.1],
    "lambda_batch_size": [50, 450],

    "artificial_relevance": [True],
    "uniform_relevance": [True],
    "split_on_random_bool": [True],
    "ndcg@5": [True],

    "use_priors": [False],
    "normalize_per_subset": [False],
    "datetime_shenanigans": [False],
    "summarize_competitors": [False],
    "travelling_within_country_bool": [False],
    "occurrence conversion": [False]

}


def generate_hyperparameters(order = [param for param in hyperparameter_settings]):
    yield hyperparameters
    for hyperparameter_name in order:
        hp = copy(hyperparameters)
        for setting in hyperparameter_settings[hyperparameter_name]:
            hp[hyperparameter_name] = setting
            yield hp


# Example usage
if __name__ == "__main__":
    # order can be different for different runs.
    # order = [param for param in hyperparameter_settings]
    order = [param for param in hyperparameter_settings][::-1]
    for _hyperparameters in generate_hyperparameters(order):
        print(_hyperparameters)
        train.train_main(_hyperparameters, eval_last=True)
