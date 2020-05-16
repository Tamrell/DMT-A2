from copy import copy

hyperparameters = {
    "learning_rate" : 1e-4,
    "layers" : 4,
    "layer_size" : 80,
    "resnet" : True,
    "attention_layer_idx" : 1,  # -1 denotes no attention layer
    "lambda_batch_size": 150,
    "relu_slope" : 0.01,

    # Do not change these
    "epochs" : 2,
    "use_priors": True,
    "artificial_relevance": False,
    "exp_ver": False,
    "device" : None
}

hyperparameter_settings = {
    "layer_size" : [20, 320],
    "learning_rate": [1e-3, 1e-5],
    "layers": [2, 8],
    "resnet": [False],
    "attention_layer_idx": [-1, 0, 2, 3],  # -1 denotes no attention layer
    "relu_slope": [0.001, 0.1],
    "lambda_batch_size": [50, 450]
}


def generate_hyperparameters(order):
    yield hyperparameters
    for hyperparameter_name in order:
        hp = copy(hyperparameters)
        for setting in hyperparameter_settings[hyperparameter_name]:
            hp[hyperparameter_name] = setting
            yield hp


# Example usage
if __name__ == "__main__":
    # order can be different for different runs.
    order = ["layers", "layer_size", "resnet"]
    for _hyperparameters in generate_hyperparameters(order):
        print(_hyperparameters)
