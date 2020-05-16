from copy import copy

hyperparameters = {
    "epochs" : 10,
    "learning_rate" : 1e-4,
    "layers" : 4,
    "layer_size" : 80,
    "attention_layer_idx" : 1,  # -1 denotes no attention layer
    "resnet" : True,
    "lambda_batch_size": 150,
    "relu_slope" : 0.01,

    # Do not change these
    "use_priors": True,
    "artificial_relevance": False,
    "exp_ver": False,
    "device" : None
}


def generate_hyperparameters():
    yield hyperparameters

    LAYER_SIZE = [20, 320]
    for layer_size in LAYER_SIZE:
        hp = copy(hyperparameters)
        hp['layer_size'] = layer_size
        yield hp

    LEARNING_RATE = [1e-3, 1e-5]
    for learning_rate in LEARNING_RATE:
        hp = copy(hyperparameters)
        hp['learning_rate'] = learning_rate
        yield hp

    LAYERS = [2, 8]
    for layers in LAYERS:
        hp = copy(hyperparameters)
        hp['layers'] = layers
        yield hp

    ATTENTION_LAYER_IDX = [-1, 0, 2, 3]
    for attention_layer_idx in ATTENTION_LAYER_IDX:
        hp = copy(hyperparameters)
        hp['attention_layer_idx'] = attention_layer_idx
        yield hp

    RESNET = [False]
    for resnet in RESNET:
        hp = copy(hyperparameters)
        hp['resnet'] = resnet
        yield hp

    RELU_SLOPE = [0.001, 0.1]
    for relu_slope in RELU_SLOPE:
        hp = copy(hyperparameters)
        hp['relu_slope'] = relu_slope
        yield hp

    BATCH_SIZES = [50, 450]
    for b_size in BATCH_SIZES:
        hp = copy(hyperparameters)
        hp['lambda_batch_size'] = b_size
        yield hp
