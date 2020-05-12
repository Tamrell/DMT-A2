from argparse import ArgumentParser
import os
import torch

from codebase import train


HYPERPARAMETERS = {
    "epochs" : 20,
    "learning_rate" : 1e-3,
    "layers" : 3,
    "layer_size" : 15,
    "attention_layer_idx" : 1,  # -1 denotes no attention layer
    "resnet" : False,

    # These hyperparameters are not in the commandline arguments.
    "device" : None,
    "relu_slope" : 0.01
}


def parse_arguments():
    """Returns an variable which attributes are the arguments passed to call
    this program.
    """
    parser = ArgumentParser()

    parser.add_argument("--train", action="store_true")
    for hyperparameter, default_value in HYPERPARAMETERS.items():
        parser.add_argument("--" + hyperparameter, type=type(default_value))

    parser.add_argument("--dummy", action="store_true")

    return parser.parse_args()


def set_device():
    global HYPERPARAMETERS

    # setting device on GPU if available, else CPU
    device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    HYPERPARAMETERS['device'] = device
    print('Using device:', device)
    print()

    # Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')


def main(ARGS):
    global HYPERPARAMETERS

    if ARGS.train:
        for key in HYPERPARAMETERS:
            value = eval(f"ARGS.{key}")
            assert value, f"missing value for {key}."
            HYPERPARAMETERS[key] = value
        train.train_main(HYPERPARAMETERS, "k_folds")

    else:
        assert ARGS.dummy, "If not train, then the dummy flaggy should be used. You dumdum"
        train.train_main(HYPERPARAMETERS, "dummy")


if __name__ == '__main__':
    ARGS = parse_arguments()
    set_device()
    main(ARGS)
