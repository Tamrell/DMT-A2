from argparse import ArgumentParser
import os
import torch

from codebase import train


HYPERPARAM_DICT = {
    "epochs" : 2,
    "learning_rate" : 1e-3,
    "layers" : 3,
    "layer_size" : 5,
    "attention_layer_idx" : -1,  # -1 denotes no attention layer
    "resnet" : False,

    # These hyperparameters are not in the commandline arguments.
    "device" : None
}


def add_hyperparameters_arguments(parser):
    global HYPERPARAM_DICT

    for hyperparameter, default_value in HYPERPARAM_DICT.items():
        parser.add_argument("--" + hyperparameter, type=type(default_value))


def parse_arguments():
    """Returns an variable which attributes are the arguments passed to call
    this program.
    """
    parser = ArgumentParser()
    parser.add_argument("--train", type=bool)

    return parser.parse_args()


def set_device():
    global HYPERPARAM_DICT

    # setting device on GPU if available, else CPU
    device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    HYPERPARAM_DICT['device'] = device
    print('Using device:', device)
    print()

    # Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')


def main(ARGS):
    set_device()
    train.train_dummy(HYPERPARAM_DICT)


if __name__ == '__main__':
    ARGS = parse_arguments()
    main(ARGS)
