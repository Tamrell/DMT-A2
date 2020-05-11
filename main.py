from argparse import ArgumentParser
import os
from codebase import config
from codebase import train


def parse_arguments():
    """Returns an variable which attributes are the arguments passed to call
    this program.
    """
    parser = ArgumentParser()

    return parser.parse_args()


def main(ARGS):
    train.train_dummy()


if __name__ == '__main__':
    ARGS = parse_arguments()
    main(ARGS)
