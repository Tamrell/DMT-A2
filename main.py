from argparse import ArgumentParser
import os
import code.config


def parse_arguments():
    """Returns an variable which attributes are the arguments passed to call
    this program.
    """
    parser = ArgumentParser()

    return parser.parse_args()


def main(ARGS):
    pass


if __name__ == '__main__':
    ARGS = parse_arguments()
    main(ARGS)
