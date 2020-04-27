from argparse import ArgumentParser
import os
import config


def parse_arguments():
    """Returns an variable which attributes are the arguments passed to call
    this program.
    """
    parser = ArgumentParser()

    # For development/testing
    parser.add_argument("--dev", help="run the code of the developers tag")

    return parser.parse_args()


def main(ARGS):
    if ARGS.dev:

        pass





if __name__ == '__main__':
    ARGS = parse_arguments()
    main(ARGS)
