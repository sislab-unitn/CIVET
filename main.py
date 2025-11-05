import argparse
import copy
import json
import os
from argparse import Namespace


import torch


from subparsers import generate, classify
from utils.utils import MAP_MODELS


def get_args() -> Namespace:
    """
    Parse command line arguments.

    Returns
    -------
    parsed_args: Namespace instance
        Parsed arguments passed through command line.
    """

    parser = argparse.ArgumentParser(
        prog="python -m main",
        description="Main module.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # arguments of the parser
    parser.add_argument(
        "model_name",
        metavar="MODEL_NAME",
        choices=MAP_MODELS.keys(),
        type=str,
        help="Name of the model to use.",
    )
    parser.add_argument(
        "data_folder",
        metavar="DATA_FOLDER",
        type=str,
        help="Folder containing train, validation, and test splits.",
    )
    parser.add_argument(
        "experiment_name",
        metavar="EXPERIMENT_NAME",
        type=str,
        help="Name of the experiment.",
    )
    parser.add_argument(
        "--device",
        metavar="DEVICE",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for generation.",
    )
    parser.add_argument(
        "--out-dir",
        metavar="OUT_DIR",
        type=str,
        default="output",
        help="Path to the output directory.",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Split the model across multiple GPUs.",
    )
    parser.add_argument(
        "--seed",
        metavar="SEED",
        type=int,
        default=42,
        help="Seed for reproducibility.",
    )

    subparsers = parser.add_subparsers(help="", dest="command")
    generate.configure_subparsers(subparsers)
    classify.configure_subparsers(subparsers)

    # parse arguments
    parsed_args = parser.parse_args()

    return parsed_args


if __name__ == "__main__":
    # get the arguments
    args = get_args()

    # set the output folder
    args.out_dir = os.path.join(args.out_dir, args.model_name, args.experiment_name)

    # create the output folder
    os.makedirs(args.out_dir, exist_ok=True)

    # save the arguments
    args_dict = copy.deepcopy(vars(args))
    del args_dict["func"]
    with open(os.path.join(args.out_dir, "args.json"), "w") as f:
        json.dump(args_dict, f, indent=4)

    args.func(args)
