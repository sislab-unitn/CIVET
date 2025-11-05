import argparse
import json
import os
import random
from argparse import Namespace


import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

from utils.clip import COCODataset, COCOCollator


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

    parser.add_argument(
        "coco_folder",
        metavar="COCO_FOLDER",
        type=str,
        help="Folder containing the COCO dataset.",
    )
    parser.add_argument(
        "experiment_name",
        metavar="EXPERIMENT_NAME",
        type=str,
        help="Name of the experiment.",
    )
    parser.add_argument(
        "--img-folder",
        metavar="IMG_FOLDER",
        type=str,
        default="images",
        help="Folder containing the COCO images.",
    )
    parser.add_argument(
        "--ann-folder",
        metavar="ANN_FOLDER",
        type=str,
        default="annotations",
        help="Folder containing the COCO annotations.",
    )
    parser.add_argument(
        "--split",
        metavar="DATASET_SPLIT",
        choices=["train2017", "val2017"],
        type=str,
        default="val2017",
        help="Dataset split to use.",
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
        "--categories",
        metavar="CATEGORY",
        type=str,
        nargs="+",
        default=None,
        help="List of allowed categories. None means all categories.",
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
    parser.add_argument(
        "--batch-size",
        metavar="BATCH_SIZE",
        type=int,
        default=2,
        help="Batch size for generation.",
    )

    # parse arguments
    parsed_args = parser.parse_args()

    return parsed_args


def main(args):
    model_name = "openai/clip-vit-large-patch14-336"
    model = CLIPModel.from_pretrained(
        model_name, device_map="auto" if args.parallel else args.device
    )
    processor = CLIPProcessor.from_pretrained(
        model_name,
        do_resize=True,  # resize shortest edge to 336
        do_center_crop=True,  # take center crop to make it 336x336
        input_data_format="channels_last",  # input images are of shape (H, W, C)
    )

    # model config
    model.eval()

    # set the seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = COCODataset(
        data_dir=args.coco_folder,
        img_folder=args.img_folder,
        ann_folder=args.ann_folder,
        split=args.split,
        valid_categories=args.categories,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=COCOCollator(
            processor=processor,  # type: ignore
            candidates=dataset.candidates,
        ),
    )

    classification_results = {}
    with torch.no_grad():
        for inputs, targets, sample_ids in tqdm(
            dataloader, desc=f"Classifying COCO Objects"
        ):
            inputs.to(model.device)
            outputs = model(**inputs)
            logits_per_image = (
                outputs.logits_per_image
            )  # this is the image-text similarity score
            probs = logits_per_image.softmax(
                dim=1
            )  # we can take the softmax to get the label probabilities

            probs, preds = probs.cpu().max(dim=1)

            for sample_id, pred, target, prob in zip(sample_ids, preds, targets, probs):
                chosen_candidate = dataset.candidates[pred.item()]
                predicted_cat = chosen_candidate.replace("a photo of a ", "")
                classification_results[sample_id] = {
                    "pred": predicted_cat,
                    "target": target,
                    "prob": prob.item(),
                }

            with open(
                os.path.join(args.out_dir, f"classification_results.json"), "w"
            ) as f:
                json.dump(classification_results, f, indent=4)


if __name__ == "__main__":
    # get the arguments
    args = get_args()

    # set the output folder
    args.out_dir = os.path.join(args.out_dir, "CLIP", args.experiment_name)

    # create the output folder
    os.makedirs(args.out_dir, exist_ok=True)

    # save the arguments
    args_dict = vars(args)
    with open(os.path.join(args.out_dir, "args.json"), "w") as f:
        json.dump(args_dict, f, indent=4)

    main(args)
