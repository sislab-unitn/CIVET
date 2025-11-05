import argparse
import json
import os
import random
import re


import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPProcessor

from utils.utils import MAP_MODELS, SUREDataset


VALID_MODELS = ["CLIP"]


def configure_subparsers(subparsers: argparse._SubParsersAction):
    """Configure a new subparser ."""
    parser = subparsers.add_parser(
        "classify",
        help="Classify using CLIP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Use only 100 questions to check the answer of the model.",
    )
    parser.add_argument(
        "--do-resize",
        action="store_true",
        help="Resize the image during pre-processing to match encoder dimension.",
    )
    parser.add_argument(
        "--do-center-crop",
        action="store_true",
        help="Take center crop of the image during pre-processing.",
    )

    parser.set_defaults(func=main)


def main(args):
    assert args.model_name in VALID_MODELS, f"This script is for {VALID_MODELS}."
    # get the model and processor based on the model name
    model_name, Model, Collator = MAP_MODELS[args.model_name]

    model = Model.from_pretrained(
        model_name, device_map="auto" if args.parallel else args.device
    )
    processor = CLIPProcessor.from_pretrained(
        model_name,
        do_resize=args.do_resize,  # resize shortest edge to 336
        do_center_crop=args.do_center_crop,  # take center crop to make it 336x336
        input_data_format="channels_last",  # input images are of shape (H, W, C)
    )

    # model config
    model.eval()

    # set the seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    classified_samples = []
    if os.path.exists(os.path.join(args.out_dir, f"classified_samples.json")):
        with open(os.path.join(args.out_dir, f"classified_samples.json"), "r") as f:
            classified_samples = json.load(f)

    test_ds = SUREDataset(args.data_folder, classified_samples, args.debug)
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=Collator(
            processor=processor,  # type: ignore
        ),
    )

    classification_results = {}
    if os.path.exists(os.path.join(args.out_dir, f"classification_results.json")):
        with open(os.path.join(args.out_dir, f"classification_results.json"), "r") as f:
            classification_results = json.load(f)

    with torch.no_grad():
        for input, candidates, target, sample_id in tqdm(
            test_loader, desc=f"Classifying Images"
        ):

            input.to(model.device)
            outputs = model(**input)
            logits_per_image = (
                outputs.logits_per_image
            )  # this is the image-text similarity score
            prob = logits_per_image.softmax(
                dim=1
            )  # we can take the softmax to get the label probabilities

            prob, pred = prob.cpu().max(dim=1)

            chosen_candidate = candidates[pred.item()]
            predicted_cat = chosen_candidate.replace("a photo of a ", "").strip()
            result_dict = {
                "target": target,
                "pred": predicted_cat,
                "prob": prob.item(),
            }

            img_id, ent_id, *q_type = sample_id.split("-")

            if img_id not in classification_results:
                classification_results[img_id] = {}

            if ent_id not in classification_results[img_id]:
                classification_results[img_id][ent_id] = {}

            if len(q_type) == 2:
                q_type, sub_q_type = q_type
                if q_type not in classification_results[img_id][ent_id]:
                    classification_results[img_id][ent_id][q_type] = {}

                classification_results[img_id][ent_id][q_type][sub_q_type] = result_dict
            elif len(q_type) == 1:
                # unpack it
                q_type = q_type[0]
                classification_results[img_id][ent_id][q_type] = result_dict
            else:
                raise ValueError(f"Invalid question type: {q_type}")

            # add the generated samples to keep track of them
            classified_samples.append(sample_id)

            with open(
                os.path.join(args.out_dir, f"classification_results.json"), "w"
            ) as f:
                json.dump(classification_results, f, indent=4)

            with open(os.path.join(args.out_dir, f"classified_samples.json"), "w") as f:
                json.dump(classified_samples, f, indent=4)
