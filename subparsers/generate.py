import argparse
import json
import os
import random
from functools import partial


import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, GenerationConfig

from utils.utils import SUREDataset, MAP_MODELS


def configure_subparsers(subparsers: argparse._SubParsersAction):
    """Configure a new subparser ."""
    parser = subparsers.add_parser(
        "generate",
        help="Generate text from a model on the test set of a specific dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--instruction",
        metavar="INSTR",
        type=str,
        default="",
        help="Instruction used for generation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        metavar="N_TOKENS",
        type=int,
        default=15,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--batch-size",
        metavar="BATCH_SIZE",
        type=int,
        default=2,
        help="Batch size for generation.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Use only 100 questions to check the answer of the model.",
    )

    parser.set_defaults(func=main)


def main(args):
    assert args.model_name != "CLIP", "This script is for text generation models only."
    # get the model and processor based on the model name
    model_name, Model, Collator = MAP_MODELS[args.model_name]

    # Fixed parameters (for now)
    top_k = 0
    top_p = 1.0
    temperature = 1.0

    model = Model(
        model_name,
        low_cpu_mem_usage=True,
        device_map="auto" if args.parallel else args.device,
    )

    # model config
    model.eval()

    Processor = partial(AutoProcessor.from_pretrained, model_name)
    if args.model_name == "qwen2-vl-7b":
        # as declared in the paper
        processor = Processor(
            min_pixels=100 * 28 * 28,
            max_pixels=16384 * 28 * 28,
        )
    elif args.model_name == "molmo-o-7b":
        processor = Processor(
            trust_remote_code=True,
        )
    else:
        processor = Processor()

    # Tok config
    processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id  # type: ignore
    processor.tokenizer.padding_side = "left"  # type: ignore

    # set the seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    generated_samples = []
    if os.path.exists(os.path.join(args.out_dir, f"generated_samples.json")):
        with open(os.path.join(args.out_dir, f"generated_samples.json"), "r") as f:
            generated_samples = json.load(f)

    test_ds = SUREDataset(args.data_folder, generated_samples, args.debug)
    test_loader = DataLoader(
        test_ds,
        batch_size=1 if args.model_name == "molmo-o-7b" else args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=Collator(
            processor=processor,  # type: ignore
            instruction=args.instruction,
        ),
    )

    generation_results = {}
    if os.path.exists(os.path.join(args.out_dir, f"generation_results.json")):
        with open(os.path.join(args.out_dir, f"generation_results.json"), "r") as f:
            generation_results = json.load(f)

    with torch.no_grad():
        for input_ids, _, targets, sample_ids in tqdm(
            test_loader, desc=f"Generating with P={top_p}, T={temperature}, K={top_k}"
        ):
            # add the generated samples to keep track of them
            generated_samples.extend(sample_ids)

            if args.model_name == "molmo-o-7b":
                # put the inputs on the correct device and make a batch of size 1
                input_ids = {
                    k: v.to(model.device).unsqueeze(0) for k, v in input_ids.items()
                }

                # cast the inputs to the right dtype
                input_ids["images"] = input_ids["images"].to(torch.bfloat16)

                output = model.generate_from_batch(
                    input_ids,
                    GenerationConfig(
                        max_new_tokens=args.max_new_tokens,
                        stop_strings="<|endoftext|>",  # suggested on MOLMo huggingface page
                        top_k=top_k,
                        temperature=temperature,
                        top_p=top_p,
                    ),
                    tokenizer=processor.tokenizer,
                )
                # discard the input part
                generated_tokens = output[0, input_ids["input_ids"].size(1) :]
                generated_answers = [
                    processor.tokenizer.decode(
                        generated_tokens, skip_special_tokens=True
                    )
                ]
            else:
                input_ids.to(model.device)
                output = model.generate(
                    **input_ids,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    max_new_tokens=args.max_new_tokens,
                )

                if "llava-next" in args.model_name:
                    output = output[:, input_ids.input_ids.size(-1) :]
                elif args.model_name == "qwen2-vl-7b":
                    output = [
                        out_ids[len(in_ids) :]
                        for in_ids, out_ids in zip(input_ids.input_ids, output)
                    ]

                generated_answers = processor.batch_decode(  # type: ignore
                    output, skip_special_tokens=True
                )

            for sample_id, pred, target in zip(sample_ids, generated_answers, targets):
                img_id, ent_id, *q_type = sample_id.split("-")

                if img_id not in generation_results:
                    generation_results[img_id] = {}

                if ent_id not in generation_results[img_id]:
                    generation_results[img_id][ent_id] = {}

                if len(q_type) == 2:
                    q_type, sub_q_type = q_type
                    if q_type not in generation_results[img_id][ent_id]:
                        generation_results[img_id][ent_id][q_type] = {}

                    generation_results[img_id][ent_id][q_type][sub_q_type] = {
                        "target": target,
                        "pred": pred,
                    }
                elif len(q_type) == 1:
                    # unpack it
                    q_type = q_type[0]
                    generation_results[img_id][ent_id][q_type] = {
                        "target": target,
                        "pred": pred,
                    }
                else:
                    raise ValueError(f"Invalid question type: {q_type}")

            with open(os.path.join(args.out_dir, f"generation_results.json"), "w") as f:
                json.dump(generation_results, f, indent=4)

            with open(os.path.join(args.out_dir, f"generated_samples.json"), "w") as f:
                json.dump(generated_samples, f, indent=4)
