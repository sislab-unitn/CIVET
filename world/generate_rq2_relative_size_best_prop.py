from collections import Counter
import json
import os
import argparse
import random

from itertools import chain, combinations, permutations, product
from typing import Any, List
from PIL import Image
from tqdm import tqdm
from questions import PROP_VALUES, relative_size_question
from sprites import SpritesLoader
from utils import SIZES, random_value_order
from world import World


def get_uniform_samples(val_list: List[Any], n: int) -> List[Any]:
    # get the same number of samples for each value
    samples_per_val = n//len(val_list)
    samples = list(chain(*[[var]*samples_per_val for var in val_list]))

    # sample remaining
    remaining = n - len(samples)
    rem_samples = random.sample(val_list, remaining)
    assert len(set(rem_samples)) == len(rem_samples)
    samples.extend(rem_samples)

    random.shuffle(samples)

    assert len(samples) == n
    # at most some values are sampled 1 additional time
    assert abs(max(Counter(samples).values()) - min(Counter(samples).values())) <= 1

    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str)
    parser.add_argument("--img_size", type=int, default=336)
    parser.add_argument("--size", type=int, default=9)
    args = parser.parse_args()

    os.makedirs(args.name, exist_ok=True)

    world_size = [args.size, args.size]

    sprites_loader = SpritesLoader()

    worlds_repr = {}
    questions = {}

    positions = [(i,j) for i in range(args.size) for j in range(args.size)]

    prop_variations = list(product(["triangle", "star"], ["yellow"], ["none"]))
    print(len(prop_variations))

    prop_variations = list(permutations(prop_variations, r=2))
    print(prop_variations)
    print(len(prop_variations))

    # add size
    size_pairs = list(product(SIZES, SIZES))
    prop_variations = list(product(prop_variations, size_pairs))
    print(len(prop_variations))

    # add positions
    pos_pairs = list(combinations(positions, r=2))
    prop_variations = list(product(prop_variations, pos_pairs))
    print(len(prop_variations))

    n_stimuli = len(prop_variations)


    question_properties = ["shape"]

    n_relative = len(PROP_VALUES["size"])
    relative_roders = random_value_order(n_stimuli, n_relative)

    for name, orders in zip(["area"], [relative_roders]):
        assert abs(max(Counter(orders).values()) - min(Counter(orders).values())) <= 1


    variations = []
    i = 0
    for v, relative_rod in zip(prop_variations, relative_roders):
        variation = [v, relative_rod]
        variations.append(tuple(chain(variation)))

    assert len(variations) == n_stimuli

    for stim_n, (variation, relative_rod) in tqdm(enumerate(variations), desc=f"Generating stimuli", unit="stimuli", total=len(variations)):

        params = {
            "size": world_size,
            "img_size": args.img_size,
            "background": "none",
            "objects": [],
        }

        prop_size_var, pos_var = variation
        prop_var, size_var = prop_size_var
        for (shape, color, sheen), size, position in zip(*[prop_var, size_var, pos_var]):
            params["objects"].append({
                "count": 1,
                "shape": shape,
                "color": color,
                "sheen": sheen,
                "size": size,
                "position": position
            })

        grid_size = tuple(params["size"])

        background = "none"
        if "background" in params:
            background = params["background"]

        objects = params["objects"]

        world = World(grid_size, background=background)

        for obj_template in objects:
            world.add(**obj_template)

        s = world.get_stimulus(sprites_loader, img_size=args.img_size)
        save_folder = os.path.join(args.name, "images")
        save_path = os.path.join(save_folder, f"{stim_n}.png")
        os.makedirs(save_folder, exist_ok=True)
        Image.fromarray(s).save(save_path)

        worlds_repr[stim_n] = world.to_dict(img_path=save_path)
        e1 = world.ents["e1"]
        e2 = world.ents["e2"]
        if stim_n not in questions:
            questions[stim_n] = {}
        if e1.id not in questions[stim_n]:
            questions[stim_n][e1.id] = {}
        if e2.id not in questions[stim_n]:
            questions[stim_n][e2.id] = {}
        # A w.r.t. B
        questions[stim_n][e1.id]["size_relative"] = relative_size_question(e1, e2, val_order=relative_rod, properties=question_properties)
        # B w.r.t. A
        questions[stim_n][e2.id]["size_relative"] = relative_size_question(e2, e1, val_order=relative_rod, properties=question_properties)

    with open(os.path.join(args.name, "repr.json"), "w") as f:
        json.dump(worlds_repr, f, indent=4)
    with open(os.path.join(args.name, "questions.json"), "w") as f:
        json.dump(questions, f, indent=4)

if __name__ == "__main__":
    main()
