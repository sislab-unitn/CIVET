from collections import Counter
import json
import os
import argparse

from statistics import mean, stdev
from itertools import chain, product
from PIL import Image
from time import time_ns
from tqdm import tqdm
from questions import PROP_VALUES, absolute_position_question, prop_question
from sprites import SpritesLoader
from utils import SIZES, CATEGORIES, random_value_order
from world import World


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

    # property variations for one object
    prop_vairations = list(product(CATEGORIES, SIZES, positions))

    t1 = 0
    t2 = 0
    t3 = 0
    t4 = 0

    n_stimuli = len(prop_vairations)

    n_categories = len(CATEGORIES)
    category_orders = random_value_order(n_stimuli, n_categories)

    n_areas = len(PROP_VALUES["area"])
    n_areas_vert = len(PROP_VALUES["area_vert"])
    n_area_hors = len(PROP_VALUES["area_hor"])
    area_orders = random_value_order(n_stimuli, n_areas)
    area_vert_orders = random_value_order(n_stimuli, n_areas_vert)
    area_hor_orders = random_value_order(n_stimuli, n_area_hors)

    for name, orders in zip(
        ["category", "area", "area_vert", "area_hor"],
        [category_orders, area_orders, area_vert_orders, area_hor_orders]
    ):
        print("-"*40)
        print(name)
        c = Counter(orders)
        print(f"{mean(c.values())} +- {stdev(c.values())}, max-min: {max(c.values()) - min(c.values())}")


    variations = []
    i = 0
    for v, category_ord, area_ord, area_vert_ord, area_hor_ord in zip(prop_vairations, category_orders, area_orders, area_vert_orders, area_hor_orders):
        variation = [v, category_ord]
        variation.extend([area_ord, area_vert_ord, area_hor_ord])
        variations.append(tuple(chain(variation)))

    assert len(variations) == n_stimuli

    ref_properties = ["category"]

    for stim_n, (variation, category_ord, area_ord, area_vert_ord, area_hor_ord) in tqdm(enumerate(variations), desc=f"Generating stimuli", unit="stimuli", total=len(variations)):

        params = {
            "size": world_size,
            "img_size": args.img_size,
            "background": "none",
            "objects": [],
        }

        category, size, position = variation
        params["objects"].append({
            "count": 1,
            "category": category,
            "size": size,
            "position": position
        })

        grid_size = tuple(params["size"])

        img_size = 100*grid_size[0]
        if "img_size" in params:
            img_size = params["img_size"]

        background = "none"
        if "background" in params:
            background = params["background"]

        objects = params["objects"]

        start1 = time_ns()
        world = World(grid_size, background=background)
        t1 += (time_ns() - start1)/10**6

        for obj_template in objects:
            world.add(**obj_template)

        start2 = time_ns()
        s = world.get_stimulus(sprites_loader, img_size=args.img_size)
        t2 += (time_ns() - start2)/10**6
        start3 = time_ns()
        save_folder = os.path.join(args.name, "images")
        save_path = os.path.join(save_folder, f"{stim_n}.png")
        os.makedirs(save_folder, exist_ok=True)
        Image.fromarray(s).save(save_path)
        t3 += (time_ns() - start3)/10**6

        start4 = time_ns()
        worlds_repr[stim_n] = world.to_dict(img_path=save_path)
        # question about the first entity
        e = world.ents["e1"]
        if stim_n not in questions:
            questions[stim_n] = {}
        if e.id not in questions[stim_n]:
            questions[stim_n][e.id] = {}
        if "properties" not in questions[stim_n][e.id]:
            questions[stim_n][e.id]["properties"] = {}
        questions[stim_n][e.id]["properties"]["category"] = prop_question("category", category_ord)
        questions[stim_n][e.id]["position_absolute"] = absolute_position_question(e, val_order=area_ord, properties=ref_properties)
        questions[stim_n][e.id]["position_absolute_vert"] = absolute_position_question(e, val_order=area_vert_ord, val_list_name="area_vert", properties=ref_properties)
        questions[stim_n][e.id]["position_absolute_hor"] = absolute_position_question(e, val_order=area_hor_ord, val_list_name="area_hor", properties=ref_properties)
        t4 += (time_ns() - start4)/10**6

    start4 = time_ns()
    with open(os.path.join(args.name, "repr.json"), "w") as f:
        json.dump(worlds_repr, f, indent=4)
    with open(os.path.join(args.name, "questions.json"), "w") as f:
        json.dump(questions, f, indent=4)
    t4 = t4 / n_stimuli
    t4 += (time_ns() - start4)/10**6

    print(f"World: {t1/n_stimuli:.2f}")
    print(f"Stimulus: {t2/n_stimuli:.2f}")
    print(f"Stimulus/ojb: {t2/n_stimuli/1:.2f}")
    print(f"Save img: {t3/n_stimuli:.2f}")
    print(f"Save repr: {t4:.2f}")


if __name__ == "__main__":
    main()
