import argparse
import os
import json
import shutil

from sprites import SpritesLoader
from world import World
from PIL import Image
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str)
    parser.add_argument("source_data", type=str)
    parser.add_argument("--img_size", type=int, default=336)
    args = parser.parse_args()

    repr_path = os.path.join(args.source_data, "repr.json")
    questions_path = os.path.join(args.source_data, "questions.json")

    with open(repr_path, "r") as f:
        source_repr = json.load(f)

    sprites_loader = SpritesLoader()
    save_folder = os.path.join(args.name, "images")
    os.makedirs(save_folder, exist_ok=True)
    for stim_n, world_repr in tqdm(source_repr.items(), desc=f"Regenerating stimuli with size {args.img_size}", unit="stimuli"):
        w = World.from_repr(world_repr)
        s = w.get_stimulus(sprites_loader, img_size=args.img_size)
        save_path = os.path.join(save_folder, f"{stim_n}.png")
        Image.fromarray(s).save(save_path)

    shutil.copy(questions_path, os.path.join(args.name, "questions.json"))
    shutil.copy(repr_path, os.path.join(args.name, "repr.json"))

if __name__ == "__main__":
    main()
