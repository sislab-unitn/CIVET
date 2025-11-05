import json
import argparse

from PIL import Image

from sprites import SpritesLoader
from world import World


def get_args():
    parser = argparse.ArgumentParser(
        prog="python -m main",
        description="Generate test image",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "params_path",
        type=str,
        help="path to the json configuration file",
    )

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    with open(args.params_path, "r") as f:
        params = json.load(f)

    grid_size = tuple(params["size"])

    img_size = 100*grid_size[0]
    if "img_size" in params:
        img_size = params["img_size"]

    background = "none"
    if "background" in params:
        background = params["background"]

    objects = params["objects"]

    world = World(grid_size, background=background)

    for obj_template in objects:
        world.add(**obj_template)

    sprites_loader = SpritesLoader()
    s = world.get_stimulus(sprites_loader, img_size=img_size)
    Image.fromarray(s).save(f"test.png")
    world.to_json("test.json", img_path="test.png")

    # Test recreating from json
    with open("test.json", "r") as f:
        repr = json.load(f)
    w_from_json = World.from_repr(repr)
    assert w_from_json == world
    assert w_from_json.to_dict() == world.to_dict()

if __name__ == "__main__":
    main()
