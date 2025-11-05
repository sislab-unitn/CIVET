from io import BytesIO
from itertools import chain, permutations
import math
from cairosvg import svg2png
import numpy as np
import random

from PIL import Image
from numpy._typing import NDArray
from typing import List, Optional, Tuple, Union

COLOR_MAP = {
    "placeholder": (0, 255, 0, 255),
    "border": (255, 255, 255, 255),
    "transparent": (0, 0, 0, 0),
    "red": (255, 0, 0, 255),
    "green": (0, 255, 0, 255),
    "blue": (0, 0, 255, 255),
    "magenta": (255, 0, 255, 255),
    "yellow": (255, 255, 0, 255),
    "cyan": (0, 255, 255, 255)
}

COLOR_EXCLUDE = ["placeholder", "border", "transparent"]

SHAPES = ["square", "circle", "triangle", "star"]

COLORS = [color for color in COLOR_MAP.keys() if color not in COLOR_EXCLUDE]

SHEENS = ["matte", "glossy", "none"]

SIZES = ["large", "small"]

BACKGROUNDS = ["none", "grass"]

CATEGORIES = ["giraffe", "elephant", "zebra"]

def gaussian(r: int, c: int, m: Tuple[float, float] = (0,0), std: float = 1) -> float:
    mr, mc = m
    return 1/(2*math.pi) * 2*std * math.exp(-0.5 * ((r - mr)**2/std + (c - mc)**2/std))

def resize(img: Union[NDArray, Image.Image], size: Tuple[int, int], resample = Image.NEAREST) -> NDArray:
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    # invert: from (row, col) to (width, height)
    img = img.resize((size[1], size[0]), resample=resample)
    img = np.array(img)
    return img

def load_resized(path: str, size: Optional[Tuple[int, int]] = None) -> NDArray:
    img = Image.open(path)
    if size is not None:
        img = resize(img, size)
    else:
        img = np.array(img)
    return img

def change_color(sprite: NDArray, old_channel: Tuple[int, int, int, int], new_channels: Tuple[int, int, int, int]) -> NDArray:
    r,g,b,a = sprite.T
    copy = np.zeros(g.shape)
    for ch, factor in zip([r,g,b], old_channel[:-1]):
        copy += (ch*factor).astype(np.uint8)
    new_sprite_channels = []
    for factor in new_channels[:-1]:
        new_sprite_channels.append((factor*copy).astype(np.uint8))
    new_sprite_channels.append(a)
    sprite = np.array(new_sprite_channels).T
    return sprite

def load_image(path: str, size: Tuple[int, int] = (1200, 1200), border_ratio: float = 0.0, resample=Image.NEAREST, add_alpha: bool = True, make_square: bool = False) -> NDArray:
    row_size, col_size = size
    if path.endswith(".png"):
        img = np.array(Image.open(path))
        if img.shape[0] != row_size or img.shape[1] != col_size:
            if make_square:

                shape_size = tuple([round(s*(1-border_ratio)) for s in size])
                row_shape_size, col_shape_size = shape_size

                tmp = np.full((row_shape_size, col_shape_size, 4), [0,0,0,255]).astype(np.uint8)
                max_size = max(img.shape)
                row_size = round(row_shape_size*(img.shape[0]/max_size))
                col_size = round(row_shape_size*(img.shape[1]/max_size))
                assert row_size == row_shape_size or col_size == col_shape_size
                img = resize(img, (row_size, col_size), resample=resample)
                row_offset = (row_shape_size - row_size)//2
                col_offset = (col_shape_size - col_size)//2
                row_start = row_offset
                row_end = row_offset + row_size
                col_start = col_offset
                col_end = col_offset + col_size
                tmp[row_start:row_end,col_start:col_end,:-1] = img

                # borders
                img = np.full((size[0], size[1], 4), [0]*4).astype(np.uint8)
                row_offset, col_offset = tuple([round((s - shape_s) / 2) for s, shape_s in zip(size, shape_size)])
                img[row_offset:row_offset+row_shape_size, col_offset:col_offset+col_shape_size] = tmp
            else:
                img = resize(img, size, resample=resample)
    else: # svg
        with open(path, "rb") as f:
             svg_img = f.read()
        shape_size = tuple([round(s*(1-border_ratio)) for s in size])
        row_shape_size, col_shape_size = shape_size
        png_img = svg2png(bytestring=svg_img, output_width=col_shape_size, output_height=row_shape_size)
        assert png_img is not None
        shape_img = np.array(Image.open(BytesIO(png_img)))
        if shape_img.shape[-1] == 3 and add_alpha:
            # transparency was 255 on the whole image (square)
            # add alpha channel
            tmp = np.full((row_shape_size, col_shape_size, 4), [0,0,0,255]).astype(np.uint8)
            tmp[:,:,:-1] = shape_img
            shape_img = tmp
        channels = 3
        if add_alpha:
            channels = 4
        img = np.full((size[0], size[1], channels), [0]*channels).astype(np.uint8)
        row_offset, col_offset = tuple([round((s - shape_s) / 2) for s, shape_s in zip(size, shape_size)])
        img[row_offset:row_offset+row_shape_size, col_offset:col_offset+col_shape_size] = shape_img

    return img


def load_sprite_flat(path: str, size: Optional[Tuple[int, int]] = None, replace_colors: Optional[List[Tuple[Tuple[int, int, int ,int], Tuple[int, int, int, int]]]] = None, thresh: int = 20):
    img = Image.open(path)
    if replace_colors is not None:
        img = np.array(img)
        for old_color, new_color in replace_colors:
            substitute_rgba_values(img, old_color=old_color, new_color=new_color, thresh=thresh)
        img = Image.fromarray(img)
    if size is not None:
        img = resize(img, size)
    else:
        img = np.array(img)
    return img



def random_value_order(n_stimuli: int, n_positions: int) -> List[tuple]:
    # all order combinations
    pos = range(n_positions)
    perm = list(permutations(pos, r=len(pos)))

    # one object per stimulus -> one question for <property> per stimulus
    stim_per_perm = n_stimuli//len(perm)
    orders = list(chain(*[[perm]*stim_per_perm for perm in perm]))

    # sample remaining
    remaining = n_stimuli - len(orders)
    rem_perms = random.sample(perm, remaining)
    assert len(set(rem_perms)) == len(rem_perms)
    orders.extend(rem_perms)

    random.shuffle(orders)

    return orders

