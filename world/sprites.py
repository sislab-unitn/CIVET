import os
import numpy as np
from typing import Dict, Optional, Tuple

from numpy._typing import NDArray

from utils import COLOR_MAP, change_color, load_image, resize


class SpritesLoader:

    def __init__(
        self,
        objects_folder: str = "./assets/shapes",
        backgrounds_folder: str = "./assets/backgrounds",
        materials_folder: str = "./assets/materials",
    ):
        self.objects_folder: str = objects_folder
        self.backgrounds_folder: str = backgrounds_folder
        self.materials_folder: str = materials_folder

        # shape color material size
        self.objects: Dict[str, Dict[str, Dict[str, Dict[Tuple[int, int], NDArray]]]] = {}

        self.backgrounds: Dict[str, Dict[Tuple[int, int], NDArray]] = {}

    def _apply_material(self, img: NDArray, material: NDArray) -> NDArray:
        # copy pixels only corresponding to the shape (green channel of img)
        r,g,b,a = img.T
        mask = g == 255
        mat_img = np.full(img.shape, [0,0,0,0]).astype(np.uint8)
        mat_img[mask.T] = material[mask.T]
        # apply img alpha to material
        mat_img[:,:,-1] = a.T
        return mat_img

    def _load_sprite(
        self,
        shape: str,
        color: str,
        material: str,
        size: Optional[Tuple[int, int]] = None,
        border_ratio: float = 0.14,
    ) -> NDArray:

        shape_path = os.path.join(self.objects_folder, f"{shape}.svg")
        # default to 1200x1200 (same as material size)
        img = load_image(shape_path, border_ratio=border_ratio)

        if material == "none":
            # apply color for no material
            old_color = COLOR_MAP["placeholder"]
            new_color = COLOR_MAP[color]
            img = change_color(img, old_color, new_color)
        else:
            # apply color-material when material is not "none"
            # default to 1200x1200 (same as material size)
            material_img = load_image(os.path.join(self.materials_folder, f"{color}_{material}.png"))
            img = self._apply_material(img, material_img)

        if size is not None:
            img = resize(img, size)

        return img


    def get_object(self, shape: str, color: str, material: str, size: Tuple[int, int]) -> NDArray:
        if shape not in self.objects:
            self.objects[shape] = {}
        if color not in self.objects[shape]:
            self.objects[shape][color] = {}
        if material not in self.objects[shape][color]:
            self.objects[shape][color][material] = {}
        if size not in self.objects[shape][color][material]:
            self.objects[shape][color][material][size] = self._load_sprite(
                shape=shape,
                color=color,
                material=material,
                size=size,
            )
        return self.objects[shape][color][material][size].copy()


    def get_background(self, background: str, size: Tuple[int, int]) -> NDArray:
        if background not in self.backgrounds:
            self.backgrounds[background] = {}
        if size not in self.backgrounds[background]:
            bg_path = os.path.join(self.backgrounds_folder, f"{background}.svg")
            if not os.path.isfile(bg_path):
                bg_path = os.path.join(self.backgrounds_folder, f"{background}.png")
            self.backgrounds[background][size] = load_image(bg_path, size=size, add_alpha=False)

        return self.backgrounds[background][size].copy()

