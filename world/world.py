from __future__ import annotations
import json
import numpy as np
import random
import numpy as np
import os

from numpy._typing import NDArray
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from typing_extensions import override
from sprites import SpritesLoader
from utils import BACKGROUNDS, SHAPES, COLORS, SIZES, SHEENS, CATEGORIES, gaussian, load_image


class Entity:
    def __init__(self, id: str, position: Tuple[int, int], **kwargs):
        self.id = id
        self.position = position

    @classmethod
    def from_repr(cls, repr: Dict[str, Any]) -> Entity:
        assert len(repr["position"]) == 2
        position = (repr["position"][0], repr["position"][1])
        return cls(id=repr["id"], position=position)

    def __str__(self) -> str:
        properties = self.to_dict()
        properties_str = "\n".join([f"  {name}: {value}" for name, value in properties.items()])
        return f"{self.id}:\n{properties_str}"

    def __eq__(self, other) -> bool:
        eq = False
        if (
            isinstance(other, Entity) and
            self.id == other.id and
            self.position == other.position
        ):
            eq = True
        return eq

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class": self.__class__.__name__,
            "position": self.position
        }


class ObjEntity(Entity):

    SHAPES = SHAPES
    COLORS = COLORS
    SHEENS = SHEENS
    SIZES = SIZES

    def __init__(
            self,
            id: str,
            position: Tuple[int, int],
            area: Optional[str] = None,
            shape: Optional[str] = None,
            color: Optional[str] = None,
            sheen: Optional[str] = None,
            size: Optional[str] = None,
            **kwargs
    ):
        super().__init__(id, position)

        assert area != "none"
        self.area = area

        self.shape: str = shape #type: ignore
        if shape is None:
            self.shape = random.choice(ObjEntity.SHAPES)

        self.color: str = color #type: ignore
        if color is None:
            self.color = random.choice(ObjEntity.COLORS)

        self.sheen: str = sheen #type: ignore
        if sheen is None:
            self.sheen = random.choice(ObjEntity.SHEENS)

        self.size: str = size #type: ignore
        if size is None:
            self.size = random.choice(ObjEntity.SIZES)

    @classmethod
    @override
    def from_repr(cls, repr: Dict[str, Any]) -> ObjEntity:
        id = repr["id"]

        assert len(repr["position"]) == 2
        position = (repr["position"][0], repr["position"][1])

        assert repr["area"] is not None
        area = repr["area"]

        shape = repr["shape"]
        color = repr["color"]
        sheen = repr["sheen"]
        size = repr["size"]

        return cls(id=id, position=position, area=area, shape=shape, color=color, sheen=sheen, size=size)

    @override
    def __eq__(self, other) -> bool:
        eq = False
        if (
            isinstance(other, ObjEntity) and
            self.id == other.id and
            self.position == other.position and
            self.area == other.area and
            self.shape == other.shape and
            self.color == other.color and
            self.sheen == other.sheen and
            self.size == other.size
        ):
            eq = True
        return eq

    @override
    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()

        if self.area is not None:
            d["area"] = self.area

        d.update({
            "shape": self.shape,
            "color": self.color,
            "sheen": self.sheen,
            "size": self.size,
        })

        return d

class ImgEntity(Entity):

    CATEGORIES = CATEGORIES
    SIZES = SIZES

    def __init__(
            self,
            id: str,
            position: Tuple[int, int],
            area: Optional[str] = None,
            category: Optional[str] = None,
            size: Optional[str] = None,
            **kwargs
    ):
        super().__init__(id, position)

        assert area != "none"
        self.area = area

        self.category: str = category #type: ignore
        if category is None:
            self.category = random.choice(ImgEntity.CATEGORIES)

        self.size: str = size #type: ignore
        if size is None:
            self.size = random.choice(ImgEntity.SIZES)

    @classmethod
    @override
    def from_repr(cls, repr: Dict[str, Any]) -> ImgEntity:
        id = repr["id"]

        assert len(repr["position"]) == 2
        position = (repr["position"][0], repr["position"][1])

        assert repr["area"] is not None
        area = repr["area"]

        category = repr["category"]
        size = repr["size"]

        return cls(id=id, position=position, area=area, category=category, size=size)

    @override
    def __eq__(self, other) -> bool:
        eq = False
        if (
            isinstance(other, ImgEntity) and
            self.id == other.id and
            self.position == other.position and
            self.area == other.area and
            self.category == other.category and
            self.size == other.size
        ):
            eq = True
        return eq

    @override
    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()

        if self.area is not None:
            d["area"] = self.area

        d.update({
            "category": self.category,
            "size": self.size,
        })

        return d


ENTITY_CLASSES: Dict[str, type] = {
    "Entity": Entity,
    "ObjEntity": ObjEntity,
    "ImgEntity": ImgEntity,
}


class Node:

    def __init__(self, e: Entity):

        self.id = e.id
        self.e = e
        # node id, relation
        self.edges: Dict[str, Set[str]] = {}

    def __eq__(self, other) -> bool:
        eq = False

        if isinstance(other, Node) and self.id == other.id and self.e == other.e:
            edges_same = True
            if set(self.edges.keys()) == set(other.edges.keys()):
                for e_id, edge in self.edges.items():
                    if edge != other.edges[e_id]:
                        edges_same = False
                        break
            if edges_same:
                eq = True

        return eq

    def add_edge(self, n: Node, rel: str):
        if n.id not in self.edges:
            self.edges[n.id] = set()
        self.edges[n.id].add(rel)

    def _get_neighbours_by_rel(self, rel: str) -> List[str]:
        neighbours = set()
        for e_id, r in self.edges.items():
            if r == rel:
                neighbours.add(e_id)
        return list(neighbours)

    def get_neighbours(self, *rel) -> List[str]:

        neighbours = set(self.edges.keys())

        if len(rel) > 0:
            for r in rel:
                r_neigh = set(self._get_neighbours_by_rel(r))
                neighbours = neighbours.intersection(r_neigh)

        return list(neighbours)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            e_id: list(sorted(rels))
            for e_id, rels in self.edges.items()
        }
        return d

class Graph:

    def __init__(self):
        self.nodes: Dict[str, Node] = {}

    def __eq__(self, other: object) -> bool:
        eq = False
        if isinstance(other, Graph):
            nodes = True
            if set(self.nodes.keys()) != set(other.nodes.keys()):
                for n_id, n in self.nodes.items():
                    if n != other.nodes[n_id]:
                        nodes = False
                        break
            if nodes:
                eq = True
        return eq

    def add_node(self, e: Entity):
        new_node = Node(e)
        self.nodes[e.id] = new_node

    def add_edge(self, e1: Entity, e2: Entity, rel: str, rev_rel: Optional[str] = None):
        self.nodes[e1.id].add_edge(self.nodes[e2.id], rel)
        if rev_rel is not None:
            self.nodes[e2.id].add_edge(self.nodes[e1.id], rev_rel)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            e_id: node.to_dict()
            for e_id, node in self.nodes.items()
        }
        return d

class RelativePosGraph(Graph):
    def __init__(self):
        super().__init__()

    def update(self, e: Entity):
        self.add_node(e)

        for n2 in self.nodes.values():
            if n2.e.id == e.id:
                continue
            e2 = n2.e
            rels = self.get_relative_pos_rels(e.position, e2.position)
            for rel in rels:
                rev_rel = self._get_rev_relative_pos_rel(rel)
                self.add_edge(e, e2, rel=rel, rev_rel=rev_rel)

    def get_relative_pos_rels(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> List[str]:

        rels = []
        row1, col1 = pos1
        row2, col2 = pos2

        if row1 < row2:
            rels.append("N")
        elif row1 > row2:
            rels.append("S")

        if col1 < col2:
            rels.append("W")
        elif col1 > col2:
            rels.append("E")

        return rels

    def _get_rev_relative_pos_rel(self, rel: str) -> str:

        rev_rel = ""

        if rel == "N":
            rev_rel = "S"
        elif rel == "S":
            rev_rel = "N"
        elif rel == "E":
            rev_rel = "W"
        else: # rel == "W"
            rev_rel = "E"

        assert rel in ["N", "S", "E", "W"]

        return rev_rel

class RelativeSizeGraph(Graph):
    def __init__(self):
        super().__init__()

    def update(self, e: Entity):
        self.add_node(e)

        for n2 in self.nodes.values():
            if n2.e.id == e.id:
                continue
            e2 = n2.e
            rel = self.get_relative_size_rel(e.size, e2.size)
            rev_rel = self._get_rev_relative_size_rel(rel)
            self.add_edge(e, e2, rel=rel, rev_rel=rev_rel)

    def get_relative_size_rel(self, size1: str, size2: str) -> str:

        rel = "same"

        if size1 == "large" and size2 == "small":
            rel = "larger"
        elif size1 == "small" and size2 == "large":
            rel = "smaller"

        return rel

    def _get_rev_relative_size_rel(self, rel: str) -> str:

        rev_rel = "same"

        if rel == "larger":
            rev_rel = "smaller"
        elif rel == "smaller":
            rev_rel = "larger"

        return rev_rel

class AreaMap:

    def __init__(
        self,
        world: World,
        areas: List[List[str]] = [
            ["NW", "N", "NE"],
            ["W",  "C", "E"],
            ["SW", "S", "SE"],
        ]
    ):
        self.world = world
        self.areas = areas
        self.sup_size = (len(areas), len(areas[0]))
        assert self.sup_size[0] <= self.world.size[0] and self.sup_size[1] <= self.world.size[1]
        self.area_to_sup_pos: Dict[str, Tuple[int, int]] = {
            area: (sr, sc)
            for sr, ar_row in enumerate(areas)
            for sc, area in enumerate(ar_row)
        }
        self.area_positions: List[List[List[Tuple[int, int]]]] = self._init_area_positions()

    def __eq__(self, other) -> bool:
        eq = False

        if (
            isinstance(other, AreaMap) and
            self.sup_size == other.sup_size and
            self.area_to_sup_pos == other.area_to_sup_pos
        ):
            areas_same = True
            for area_row, other_area_row in zip(self.areas, other.areas):
                for area, other_area in zip(area_row, other_area_row):
                    if area != other_area:
                        areas_same = False
                        break

            if areas_same:
                eq = True

        return eq

    def _init_area_positions(self) -> List[List[List[Tuple[int, int]]]]:
        sup_rows, sup_cols = self.sup_size
        rows, cols = self.world.size

        positions: List[List[List[Tuple[int, int]]]] = []

        for sr in range(sup_rows):
            positions.append([])

            # divide rows into sup_rows parts
            sup_rows_start = round((rows/sup_rows)*sr)
            sup_rows_end = round((rows/sup_rows)*(sr+1))

            for sc in range(sup_cols):
                positions[sr].append([])

                # divide columns into sup_cols parts
                sup_cols_start = round((cols/sup_cols)*sc)
                sup_cols_end = round((cols/sup_cols)*(sc+1))

                positions[sr][sc] = [
                    (i, j)
                    for i in range(sup_rows_start, sup_rows_end)
                    for j in range(sup_cols_start, sup_cols_end)
                ]

        return positions

    def get_area(self, pos: Tuple[int, int]) -> str:

        area = None

        for i, ap_row in enumerate(self.area_positions):
            for j, positions in enumerate(ap_row):
                if pos in positions:
                    area = self.areas[i][j]
                    break

        assert area is not None, f"Position {pos} is not in any area."

        return area

    def get_area_positions(self, area: str) -> List[Tuple[int, int]]:

        sup_pos = self.area_to_sup_pos[area]
        sup_row, sup_col = sup_pos

        return self.area_positions[sup_row][sup_col]

    def get_areas_list(self) -> List[str]:
        return [ar for ar_row in self.areas for ar in ar_row]

    def to_dict(self) -> Dict[str, Any]:
        d = {}

        for i in range(len(self.area_positions)):
            for j in range(len(self.area_positions[0])):
                start = min(self.area_positions[i][j])
                end = max(self.area_positions[i][j])
                d[self.areas[i][j]] = {
                    "start": start,
                    "end": end
                }

        return d


class World:

    def __init__(
        self,
        size: Tuple[int, int] = (10,10),
        pos_std: float = 0.5,
        background: str = "none",
        areas: List[List[str]] = [
            ["NW", "N", "NE"],
            ["W",  "C", "E"],
            ["SW", "S", "SE"],
        ]
    ):
        self.pos_std = pos_std
        self.size: Tuple[int, int] = size

        self.background = "none"
        if background in BACKGROUNDS:
            self.background: str = background

        # world with fewer cell than areas -> only one area: center
        if self.size[0] < len(areas):
            areas = [["C"]]
        self.area_map = AreaMap(self, areas=areas)

        # special world entity
        w_ent = Entity("w", ((self.size[0]-1)/2, (self.size[1]-1)/2))
        self.ents: Dict[str, Entity] = {"w": w_ent}

        # Relative postion relations (entity-entity, entity-world)
        self.relative_pos_rels: RelativePosGraph = RelativePosGraph()
        self.relative_pos_rels.add_node(w_ent)

        # Relative size relation (entity-entity)
        self.relative_size_rels: RelativeSizeGraph = RelativeSizeGraph()

    @classmethod
    def from_repr(cls, repr: Dict[str, Any]) -> World:

        assert len(repr["size"]) == 2
        size = (repr["size"][0], repr["size"][1])
        background = repr["background"]
        world = cls(size=size, background=background)

        # entities
        for e_id, e in repr["ents"].items():
            e["id"] = e_id
            ent_class = ENTITY_CLASSES[e["class"]]
            new_ent: Entity = ent_class.from_repr(e)
            if new_ent.id == "w":
                assert world.ents["w"] == new_ent
            else:
                world.ents[e_id] = new_ent

        # relations
        for e_id, e in world.ents.items():
            world.relative_pos_rels.update(e)
            if e.id != "w":
                world.relative_size_rels.update(e)

        # area_map
        world.area_map = AreaMap(world)

        return world

    def __str__(self) -> str:
        ents_strs = []
        for e in self.ents.values():
            ents_strs.append(e.__str__())
        return "\n".join(ents_strs)

    def __eq__(self, other) -> bool:
        eq = False
        if (
            isinstance(other, World) and
            self.background == other.background and
            self.size == other.size and
            self.area_map == other.area_map and
            self.relative_pos_rels == other.relative_pos_rels and
            self.relative_size_rels == other.relative_size_rels
        ):
            # entities
            entities = True
            if set(self.ents.keys()) != set(other.ents.keys()):
                entities = False

            if entities:
                for e_id, e in self.ents.items():
                    if e != other.ents[e_id]:
                        entities = False
                        break
            if entities:
                eq = True
        return eq

    def _get_probs(self, start_pos: Tuple[float, float], valid_positions: List[Tuple[int, int]]):

        valid_pos_probs = []

        for valid_pos in valid_positions:
            valid_pos_probs.append(gaussian(valid_pos[0], valid_pos[1], m=start_pos, std=self.pos_std))

        for prob in valid_pos_probs:
            assert prob > 0

        tot = sum(valid_pos_probs)
        valid_pos_probs = [p/tot for p in valid_pos_probs]

        return valid_pos_probs

    def _get_valid_positions(self, area: Optional[str] = None) -> List[Tuple[int, int]]:

        all_positions = set([(i,j) for j in range(self.size[1]) for i in range(self.size[0])])
        # the world entity does not occupy a position
        occupied_positions = set([e.position for e in self.ents.values() if e.id != "w"])

        area_positions = set()
        if area is None:
            # whole world
            area_positions = all_positions
        else:
            # only specified area
            area_positions.update(self.area_map.get_area_positions(area))

        # free = (all \ occupied)
        free_positions = all_positions.difference(occupied_positions)
        # valid = area ^ free
        valid_positions = area_positions.intersection(free_positions)

        return sorted(list(valid_positions))

    def _get_positions(self, start_pos: Optional[Union[Tuple[float, float], str]], count: int = 1) -> List[Tuple[int, int]]:

        positions = []

        # separate area from position
        area = None
        if isinstance(start_pos, str):
            area = start_pos
            assert area in self.area_map.get_areas_list()
            start_pos = None

        # if an area is specified will consider only positions in that area
        valid_positions = self._get_valid_positions(area=area)

        if start_pos in valid_positions and count == 1:
            # use the position
            positions = [tuple(map(round, start_pos))]
        else:
            if start_pos == None:
                # uniform sample from area
                positions = random.sample(valid_positions, k=count)
            else:
                # either count > 1 or occupied or in between (e.g center in 10x10 grid) => sample around
                valid_pos_probs = self._get_probs(start_pos, valid_positions)
                positions_pos = np.random.choice(list(range(len(valid_positions))), size=count, replace=False, p=valid_pos_probs)
                positions = [valid_positions[pos] for pos in positions_pos]

        return positions

    def add(self, count: int, position: Optional[Union[List[float], str]] = None, **kwargs):

        if position is not None and not isinstance(position, str):
            # convert to Tuple[float, float]
            position = tuple(position) # type: ignore
            assert len(position) == 2 # type: ignore
        positions = self._get_positions(position, count=count) # type: ignore

        for pos in positions:
            id = f"e{len(self.ents)}"
            area = self.area_map.get_area(pos)
            if "category" in kwargs:
                self.ents[id] = ImgEntity(id, pos, area=area, **kwargs)
            else:
                self.ents[id] = ObjEntity(id, pos, area=area, **kwargs)

            self.relative_pos_rels.update(self.ents[id])
            self.relative_size_rels.update(self.ents[id])

    def get_stimulus(self, sprites_loader: SpritesLoader, img_size: int = 336) -> NDArray:

        rows, cols = self.size
        # approximate to floor
        cell_size = img_size//rows
        # consider missing pixels as extra padding
        img_size_missing = img_size - cell_size*rows

        # background
        # load background of correct size, add padding to the cells later
        stimulus = sprites_loader.get_background(self.background, (img_size, img_size))

        # entities
        for e_id, e in self.ents.items():
            if e_id == "w":
                continue
            sprite_size = cell_size
            offset = 0
            if isinstance(e, ObjEntity):
                if e.size == "small":
                    sprite_size = round(cell_size/2)
                    offset = round(cell_size/4)
                sprite = sprites_loader.get_object(e.shape, e.color, e.sheen, size=(sprite_size, sprite_size))
            elif isinstance(e, ImgEntity):
                if e.size == "small":
                    sprite_size = round(cell_size/2)
                    offset = round(cell_size/4)

                img_path = os.path.join("assets", "coco", f"{e.category}.png")
                sprite = load_image(img_path, size=(sprite_size, sprite_size), make_square=True, border_ratio=0.14)
            else:
                raise Exception(f"Entities of class {e.__class__} are not supported")

            # ignore when alpha channel is zero (keep background)
            # -1 is the last channel: alpha
            alpha_mask = np.ma.masked_greater(sprite[:,:,-1], 0).mask
            i, j = e.position

            # offset is zero for size="large"
            # if size="small" the shape will have 1/2 cell_size columns and rows, and the offset of cell_size/4 will put it in the center
            row_start = round(img_size_missing/2) + i*cell_size+offset
            row_end = row_start + sprite_size

            col_start = round(img_size_missing/2) + j*cell_size+offset
            col_end = col_start + sprite_size

            # select pixels with alpha > 0
            stimulus[row_start:row_end, col_start:col_end, :][alpha_mask] = sprite[:,:,:-1][alpha_mask]

        return stimulus

    def to_dict(self, img_path: Optional[str] = None) -> Dict[str, Any]:

        w_dict: Dict[str, Any] = {
            "size": self.size,
            "background": self.background
        }

        if img_path is not None:
            w_dict["img"] = img_path

        w_dict["ents"] = {}
        for e_id, e in self.ents.items():
            w_dict["ents"][e_id] = e.to_dict()

        w_dict["relations"] = {
            "relative": self.relative_pos_rels.to_dict(),
            "size": self.relative_size_rels.to_dict(),
        }

        w_dict["area_map"] = self.area_map.to_dict()

        return w_dict

    def to_json(self, path: str, img_path: Optional[str] = None):
        with open(path, "w") as f:
            json.dump(self.to_dict(img_path=img_path),f, indent=4)
