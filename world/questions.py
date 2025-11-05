from typing import Dict, List, Optional
from utils import CATEGORIES, COLORS, SHAPES, SHEENS
from world import Entity, ImgEntity, ObjEntity

PROPERTIES_ORDER = ("size", "sheen", "color", "shape")
PROPERTIES_ORDER_IMG = ("category")

TEMPLATES = {
    "properties": "{}What is the {} of the object? Choose from [{}].",
    "position_absolute": "Where is the {}? Choose from [{}].",
    "position_relative": "Where is the {} positioned with respect to the {}? Choose from [{}].",
    "size_relative": "Is the {} {} than the {}", # e.g. "Is the {red square} {smaller/larger} than the {blue circle}?"
    "distance_relative": "What is the closest object to the {}? Choose from [{}].",
    "size_relative": "What is the size of the {} with respect to the {}? Choose from [{}].",
}

# property values
PROP_VALUES = {
    "shape": SHAPES,
    "color": COLORS,
    "sheen": [s for s in SHEENS if s != "none"],
    "area": ["top left", "top center", "top right", "center left", "center", "center right", "bottom left", "bottom center", "bottom right"],
    "area_vert": ["left", "center", "right"],
    "area_hor": ["top", "center", "bottom"],
    "relative": ["above left", "directly above", "above right", "directly left", "directly right", "below left", "directly below", "below right"],
    "size": ["larger", "same", "smaller"],
    "category": CATEGORIES
}

PROP_DESCR = {
    "shape": "",
    "color": "",
    "sheen": "Sheen is a measure of the reflected light from a material. ",
    "category": "",
}

def get_object_reference(e: Entity, properties: List[str] = ["shape", "color", "sheen"]) -> str:
    values = []
    if isinstance(e, ObjEntity):
        values = [
           getattr(e, property)
            for property in PROPERTIES_ORDER
            if property in properties and getattr(e, property) != "none"
        ]
    elif isinstance(e, ImgEntity):
        assert len(properties) == 1 and properties[0] == "category"
        values = [getattr(e,properties[0])]
    return " ".join(values)

def get_val_list(val_order: Optional[List[int]], val_list_name: str) -> str:
    if val_order is None:
        val_list = PROP_VALUES[val_list_name]
    else:
        assert len(val_order) == len(PROP_VALUES[val_list_name])
        val_list = []
        for pos in val_order:
            val_list.append(PROP_VALUES[val_list_name][pos])
    return ", ".join(val_list)


def properties_questions(properties: List[str] = ["shape", "color", "sheen"]) -> Dict[str, str]:
    questions = {}
    template = TEMPLATES["properties"]

    for property in properties:
        questions[property] = prop_question(property)

    return questions

def prop_question(property: str, val_order: Optional[List[int]] = None) -> str:
    template = TEMPLATES["properties"]

    val_list_str = get_val_list(val_order, property)
    q = template.format(PROP_DESCR[property], property, val_list_str)

    return q


def absolute_position_question(e: Entity, properties: List[str] = ["shape", "color", "sheen"], val_order: Optional[List[int]] = None, val_list_name: str = "area") -> str:
    template = TEMPLATES["position_absolute"]

    reference = get_object_reference(e, properties)

    val_list_str = get_val_list(val_order, val_list_name)
    question = template.format(reference, val_list_str)

    return question

def relative_position_question(e1: Entity, e2: Entity, properties: List[str] = ["shape", "color", "sheen"], val_order: Optional[List[int]] = None) -> str:
    template = TEMPLATES["position_relative"]

    e1_ref = get_object_reference(e1, properties)
    e2_ref = get_object_reference(e2, properties)

    val_list_str = get_val_list(val_order, "relative")

    question = template.format(e1_ref, e2_ref, val_list_str)

    return question

def relative_distance_question(e1: Entity, ents: List[Entity], properties: List[str] = ["shape", "color", "sheen"], val_order: Optional[List[int]] = None) -> str:
    template = TEMPLATES["distance_relative"]

    e1_ref = get_object_reference(e1, properties)
    other_refs = []
    for e in ents:
        other_refs.append(get_object_reference(e, properties))

    if val_order is None:
        val_list = other_refs
    else:
        assert len(val_order) == len(other_refs)
        val_list = []
        for pos in val_order:
            val_list.append(other_refs[pos])
    val_list_str = ", ".join(val_list)

    question = template.format(e1_ref, val_list_str)

    return question


def relative_size_question(e1: Entity, e2: Entity, properties: List[str] = ["shape", "color", "sheen"], val_order: Optional[List[int]] = None) -> str:
    template = TEMPLATES["size_relative"]

    e1_ref = get_object_reference(e1, properties)
    e2_ref = get_object_reference(e2, properties)

    val_list_str = get_val_list(val_order, "size")

    question = template.format(e1_ref, e2_ref, val_list_str)

    return question
