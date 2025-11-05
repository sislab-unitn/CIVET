import re
from typing import List, Optional, Tuple

from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from transformers import CLIPProcessor


class COCODataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        img_folder: str = "images",
        ann_folder: str = "annotations",
        split: str = "val2017",
        valid_categories: Optional[List[str]] = None,
    ):
        self.annotation_file: str = f"{data_dir}/{ann_folder}/instances_{split}.json"
        self.image_folder: str = f"{data_dir}/{img_folder}/{split}"

        # initialize COCO api for instance annotations
        self.coco = COCO(annotation_file=self.annotation_file)

        coco_categories = self.coco.loadCats(self.coco.getCatIds())
        category_names: List[str] = [cat["name"] for cat in coco_categories]  # type: ignore

        if valid_categories is not None:
            valid_categories = [
                cat for cat in valid_categories if cat in category_names
            ]
        else:
            valid_categories = category_names

        # create the candidates
        self.candidates = ["a photo of a " + cat for cat in valid_categories]

        self.sample_ids: List[str] = []
        self.images: List[str] = []
        self.gts: List[str] = []
        self.bboxs: List[List[int]] = []

        for img_id in self.coco.getImgIds():
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            cats = self.coco.loadCats([ann["category_id"] for ann in anns])  # type: ignore
            img_file = self.coco.loadImgs(img_id).pop()["file_name"]  # type: ignore
            for i, (ann, cat) in enumerate(zip(anns, cats)):  # type: ignore
                # only consider objects in the categories
                if cat["name"] in valid_categories:
                    self.sample_ids.append(f"img-{img_id}-ent-{i}")
                    self.images.append(f"{self.image_folder}/{img_file}")
                    self.gts.append(cat["name"])
                    # convert the bbox to integers
                    bbox = list(map(round, ann["bbox"]))
                    # ensure the width and height are at least 1
                    x, y, w, h = bbox
                    w, h = max(w, 1), max(h, 1)
                    self.bboxs.append([x, y, w, h])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx) -> Tuple[str, str, List[int], str]:
        return self.images[idx], self.gts[idx], self.bboxs[idx], self.sample_ids[idx]


class COCOCollator:
    def __init__(self, processor: CLIPProcessor, candidates: List[str]):
        self.processor = processor
        self.candidates = candidates

    def __call__(self, batch: List[Tuple[str, str, List[int], str]]):
        images = []
        targets = []
        sample_ids = []
        for image, gt, bbox, sample_id in batch:
            image = Image.open(image)
            # unpack the bbox
            x, y, w, h = bbox
            # crop the image passing left, top, right, bottom
            cropped_image = image.crop((x, y, x + w, y + h))
            images.append(cropped_image)
            targets.append(gt)
            sample_ids.append(sample_id)

        inputs = self.processor(
            text=self.candidates,
            images=images,
            return_tensors="pt",
            padding=True,
            input_data_format="channels_last",
        )

        return inputs, targets, sample_ids


class SURECollator:
    def __init__(
        self,
        processor: CLIPProcessor,
    ):
        self.processor = processor

    def __call__(self, batch: List[Tuple[Image.Image, str, str, str]]):

        assert len(batch) == 1, "SURECollator only supports batch size of 1"
        image, question, answer, sample_id = batch[0]

        # exract the candidates
        candidates = re.search(r"\[.*\]", question)
        assert candidates is not None, f"candidates not found in '{question}'"
        candidates = candidates.group()[1:-1]
        assert (
            "[" not in candidates and "]" not in candidates
        ), f"squared brackets not removed from {question}"

        # remove the candidates from the question
        question = re.sub(r" Choose from \[.*\].", "", question)
        assert "Choose from" not in question, f"candidates not removed from {question}"

        # create a candidate by combining the question and the candidate
        texts = [
            f"a photo of a {candidate.strip()}" for candidate in candidates.split(", ")
        ]

        inputs = self.processor(
            text=texts,
            images=image,
            return_tensors="pt",
            padding=True,
            input_data_format="channels_last",
        )

        return inputs, texts, answer, sample_id
