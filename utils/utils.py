import json
import math
import os
import random
from argparse import Namespace
from functools import partial
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from peft.peft_model import PeftModel
from PIL import Image
from tqdm import tqdm, trange
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    CLIPModel,
    LlavaNextForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
)

from utils.llava_next import GenerationCollator as LLavaNextGenerationCollator
from utils.molmo import GenerationCollator as MOLMoGenerationCollator
from utils.qwen2_vl import GenerationCollator as Qwen2VLGenerationCollator
from utils.clip import SURECollator


MAP_MODELS = {
    "molmo-o-7b": (
        "allenai/Molmo-7B-O-0924",
        partial(
            AutoModelForCausalLM.from_pretrained,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,  # correct dtype is torch.float32, but performance is negligible
        ),
        MOLMoGenerationCollator,
    ),
    "llava-next-7b": (
        "llava-hf/llava-v1.6-vicuna-7b-hf",
        partial(
            LlavaNextForConditionalGeneration.from_pretrained,
            torch_dtype=torch.float16,
        ),
        LLavaNextGenerationCollator,
    ),
    "llava-next-13b": (
        "llava-hf/llava-v1.6-vicuna-13b-hf",
        partial(
            LlavaNextForConditionalGeneration.from_pretrained,
            torch_dtype=torch.float16,
        ),
        LLavaNextGenerationCollator,
    ),
    "qwen2-vl-7b": (
        "Qwen/Qwen2-VL-7B-Instruct",
        partial(
            Qwen2VLForConditionalGeneration.from_pretrained,
            torch_dtype=torch.bfloat16,
        ),
        Qwen2VLGenerationCollator,
    ),
    "CLIP": (
        "openai/clip-vit-large-patch14-336",
        CLIPModel,
        SURECollator,
    ),
}


class SUREDataset(Dataset):
    def __init__(self, data_folder: str, generated_samples, debug: bool = False):
        self.questions = []
        self.sample_ids = []
        self.answers = []
        self.images = []

        with open(os.path.join(data_folder, "questions.json"), "r") as f:
            data: Dict[str, Dict[str, Dict[str, Any]]] = json.load(f)

        if debug:
            valid_ids = random.sample(list(data.keys()), 100)

        for img_id, entities in data.items():
            # Skip images not in the set of valid ids when debugging
            if debug and img_id not in valid_ids:
                continue
            for ent_id, question_types in entities.items():
                for q_type, questions in question_types.items():
                    if isinstance(questions, dict):
                        for sub_q_type, sub_question in questions.items():
                            key = f"{img_id}-{ent_id}-{q_type}-{sub_q_type}"
                            if key not in generated_samples:
                                self.questions.append(sub_question)
                                self.sample_ids.append(key)
                                self.answers.append("")
                                self.images.append(
                                    os.path.join(data_folder, "images", f"{img_id}.png")
                                )
                    else:
                        key = f"{img_id}-{ent_id}-{q_type}"
                        if key not in generated_samples:
                            self.questions.append(questions)
                            self.sample_ids.append(key)
                            self.answers.append("")
                            self.images.append(
                                os.path.join(data_folder, "images", f"{img_id}.png")
                            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx) -> Tuple[Image.Image, str, str, str]:
        return (
            Image.open(self.images[idx]),
            self.questions[idx],
            self.answers[idx],
            self.sample_ids[idx],
        )

def generate(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    model.eval()  # type: ignore
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,  # type: ignore
            do_sample=True,
        )
        return outputs
