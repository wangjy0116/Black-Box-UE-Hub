import json
import os
from datasets import load_dataset
from collections import OrderedDict
import random

from src import config


def build_triviaqa_dataset(mode: str = "test", output_dir: str | None = None):
    if mode not in {"test", "dev"}:
        raise ValueError("mode must be 'test' or 'dev'")

    output_dir = output_dir or os.path.join(config.DATASET_DIR, "triviaqa")
    split = "validation" if mode == "test" else "train"
    data = load_dataset("trivia_qa", "rc.nocontext", split=split)

    seen_ids = set()
    unique_samples = []
    for sample in data:
        qid = sample["question_id"]
        if qid not in seen_ids:
            seen_ids.add(qid)
            unique_samples.append(sample)

    def format_sample(sample: dict):
        answers = [sample["answer"]["value"]] + sample["answer"]["aliases"]
        unique_answers = list(OrderedDict.fromkeys(answers))
        return {
            "id": sample["question_id"],
            "question": sample['question'],
            "description": "The answer must be in the form of short keywords.",
            "ground_truth": unique_answers
        }

    formatted = []
    for sample in unique_samples:
        formatted.append(format_sample(sample))

    random.seed(0)
    random.shuffle(formatted)

    output_path = os.path.join(output_dir, f"{mode}.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(formatted, f, ensure_ascii=False, indent=4)