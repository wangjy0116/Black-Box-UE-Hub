import json
import os
from datasets import load_dataset
import random
from src import config


def build_hotpotqa_dataset(mode: str = "test", output_dir: str | None = None):
    if mode not in {"test", "dev"}:
        raise ValueError("mode must be 'test' or 'dev'")

    output_dir = output_dir or os.path.join(config.DATASET_DIR, "hotpotqa")
    split = "train" if mode == "dev" else "validation"
    data = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)

    seen_ids = set()
    unique_samples = []

    for sample in data:
        qid = sample["id"]
        if qid not in seen_ids:
            seen_ids.add(qid)
            unique_samples.append(sample)

    def format_sample(sample: dict):
        return {
            "id": sample["id"],
            "question": sample['question'],
            "description": "The answer must be in the form of short keywords.",
            "ground_truth": sample["answer"]
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
