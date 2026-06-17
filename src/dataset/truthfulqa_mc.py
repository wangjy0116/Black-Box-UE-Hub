import json
import os
from datasets import load_dataset
import random
from src import config


def build_truthfulqa_mc_dataset(output_dir: str | None = None):
    output_dir = output_dir or os.path.join(config.DATASET_DIR, "truthfulqa_mc")
    data = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")

    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def format_sample(sample: dict, index: int, rng: random.Random):
        choices = list(sample["mc1_targets"]["choices"])
        labels = list(sample["mc1_targets"]["labels"])

        assert len(choices) == len(labels), "choices/labels length mismatch"
        if 1 not in labels:
            return None

        correct_old_idx = labels.index(1)
        perm = list(range(len(choices)))
        rng.shuffle(perm)

        new_choices = [choices[i] for i in perm]
        new_correct_idx = perm.index(correct_old_idx)
        gt_letter = letters[new_correct_idx]

        options_lines = [f"{letters[i]}. {c}" for i, c in enumerate(new_choices)]
        options_str = "\n".join(options_lines)
        options_list = [{"label": letters[i], "text": c} for i, c in enumerate(new_choices)]

        return {
            "id": str(index),
            "question": sample["question"] + "\nOptions:\n" + options_str,
            "clean_question": sample["question"],
            "options": options_list,
            "options_num": len(choices),
            "description": "The answer must be in the form of a single option letter.",
            "ground_truth": gt_letter
        }

    formatted = []
    rng = random.Random(0)
    for idx, sample in enumerate(data):
        ex = format_sample(sample, idx, rng)
        if ex is not None:
            formatted.append(ex)

    random.seed(0)
    random.shuffle(formatted)

    output_path = os.path.join(output_dir, "test.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(formatted, f, ensure_ascii=False, indent=4)