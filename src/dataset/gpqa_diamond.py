import json
import os
from datasets import load_dataset
import random
from src import config


def build_gpqa_diamond_dataset(output_dir: str | None = None):
    output_dir = output_dir or os.path.join(config.DATASET_DIR, "gpqa_diamond")
    data = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")

    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def format_sample(sample: dict, index: int, rng: random.Random):
        choices = [
            sample["Correct Answer"],
            sample["Incorrect Answer 1"],
            sample["Incorrect Answer 2"],
            sample["Incorrect Answer 3"],
        ]

        perm = list(range(len(choices)))
        rng.shuffle(perm)

        new_choices = [choices[i] for i in perm]
        new_correct_idx = perm.index(0)
        gt_letter = letters[new_correct_idx]

        options_lines = [f"{letters[i]}. {c}" for i, c in enumerate(new_choices)]
        options_str = "\n".join(options_lines)
        options_list = [
            {"label": letters[i], "text": c}
            for i, c in enumerate(new_choices)
        ]

        formatted = {
            "id": sample.get("Record ID") or str(index),
            "question": sample["Question"] + "\nOptions:\n" + options_str,
            "clean_question": sample["Question"],
            "options": options_list,
            "options_num": len(choices),
            "description": "The answer must be in the form of a single option letter.",
            "ground_truth": gt_letter
        }

        if sample.get("High-level domain"):
            formatted["domain"] = sample["High-level domain"]
        if sample.get("Subdomain"):
            formatted["subdomain"] = sample["Subdomain"]
        if sample.get("Explanation"):
            formatted["reference_explanation"] = sample["Explanation"]

        return formatted

    formatted = []
    rng = random.Random(0)
    for idx, sample in enumerate(data):
        formatted.append(format_sample(sample, idx, rng))

    rng.shuffle(formatted)

    output_path = os.path.join(output_dir, "test.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(formatted, f, ensure_ascii=False, indent=4)
