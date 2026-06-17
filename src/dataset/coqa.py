import json
import os
import random
from collections import OrderedDict
from src import config


def build_coqa_dataset(mode: str = "test", output_dir: str | None = None):
    if mode not in {"test", "dev"}:
        raise ValueError("mode must be 'test' or 'dev'")

    output_dir = output_dir or os.path.join(config.DATASET_DIR, "coqa")
    if mode == "test":
        source_path = f"{config.DATASET_DIR}/coqa_ori_dev.json"
    else:
        source_path = f"{config.DATASET_DIR}/coqa_ori_train.json"
    output_path = os.path.join(output_dir, f"{mode}.json")
    with open(source_path, "r") as file:
        data = json.load(file)["data"]

    dataset = []

    for sample in data:
        story = sample["story"]
        questions = sample["questions"]
        answers = sample["answers"]
        if mode != "dev":
            additional_answers = sample["additional_answers"]

        for question_index, question in enumerate(questions):
            additional_answers_list = []
            if mode != "dev":
                for i in range(3):
                    additional_answers_list.append(
                        additional_answers[str(i)][question_index]["input_text"]
                    )
            ground_truth = [answers[question_index]["input_text"]] + additional_answers_list
            unique_ground_truth = list(OrderedDict.fromkeys(ground_truth))
            dataset.append({
                "id": sample["id"]+"_"+str(question_index),
                "story": story,
                "question": question["input_text"],
                "description": "The answer must be in the form of short keywords or a sentence based on the context.",
                "ground_truth": unique_ground_truth,
            })
            story = story + ' Question: ' + question['input_text'] + ' Answer: ' + answers[question_index]['input_text']
            if not story[-1] == '.':
                story = story + '.'

    random.seed(0)
    random.shuffle(dataset)
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)