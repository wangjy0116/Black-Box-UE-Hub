import os
import json
from src import config

def _split_path(cache_dir: str, dataset_name: str, split: str):
    return os.path.join(os.path.join(cache_dir, dataset_name), f"{split}.json")

def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _build_dataset(dataset_name: str, cache_dir: str):
    dataset_cache_dir = os.path.join(cache_dir, dataset_name)
    if dataset_name == "coqa":
        from src.dataset.coqa import build_coqa_dataset
        build_coqa_dataset("dev", output_dir=dataset_cache_dir)
        build_coqa_dataset("test", output_dir=dataset_cache_dir)
    elif dataset_name == "hotpotqa":
        from src.dataset.hotpotqa import build_hotpotqa_dataset
        build_hotpotqa_dataset("dev", output_dir=dataset_cache_dir)
        build_hotpotqa_dataset("test", output_dir=dataset_cache_dir)
    elif dataset_name == "triviaqa":
        from src.dataset.triviaqa import build_triviaqa_dataset
        build_triviaqa_dataset("dev", output_dir=dataset_cache_dir)
        build_triviaqa_dataset("test", output_dir=dataset_cache_dir)
    elif dataset_name == "truthfulqa_mc":
        from src.dataset.truthfulqa_mc import build_truthfulqa_mc_dataset
        build_truthfulqa_mc_dataset(output_dir=dataset_cache_dir)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def load_dataset(dataset_name: str, dev_size: int, test_size: int):
    cache_dir = config.DATASET_DIR

    if dataset_name == "truthfulqa_mc":
        test_path = _split_path(cache_dir, dataset_name, "test")
        if not os.path.exists(test_path):
            _build_dataset(dataset_name, cache_dir)
        test_dataset = read_json(test_path)
        dev_dataset = test_dataset[:min(len(test_dataset), dev_size)]
        test_dataset = test_dataset[len(dev_dataset):min(len(test_dataset), test_size + dev_size)] 
    else:
        dev_path = _split_path(cache_dir, dataset_name, "dev")
        test_path = _split_path(cache_dir, dataset_name, "test")
        if not os.path.exists(dev_path) or not os.path.exists(test_path):
            _build_dataset(dataset_name, cache_dir)
        dev_dataset = read_json(dev_path)
        test_dataset = read_json(test_path)
        dev_dataset = dev_dataset[:min(len(dev_dataset), dev_size)]
        test_dataset = test_dataset[:min(len(test_dataset), test_size)]
    return dev_dataset, test_dataset
