import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Dict, List
import yaml
from src.dataset import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent

METHOD_CLASS_NAMES = {
    "cot": "CoT",
    "topk": "TopK",
    "vpd": "VPD",
    "ling": "Ling",
    "se": "SE",
    "selfcheck": "SelfCheckGPT",
    "eigv": "EigV",
    "kle": "KLE",
    "deg": "Deg",
    "ecc": "Ecc",
    "bsdetector": "BSDetector",
    "dinco": "DiNCo",
    "collab": "Collab",
    "argllms": "ArgLLMs",
    "t3": "T3",
    "uf": "UF",
    "cota": "COTA",
    "seu": "SEU",
    "snne": "SNNE",
    "steerconf": "SteerConf",
    "pathweight": "PathWeight",
    "spuq": "SPUQ",
    "inve": "InvE",
    "sindex": "SINdex",
    "sure": "SURE",
}

DOUBLE_MAX_TOKENS_DATASETS = {"gpqa_diamond"}

def _parse_list(values: List[str]):
    return [value.strip() for value in values if value and value.strip()]


def _load_method_config(path: str):
    if not path:
        return {}
    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as f:
        suffix = config_path.suffix.lower()
        if suffix in {".yaml", ".yml"}:
            data = yaml.safe_load(f)
        else:
            data = json.load(f)
    if data is None:
        return {}
    return data


def _load_method_class(method_name: str):
    class_name = METHOD_CLASS_NAMES[method_name]
    module = importlib.import_module(f"src.method.{method_name}")
    return getattr(module, class_name)

def run(model_name, model, dataset_name, dev_dataset, test_dataset, method_name: str, args, method_cfg: Dict[str, Any]):
    method_class = _load_method_class(method_name)
    init_kwargs = dict(method_cfg)
    other_kwargs = {
        "model_name": model_name,
        "model": model,
        "dataset_name": dataset_name,
        "dev_dataset": dev_dataset,
        "test_dataset": test_dataset}
    
    init_kwargs.update(other_kwargs)
    method = method_class(**init_kwargs)

    print(f"[RUN] model={model_name} | dataset={dataset_name} | method={method_name}")

    if "generate" in args.steps:
        method.generate()
    if "extract" in args.steps:
        method.extract()
    if "calculate" in args.steps:
        method.calculate()


def main(args) -> None:
    models = _parse_list(args.models)
    datasets = _parse_list(args.datasets)
    methods = [value.lower() for value in _parse_list(args.methods)]
    method_configs = _load_method_config(args.method_config)

    for model_name in models:
        for dataset_name in datasets:
            max_tokens = (
                args.max_tokens * 2
                if dataset_name in DOUBLE_MAX_TOKENS_DATASETS
                else args.max_tokens
            )
            if "generate" in args.steps:
                from src.model import load_model

                model = load_model(
                    model_name,
                    max_workers=args.max_workers,
                    max_tokens=max_tokens,
                )
                print(
                    f"[GENERATION] dataset={dataset_name} "
                    f"| max_tokens={max_tokens}"
                )
            else:
                model = None
            dev_dataset, test_dataset = load_dataset(dataset_name, args.dev_size, args.test_size)
            for method_name in methods:
                method_cfg = method_configs.get(method_name, {})
                run(model_name, model, dataset_name, dev_dataset, test_dataset, method_name, args, method_cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch run uncertainty methods over model/dataset combinations")
    parser.add_argument("--models", nargs="+", required=True, help="Model names to run")
    parser.add_argument("--datasets", nargs="+", required=True, help="Dataset names to run")
    parser.add_argument("--methods", nargs="+", required=True, help="Methods to run")
    parser.add_argument("--method_config", type=str, default="run_pipeline.methods.example.yaml", help="Path to a YAML or JSON file with per-method init/call config")
    parser.add_argument("--dev_size", type=int, default=50)
    parser.add_argument("--test_size", type=int, default=500)
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=8192,
        help="Base generation token limit; doubled automatically for GPQA Diamond.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=10,
        help="Maximum concurrent requests for OpenRouter and vLLM servers.",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["generate", "extract", "calculate"],
        default=["generate", "extract", "calculate"],
        help="Which pipeline steps to execute",
    )
    args = parser.parse_args()
    main(args)
