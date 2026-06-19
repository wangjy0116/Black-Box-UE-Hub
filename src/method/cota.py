from typing import Any, Dict, List, Optional, Tuple
import re
import json
from tqdm import tqdm
import numpy as np
import ast
from src.utils import compute_sample_semantic, build_sample_path, sample_generate, extract_final_answer, load_nli_model, generate_steps
from src import config
import os

class COTA:
    def __init__(self, model_name, model, dataset_name, dev_dataset, test_dataset, sample_num, temperature):
        self.model_name = model_name
        self.model = model
        self.dataset_name = dataset_name
        self.dataset = test_dataset
        self.sample_num = sample_num
        self.temperature = temperature
        self.save_path = f"{config.OUTPUT_DIR}/cota/{dataset_name}/{model_name}.json"
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        self.show = True
        self.system_prompt = """Read the following question and reason step-by-step to formulate your final answer. {description}

Output ONLY a single JSON object with fields in this exact order:
{{
  "reasoning": "Your step-by-step analysis",
  "final_answer": "Your final answer"
}}"""

    def build_system_prompt(self, item):
        return self.system_prompt.format(description=item["description"]) 

    def build_user_prompt(self, item: Dict[str, Any]) -> str:
        return f"Question: {item['question']}"

    def build_coqa_user_prompt(self, item):
        return f"Context: {item['story']}\nQuestion: {item['question']}"

    def parse_cot(self, text: Any):
        if not isinstance(text, str):
            return None

        raw = text.strip()
        if not raw:
            return None

        def _normalize_reasoning(x: Any) -> Optional[str]:
            if x is None:
                return None
            if isinstance(x, str):
                s = x.strip()
                return s if s else None
            if isinstance(x, list):
                parts = [str(v).strip() for v in x if str(v).strip()]
                return "\n".join(parts) if parts else None
            s = str(x).strip()
            return s if s else None

        def _extract_from_obj(obj: Any) -> Optional[str]:
            if not isinstance(obj, dict):
                return None
            reasoning = obj.get("reasoning")
            if reasoning is None:
                reasoning = obj.get("cot")
            return _normalize_reasoning(reasoning)

        for parser in (json.loads, ast.literal_eval):
            try:
                obj = parser(raw)
            except Exception:
                continue

            reasoning = _extract_from_obj(obj)
            if reasoning is not None:
                return reasoning

            if isinstance(obj, str) and obj.strip():
                try:
                    obj2 = parser(obj.strip())
                except Exception:
                    continue
                reasoning = _extract_from_obj(obj2)
                if reasoning is not None:
                    return reasoning

        return None
        

    def mutual_entail(self, class_mat, entail_id: int, i: int, j: int) -> int:
        return int(class_mat[i, j] == entail_id and class_mat[j, i] == entail_id)

    def compute_cota(self, question, steps_a, steps_b, tokenizer, nli_model):
        steps_a = [s.strip() for s in (steps_a or []) if isinstance(s, str) and s.strip()]
        steps_b = [s.strip() for s in (steps_b or []) if isinstance(s, str) and s.strip()]
        na, nb = len(steps_a), len(steps_b)
        if na == 0 or nb == 0:
            return None

        all_steps = steps_a + steps_b
        _, _, _, _, _, _, class_mat, id_map = compute_sample_semantic(
            question=question,
            sample_texts=all_steps,
            tokenizer=tokenizer,
            nli_model=nli_model
        )
        entail_id = id_map["entail_id"]
        sum_a = 0
        for i in range(na):
            best = 0
            for j in range(nb):
                best = max(best, self.mutual_entail(class_mat, entail_id, i, na + j))
            sum_a += best

        sum_b = 0
        for j in range(nb):
            best = 0
            for i in range(na):
                best = max(best, self.mutual_entail(class_mat, entail_id, i, na + j))
            sum_b += best
        return float((sum_a + sum_b) / (na + nb))

    def generate(self, **kwargs: Any):
        sample_test_path, _ = build_sample_path(self.save_path)
        if self.dataset_name == "coqa":
            build_user=self.build_coqa_user_prompt
        else:
            build_user=self.build_user_prompt

        if not os.path.exists(sample_test_path):
            sample_generate(
                model=self.model,
                dataset=self.dataset,
                save_path=sample_test_path,
                build_system=self.build_system_prompt,
                build_user=build_user,
                sample_num=self.sample_num,
                temperature=self.temperature,
                **kwargs
            )

    def extract(self):
        sample_test_path, _ = build_sample_path(self.save_path)
        with open(sample_test_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        for item in tqdm(dataset):
            for sample in item.get("sample_results", []):
                if "answer" not in sample.keys():
                    text = sample.get("text", "")
                    sample["answer"] = extract_final_answer(text)
                if "reasoning" not in sample.keys():
                    text = sample.get("text", "")
                    cot = self.parse_cot(text)
                    sample["reasoning"] = cot

            if "answer" not in item["greedy_results"].keys():
                text = item["greedy_results"].get("text", "")
                item["greedy_results"]["answer"] = extract_final_answer(text)
            if "reasoning" not in item["greedy_results"].keys():
                text = item["greedy_results"].get("text", "")
                cot = self.parse_cot(text)
                item["greedy_results"]["reasoning"] = cot

        with open(sample_test_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)

    def calculate(self):
        sample_test_path, _ = build_sample_path(self.save_path)
        with open(sample_test_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        if "cots" not in dataset[0]["sample_results"][0]:
            dataset = generate_steps(dataset, self.sample_num)

        with open(sample_test_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)

        cota_scores = []
        valid_num = self.sample_num // 2
        tokenizer, nli_model = load_nli_model(config.NLI_MODEL_PATH)
        for item in tqdm(dataset):
            greedy_cots = item["greedy_results"]["cots"]
            if greedy_cots is None or greedy_cots == []:
                cota_scores.append(None)
                continue
            sum_score = 0.0
            cnt = 0
            for i in range(self.sample_num):
                sample_cots = item["sample_results"][i]["cots"]
                if sample_cots is not None and sample_cots != []:
                    sum_score += self.compute_cota(item["question"], greedy_cots, sample_cots, tokenizer, nli_model)
                    cnt+=1
            if cnt <= valid_num:
                cota_scores.append(None)
            else:
                cota_scores.append(sum_score/cnt)

        output = []
        for item, cota_score in zip(dataset, cota_scores):
            if cota_score is not None and item["greedy_results"]["answer"] is not None:
                output.append({"id": item["id"], "question": item["question"],
                    "description": item["description"], "ground_truth": item["ground_truth"],
                    "pred_answer": item["greedy_results"]["answer"],
                    "pred_score": cota_score})
            else:
                output.append({"id": item["id"], "question": item["question"],
                    "description": item["description"], "ground_truth": item["ground_truth"],
                    "pred_answer": item["greedy_results"]["answer"],
                    "pred_score": None})

            if self.dataset_name == "truthfulqa_mc":
                output[-1]["options"] = item["options"]
            if self.dataset_name == "coqa":
                output[-1]["story"] = item["story"]

        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=4)

        return dataset