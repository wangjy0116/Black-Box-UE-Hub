from typing import Any, Dict, List, Optional, Tuple
import os
import re
from src.utils import compute_sample_semantic, build_sample_path, sample_generate, extract_final_answer, norm_mc, load_nli_model
import numpy as np
import scipy.linalg
import copy
import json
import torch
from tqdm import tqdm
from src import config


class SE:
    def __init__(self, model_name, model, dataset_name, dev_dataset, test_dataset, sample_num, temperature):
        self.system_prompt = """Read the following question and reason step-by-step to formulate your final answer. {description}

Output ONLY a single JSON object with fields in this exact order:
{{
  "reasoning": "Your step-by-step analysis",
  "final_answer": "Your final answer"
}}"""
        self.model_name = model_name
        self.model = model
        self.dataset_name = dataset_name
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.sample_num = sample_num
        self.temperature = temperature
        self.save_path = f"{config.OUTPUT_DIR}/se/{dataset_name}/{model_name}.json"
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        self.show = True

    def build_system_prompt(self, item):
        return self.system_prompt.format(description=item["description"]) 

    def build_user_prompt(self, item: Dict[str, Any]) -> str:
        return f"Question: {item['question']}"

    def build_coqa_user_prompt(self, item):
        return f"Context: {item['story']}\nQuestion: {item['question']}"

    def se_score(self, class_mat, entail_id) -> float:
        C = class_mat
        m = C.shape[0]
        entail = (C == entail_id)
        mutual = entail & entail.T
        np.fill_diagonal(mutual, True)

        parent = list(range(m))
        rank = [0] * m

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            if rank[ra] < rank[rb]:
                parent[ra] = rb
            elif rank[ra] > rank[rb]:
                parent[rb] = ra
            else:
                parent[rb] = ra
                rank[ra] += 1

        for i in range(m):
            for j in range(i + 1, m):
                if mutual[i, j]:
                    union(i, j)

        comp_sizes = {}
        for i in range(m):
            r = find(i)
            comp_sizes[r] = comp_sizes.get(r, 0) + 1

        sizes = np.array(list(comp_sizes.values()), dtype=np.float64)
        p = sizes / max(float(sizes.sum()), 1e-12)

        p = np.clip(p, 1e-12, 1.0)
        se = float(-(p * np.log(p)).sum())
        return se, len(sizes)

    def generate(self, **kwargs: Any) -> List[Dict[str, Any]]:

        sample_test_path, sample_dev_path = build_sample_path(self.save_path)
        if self.dataset_name == "coqa":
            build_user=self.build_coqa_user_prompt
        else:
            build_user=self.build_user_prompt

        if not os.path.exists(sample_test_path):
            sample_generate(
                model=self.model,
                dataset=self.test_dataset,
                save_path=sample_test_path,
                build_system=self.build_system_prompt,
                build_user=build_user,
                sample_num=self.sample_num,
                temperature=self.temperature,
                **kwargs
            )
        if not os.path.exists(sample_dev_path):
            sample_generate(
                model=self.model,
                dataset=self.dev_dataset,
                save_path=sample_dev_path,
                build_system=self.build_system_prompt,
                build_user=build_user,
                sample_num=self.sample_num,
                temperature=self.temperature,
                **kwargs
            )

    def extract(self):

        def process_cache_file(cache_path: str) -> None:
            with open(cache_path, "r", encoding="utf-8") as f:
                dataset = json.load(f)

            for item in tqdm(dataset):
                samples = item.get("sample_results", [])
                for s in samples:
                    if "answer" not in s.keys():
                        text = s.get("text", "")
                        s["answer"] = extract_final_answer(text)
                        if self.dataset_name == "truthfulqa_mc":
                            s["answer"] = norm_mc(s["answer"], item)

                greedy = item.get("greedy_results", None)
                if "answer" not in greedy.keys():
                    text = greedy.get("text", "")
                    greedy["answer"] = extract_final_answer(text)
                    if self.dataset_name == "truthfulqa_mc":
                        greedy["answer"] = norm_mc(greedy["answer"], item)

            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(dataset, f, ensure_ascii=False, indent=4)

        sample_test_path, sample_dev_path = build_sample_path(self.save_path)
        process_cache_file(sample_test_path)
        process_cache_file(sample_dev_path)

    def calculate_scores(self, path):
        with open(path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        valid_num = min(self.sample_num // 2, 1)
        tokenizer, nli_model = load_nli_model(config.NLI_MODEL_PATH)

        if "class_mat_sc" not in dataset[0].keys():
            for item in tqdm(dataset):
                sample_texts = []
                for idx, s in enumerate(item["sample_results"][:self.sample_num]):
                    ans = s.get("answer", None)
                    if ans is None:
                        continue
                    if not isinstance(ans, str):
                        ans = str(ans)

                    ans_strip = ans.strip()
                    if not ans_strip:
                        continue
                    sample_texts.append(ans_strip)
                
                if self.dataset_name == "truthfulqa_mc":
                    class_mat = [[0] * len(sample_texts)]* len(sample_texts)
                    for i in range(len(sample_texts)):
                        for j in range(len(sample_texts)):
                            if sample_texts[i] == sample_texts[j]:
                                class_mat[i][j] = 0
                            else:
                                class_mat[i][j] = 2
                    item["class_mat_sc"] = class_mat
                    id_map = {"entail_id": 0, "contrary_id": 2}
                
                else:
                    question = item["question"]
                    E, C, N, E_logits, C_logits, N_logits, class_mat, id_map = compute_sample_semantic(
                        question=question,
                        sample_texts=sample_texts,
                        tokenizer=tokenizer,
                        nli_model=nli_model
                    )
                    item["class_mat_sc"] = class_mat.tolist()
                
            with open(path, "w", encoding="utf-8") as f:
                json.dump(dataset, f, ensure_ascii=False, indent=4)

        se_scores = []
        all_size = 0
        for item in dataset:
            C = np.asarray(item["class_mat_sc"])
            if C.ndim != 2 or C.shape[0] <= valid_num or C.shape[1] <= valid_num:
                se_scores.append(None)
            else:
                se_score, sizes = self.se_score(C, id_map["entail_id"])
                all_size += sizes
                se_scores.append(se_score)
        return se_scores

    def calculate(self):
        sample_test_path, sample_dev_path = build_sample_path(self.save_path)
        se_scores_test = self.calculate_scores(sample_test_path)
        se_scores_dev = self.calculate_scores(sample_dev_path)

        with open(sample_test_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        
        output = []

        valid_scores_dev = [
            float(x) for x in se_scores_dev
            if x is not None and not np.isnan(x)
        ]
        u_min = float(np.min(valid_scores_dev))
        u_max = float(np.max(valid_scores_dev))
        denom = (u_max - u_min)

        def minmax(x: float) -> float:
            if denom <= 1e-12:
                return 0.0
            v = (x - u_min) / denom
            if v < 0.0: v = 0.0
            if v > 1.0: v = 1.0
            return float(v)

        for item, se_score in zip(dataset, se_scores_test):
            if se_score is not None and item["greedy_results"]["answer"] is not None:
                se_score_norm = minmax(se_score)             
                conf = 1.0 - se_score_norm 
                output.append({"id": item["id"], "question": item["question"],
                    "description": item["description"], "ground_truth": item["ground_truth"],
                    "pred_answer": item["greedy_results"]["answer"],
                    "se_score": se_score, "pred_score": conf})
            else:
                output.append({"id": item["id"], "question": item["question"],
                    "description": item["description"], "ground_truth": item["ground_truth"],
                    "pred_answer": item["greedy_results"]["answer"],
                    "se_score": None, "pred_score": None})
            
            if "story" in item.keys():
                output[-1]["story"] = item["story"]


        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=4)


