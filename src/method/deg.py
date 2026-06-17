from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal, Tuple
import os
import re
import json
import copy
import torch 
import numpy as np
from tqdm import tqdm
from src.utils import compute_sample_semantic, build_sample_path, sample_generate, extract_final_answer, load_nli_model
from src import config

class Deg:

    def __init__(
        self,
        model_name,
        model, 
        dataset_name,
        dev_dataset,
        test_dataset,
        sample_num, 
        temperature,
        similarity_score: Literal["NLI_score", "Jaccard_score"] = "NLI_score",
        affinity: Literal["entail", "contra"] = "entail",
        jaccard_lower: bool = True,
    ):
        self.similarity_score = similarity_score
        self.affinity = affinity
        self.jaccard_lower = bool(jaccard_lower)
        self.model_name = model_name
        self.model = model
        self.dataset_name = dataset_name
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.sample_num = sample_num
        self.temperature = temperature
        self.save_path = f"{config.OUTPUT_DIR}/deg/{dataset_name}/{model_name}.json"
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        self.system_prompt = """Read the following question and reason step-by-step to formulate your final answer. {description}

Output ONLY a single JSON object with fields in this exact order:
{{
  "reasoning": "Your step-by-step analysis",
  "final_answer": "Your final answer"
}}"""

    def build_user(self, item: Dict[str, Any]) -> str:
        return f"Question: {item.get('question','')}"

    def build_user_prompt(self, item: Dict[str, Any]) -> str:
        return f"Question: {item['question']}"

    def build_coqa_user_prompt(self, item):
        return f"Context: {item['story']}\nQuestion: {item['question']}"


    def compute_jaccard_matrix(self, texts: List[str]) -> np.ndarray:
        n = len(texts)
        if n == 0:
            return np.zeros((0, 0), dtype=np.float32)

        def tokenize(s: str) -> set:
            if not isinstance(s, str):
                return set()
            s = s.strip()
            if self.jaccard_lower:
                s = s.lower()
            toks = [t for t in re.split(r"\s+", s) if t]
            return set(toks)

        toks = [tokenize(t) for t in texts]
        W = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                a, b = toks[i], toks[j]
                if not a and not b:
                    W[i, j] = 1.0
                    continue
                inter = len(a & b)
                union = len(a | b)
                W[i, j] = float(inter) / float(union) if union > 0 else 0.0
        return W

    def compute_degmat_uncertainty(self, W: np.ndarray) -> float:
        n = int(W.shape[0])
        if n == 0:
            return 0.0

        assert W.shape[0] == W.shape[1], f"W must be square, got {W.shape}"

        D = np.diag(W.sum(axis=1))
        return float(np.trace(n - D) / (n**2))


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

                greedy = item.get("greedy_results", None)
                if "answer" not in greedy.keys():
                    text = greedy.get("text", "")
                    greedy["answer"] = extract_final_answer(text)

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

        if self.similarity_score == "NLI_score" and "semantic_matrix_entail" not in dataset[0].keys():
            for item in tqdm(dataset):
                question = item.get("question", "")
                sample_texts = []
                for idx, s in enumerate(item.get("sample_results", [])):
                    ans = s.get("answer", None)
                    if ans is None:
                        continue
                    if not isinstance(ans, str):
                        ans = str(ans)
                    ans_strip = ans.strip()
                    if not ans_strip:
                        continue
                    sample_texts.append(ans_strip)

                E, C, N, E_logits, C_logits, N_logits, class_mat, id_map = compute_sample_semantic(
                    question=question,
                    sample_texts=sample_texts,
                    tokenizer=tokenizer,
                    nli_model=nli_model
                )

                item["semantic_matrix_entail"] = E.tolist()
                item["semantic_matrix_contra"] = C.tolist()

            with open(path, "w", encoding="utf-8") as f:
                json.dump(dataset, f, ensure_ascii=False, indent=4)

        deg_scores = []

        for item in dataset:
            if self.similarity_score == "NLI_score":
                E = np.asarray(item.get("semantic_matrix_entail", []), dtype=np.float32)
                C = np.asarray(item.get("semantic_matrix_contra", []), dtype=np.float32)

                if E.size <= valid_num or C.size <= valid_num or E.shape[0] != E.shape[1] or C.shape[0] != C.shape[1]:
                    deg_scores.append(None)
                    continue

                if self.affinity == "entail":
                    W = (E + E.T) / 2.0
                    u = float(self.compute_degmat_uncertainty(W))
                else:
                    W = 1.0 - C
                    W = (W + W.T) / 2.0
                    u = float(self.compute_degmat_uncertainty(W))
                deg_scores.append(u)

            else:
                texts = [(s.get("text") or "").strip() for s in item.get("sample_results", [])]
                W = self.compute_jaccard_matrix(texts)
                u = float(self.compute_degmat_uncertainty(W))
                deg_scores.append(u)

        return deg_scores

    def calculate(self) -> List[Dict[str, Any]]:
        sample_test_path, sample_dev_path = build_sample_path(self.save_path)
        deg_scores_dev = self.calculate_scores(sample_dev_path)
        deg_scores_test = self.calculate_scores(sample_test_path)

        with open(sample_test_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        output = []

        valid_scores_dev = [
            float(x) for x in deg_scores_dev
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

        for item, deg_score in zip(dataset, deg_scores_test):
            if deg_score is not None and item["greedy_results"]["answer"] is not None:
                deg_score_norm = minmax(deg_score)           
                conf = 1.0 - deg_score_norm 
                output.append({"id": item["id"], "question": item["question"],
                    "description": item["description"], "ground_truth": item["ground_truth"],
                    "pred_answer": item["greedy_results"]["answer"],
                    "deg_score": deg_score, "pred_score": conf})
            else:
                output.append({"id": item["id"], "question": item["question"],
                    "description": item["description"], "ground_truth": item["ground_truth"],
                    "pred_answer": item["greedy_results"]["answer"],
                    "deg_score": None, "pred_score": None})

            if "story" in item.keys():
                output[-1]["story"] = item["story"]

        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=4)

        return output
