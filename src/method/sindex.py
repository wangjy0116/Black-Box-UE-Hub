from typing import Any, Dict, List, Optional, Tuple
import os
import re
import json
import numpy as np
from tqdm import tqdm
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from src.utils import compute_sample_cosine, build_sample_path, sample_generate, extract_final_answer
from src import config

class SINdex:
    def __init__(
        self,
        model_name,
        model, 
        dataset_name,
        dev_dataset,
        test_dataset,
        sample_num, 
        temperature,
        cluster_similarity_threshold: float = 0.95,
    ):
        self.cluster_similarity_threshold = float(cluster_similarity_threshold)
        self.cluster_distance_threshold = 1.0 - self.cluster_similarity_threshold
        self.model_name = model_name
        self.model = model
        self.dataset_name = dataset_name
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.sample_num = sample_num
        self.temperature = temperature
        self.save_path = f"{config.OUTPUT_DIR}/sindex/{dataset_name}/{model_name}.json"
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
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

    def hierarchical_cluster(
        self,
        cosine_matrix: np.ndarray,
        distance_threshold: float = 0.05,
    ) -> np.ndarray:

        dist_matrix = 1.0 - cosine_matrix
        dist_matrix = np.clip(dist_matrix, 0.0, 2.0)
        dist_matrix = (dist_matrix + dist_matrix.T) * 0.5
        np.fill_diagonal(dist_matrix, 0.0)
        condensed = squareform(dist_matrix, checks=False)

        Z = linkage(condensed, method="average")
        labels = fcluster(Z, t=distance_threshold, criterion="distance")
        return labels.astype(np.int32)

    def compute_cluster_cohesion(
        self,
        cosine_matrix: np.ndarray,
        indices: List[int],
    ) -> float:

        n = len(indices)
        if n <= 1:
            return 1.0
        sub = cosine_matrix[np.ix_(indices, indices)]
        vals = sub[np.triu_indices(n, k=1)]
        if vals.size == 0:
            return 1.0
        return float(np.mean(vals))

    def sindex_score(
        self,
        cosine_matrix: np.ndarray,
        cluster_labels: np.ndarray,
    ) -> float:
    
        unique_labels = sorted(set(cluster_labels.tolist()))
        score = 0.0

        for lb in unique_labels:
            idx = np.where(cluster_labels == lb)[0].tolist()
            p_k = len(idx) / cosine_matrix.shape[0]
            coh_k = self.compute_cluster_cohesion(cosine_matrix, idx)
            p_k_prime = p_k * coh_k

            if p_k_prime > 0:
                score -= p_k_prime * np.log(p_k_prime)

        return float(score)

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

    def calculate_scores(self, path: str):

        with open(path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        valid_num = self.sample_num // 2
        if "cosine_matrix" not in dataset[0].keys():
            for item in tqdm(dataset):
                question = item.get("question", "")
                sample_texts = []

                for idx, s in enumerate(item["sample_results"][:self.sample_num]):
                    ans = s.get("answer", None)
                    if ans is None:
                        text = s.get("text", "")
                        ans = extract_final_answer(text)
                        s["answer"] = ans

                    if ans is None:
                        continue
                    if not isinstance(ans, str):
                        ans = str(ans)

                    ans_strip = ans.strip()
                    if not ans_strip:
                        continue
                    sample_texts.append(ans_strip)

                cosine_matrix = compute_sample_cosine(
                    question=question,
                    sample_texts=sample_texts
                )

                item["cosine_matrix"] = cosine_matrix.tolist()

            with open(path, "w", encoding="utf-8") as f:
                json.dump(dataset, f, ensure_ascii=False, indent=4)
        
        sindex_scores = []
        for item in dataset:
            C = np.asarray(item["cosine_matrix"], dtype=np.float32)
            if C.ndim != 2 or C.shape[0] <= valid_num or C.shape[1] <= valid_num:
                sindex_scores.append(None)
            else:
                cluster_labels = self.hierarchical_cluster(
                    cosine_matrix=C,
                    distance_threshold=self.cluster_distance_threshold)
                sindex_scores.append(self.sindex_score(C, cluster_labels))

        return sindex_scores

    def calculate(self):

        sample_test_path, sample_dev_path = build_sample_path(self.save_path)
        sindex_scores_test = self.calculate_scores(sample_test_path)
        sindex_scores_dev = self.calculate_scores(sample_dev_path)

        with open(sample_test_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        
        output = []

        valid_scores_dev = [
            float(x) for x in sindex_scores_dev
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

        for item, sindex_score in zip(dataset, sindex_scores_test):
            if sindex_score is not None and item["greedy_results"]["answer"] is not None:
                sindex_score_norm = minmax(sindex_score)           
                conf = 1.0 - sindex_score_norm 
                output.append({"id": item["id"], "question": item["question"],
                    "description": item["description"], "ground_truth": item["ground_truth"],
                    "pred_answer": item["greedy_results"]["answer"],
                    "sindex_score": sindex_score, "pred_score": conf})
            else:
                output.append({"id": item["id"], "question": item["question"],
                    "description": item["description"], "ground_truth": item["ground_truth"],
                    "pred_answer": item["greedy_results"]["answer"],
                    "sindex_score": None, "pred_score": None})

            if self.dataset_name == "coqa":
                output[-1]["story"] = item["story"]


        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=4)


  
