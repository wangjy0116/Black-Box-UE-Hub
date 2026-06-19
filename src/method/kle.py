from typing import Any, Dict, List, Optional, Tuple
import os
import re
from src.utils import compute_sample_semantic, build_sample_path, sample_generate, extract_final_answer, load_nli_model
import numpy as np
import scipy.linalg
import json
from tqdm import tqdm
import scipy
import numpy as np
from numpy.linalg import matrix_power as mp
from src import config

class KLE:
    def __init__(
        self,
        model_name,
        model, 
        dataset_name,
        dev_dataset,
        test_dataset,
        sample_num, 
        temperature,
        t: float = 0.3,
        normalize: bool = True,
        scale: bool = True,
        jitter: float = 0.0
    ):
        self.t = float(t)
        self.normalize = bool(normalize)
        self.scale = bool(scale)
        self.jitter = float(jitter)
        self.model_name = model_name
        self.model = model
        self.dataset_name = dataset_name
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.sample_num = sample_num
        self.temperature = temperature
        self.save_path = f"{config.OUTPUT_DIR}/kle/{dataset_name}/{model_name}.json"
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



    @staticmethod
    def laplacian_matrix(weighted_graph: np.ndarray) -> np.ndarray:
        degrees = np.diag(np.sum(weighted_graph, axis=0))
        return degrees - weighted_graph

    @staticmethod
    def heat_kernel(laplacian: np.ndarray, t: float) -> np.ndarray:
        return scipy.linalg.expm(-t * laplacian)

    @staticmethod
    def normalize_kernel(K: np.ndarray) -> np.ndarray:
        eps = 1e-12
        diagonal_values = np.sqrt(np.diag(K)) + eps
        return K / np.outer(diagonal_values, diagonal_values)

    @staticmethod
    def scale_entropy(entropy: float, n_classes: int) -> float:
        max_entropy = -np.log(1.0 / n_classes)
        return float(entropy) / float(max_entropy)

    def vn_entropy(self, K: np.ndarray) -> float:
        if self.normalize:
            K = self.normalize_kernel(K) / K.shape[0]

        try:
            eigvs = np.linalg.eig(K + self.jitter * np.eye(K.shape[0])).eigenvalues.astype(np.float64)
        except AttributeError:
            eigvs = np.linalg.eig(K + self.jitter * np.eye(K.shape[0]))[0].astype(np.float64)

        result = 0.0
        for e in eigvs:
            if np.abs(e) > 1e-8:
                result -= float(e) * float(np.log(e))

        if self.scale:
            result = self.scale_entropy(result, K.shape[0])

        return float(result)
 
    def kle_score(self, entail, contra, class_mat) -> float:
        e = (entail + entail.T)/2
        c = (contra + contra.T)/2
        neutral = np.ones_like(e) - e - c
        weighted_graph = e + 0.5 * neutral
        L = self.laplacian_matrix(weighted_graph)
        K = self.heat_kernel(L, self.t)
        return float(self.vn_entropy(K))

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

        valid_num = self.sample_num // 2
        tokenizer, nli_model = load_nli_model(config.NLI_MODEL_PATH)

        if "semantic_matrix_entail" not in dataset[0].keys():
            for item in tqdm(dataset):
                question = item.get("question", "")
                
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

                E, C, N, E_logits, C_logits, N_logits, class_mat, _ = compute_sample_semantic(
                    question=question,
                    sample_texts=sample_texts,
                    tokenizer=tokenizer,
                    nli_model=nli_model
                )

                item["semantic_matrix_entail"] = E.tolist()
                item["semantic_matrix_contra"] = C.tolist()
                item["semantic_matrix_entail_logits"] = E_logits.tolist()
                item["semantic_matrix_contra_logits"] = C_logits.tolist()
                item["class_mat"] = class_mat.tolist()
            
            with open(path, "w", encoding="utf-8") as f:
                json.dump(dataset, f, ensure_ascii=False, indent=4)

        kle_scores = []
        for item in dataset:
            E = np.asarray(item["semantic_matrix_entail"], dtype=np.float32)
            C = np.asarray(item["semantic_matrix_contra"], dtype=np.float32)
            class_mat = np.asarray(item["class_mat"])
            if E.ndim != 2 or E.shape[0] <= valid_num or E.shape[1] <= valid_num:
                kle_scores.append(None)
            else:
                kle_scores.append(self.kle_score(E, C, class_mat))

        return kle_scores

    def calculate(self):

        sample_test_path, sample_dev_path = build_sample_path(self.save_path)
        kle_scores_test = self.calculate_scores(sample_test_path)
        kle_scores_dev = self.calculate_scores(sample_dev_path)

        with open(sample_test_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        
        output = []

        valid_scores_dev = [
            float(x) for x in kle_scores_dev
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

        for item, kle_score in zip(dataset, kle_scores_test):
            if kle_score is not None and item["greedy_results"]["answer"] is not None:
                kle_score_norm = minmax(kle_score)           
                conf = 1.0 - kle_score_norm 
                output.append({"id": item["id"], "question": item["question"],
                    "description": item["description"], "ground_truth": item["ground_truth"],
                    "pred_answer": item["greedy_results"]["answer"],
                    "kle_score": kle_score, "pred_score": conf})
            else:
                output.append({"id": item["id"], "question": item["question"],
                    "description": item["description"], "ground_truth": item["ground_truth"],
                    "pred_answer": item["greedy_results"]["answer"],
                    "kle_score": None, "pred_score": None})

            if "story" in item.keys():
                output[-1]["story"] = item["story"]


        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=4)
