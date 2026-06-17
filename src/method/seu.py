from typing import Any, Dict, List, Optional, Tuple
import os
import re
import numpy as np
import scipy.linalg
import copy
import json
import torch
from tqdm import tqdm
from src.utils import compute_sample_semantic, build_sample_path, sample_generate, extract_final_answer, load_nli_model, compute_sample_cosine
from src import config

class SEU:
    def __init__(
        self,
        model_name,
        model, 
        dataset_name,
        dev_dataset,
        test_dataset,
        sample_num, 
        temperature
    ):

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
        self.save_path = f"{config.OUTPUT_DIR}/seu/{dataset_name}/{model_name}.json"
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

    def build_system_prompt(self, item):
        return self.system_prompt.format(description=item["description"]) 

    def build_user_prompt(self, item: Dict[str, Any]) -> str:
        return f"Question: {item['question']}"

    def build_coqa_user_prompt(self, item):
        return f"Context: {item['story']}\nQuestion: {item['question']}"
   
    def seu_score(self, cosine) -> float:
        m = len(cosine)
        score = 0.0
        for i in range(m-1):
            for j in range(i+1, m):
                score+=cosine[i][j]
        score = 1.0-(2.0*score/(m*(m-1)))
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

    def calculate_scores(self, path):

        with open(path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        valid_num = min(self.sample_num // 2, 1)
        tokenizer, nli_model = load_nli_model(config.NLI_MODEL_PATH)
        if "cosine_matrix" not in dataset[0].keys():
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

                cosine_matrix = compute_sample_cosine(
                    question=question,
                    sample_texts=sample_texts
                )

                item["cosine_matrix"] = cosine_matrix.tolist()
            
            with open(path, "w", encoding="utf-8") as f:
                json.dump(dataset, f, ensure_ascii=False, indent=4)

        seu_scores = []
        for item in dataset:
            C = np.asarray(item["cosine_matrix"], dtype=np.float32)
            if C.ndim != 2 or C.shape[0] <= valid_num or C.shape[1] <= valid_num:
                seu_scores.append(None)
            else:
                seu_scores.append(self.seu_score(C))

        return seu_scores


    def calculate(self):

        sample_test_path, sample_dev_path = build_sample_path(self.save_path)
        seu_scores_test = self.calculate_scores(sample_test_path)
        seu_scores_dev = self.calculate_scores(sample_dev_path)

        with open(sample_test_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        
        output = []

        valid_scores_dev = [
            float(x) for x in seu_scores_dev
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

        for item, seu_score in zip(dataset, seu_scores_test):
            if seu_score is not None and item["greedy_results"]["answer"] is not None:
                seu_score_norm = minmax(seu_score)           
                conf = 1.0 - seu_score_norm 
                output.append({"id": item["id"], "question": item["question"],
                    "description": item["description"], "ground_truth": item["ground_truth"],
                    "pred_answer": item["greedy_results"]["answer"],
                    "seu_score": seu_score, "pred_score": conf})
            else:
                output.append({"id": item["id"], "question": item["question"],
                    "description": item["description"], "ground_truth": item["ground_truth"],
                    "pred_answer": item["greedy_results"]["answer"],
                    "seu_score": None, "pred_score": None})

            if "story" in item.keys():
                output[-1]["story"] = item["story"]


        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=4)



