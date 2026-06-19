from typing import Any, Dict, List, Optional, Tuple
import os
import re
from src.utils import build_sample_path, sample_generate, extract_final_answer, compute_sentence_greedy_semantic, load_nli_model
import numpy as np
import copy
import json
import torch
from tqdm import tqdm
from src import config


class SelfCheckGPT:

    def __init__(
        self,
        model_name,
        model, 
        dataset_name,
        dev_dataset,
        test_dataset,
        sample_num, 
        temperature,
    ):  
        self.model_name = model_name
        self.model = model
        self.dataset_name = dataset_name
        self.dataset = test_dataset
        self.sample_num = sample_num
        self.temperature = temperature
        self.save_path = f"{config.OUTPUT_DIR}/selfcheck/{dataset_name}/{model_name}.json"
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

    def selfcheck_score(self, entail_logits, contra_logits):

        e = torch.as_tensor(entail_logits, dtype=torch.float32)
        c = torch.as_tensor(contra_logits, dtype=torch.float32)

        if e.shape != c.shape:
            raise ValueError(f"entail_logits and contra_logits have mismatched shapes: {e.shape} vs {c.shape}")

        if e.ndim == 1:
            e = e.unsqueeze(0)  
            c = c.unsqueeze(0)

        if e.ndim != 2:
            raise ValueError(f"Expected logits to be 2D (M, N) or 1D (N,), but got ndim={e.ndim}, shape={tuple(e.shape)}")

        p_contra = torch.softmax(torch.stack([e, c], dim=-1), dim=-1)[..., 1] 
        per_sentence = p_contra.mean(dim=1)
        return per_sentence.detach().cpu().tolist()

    def generate(self, **kwargs: Any) -> None:

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
            samples = item.get("sample_results", [])
            for s in samples:
                if "answer" not in s.keys():
                    text = s.get("text", "")
                    s["answer"] = extract_final_answer(text)

            greedy = item.get("greedy_results", None)
            if "answer" not in greedy.keys():
                text = greedy.get("text", "")
                greedy["answer"] = extract_final_answer(text)

        with open(sample_test_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)

    def calculate_scores(self, path):

        with open(path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        
        valid_num = self.sample_num // 2
        tokenizer, nli_model = load_nli_model(config.NLI_MODEL_PATH)

        if "sentence_greedy_semantic_entail_logits" not in dataset[0]:
            for item in tqdm(dataset):
                question = item.get("question", "")
                sample_texts = []
                for idx, s in enumerate(item["sample_results"][:self.sample_num]):
                    ans = s.get("answer", None)
                    if ans is None:
                        continue
                    ans = ans if isinstance(ans, str) else str(ans)
                    ans_strip = ans.strip()
                    if not ans_strip:
                        continue
                    sample_texts.append(ans_strip)

                E, C, E_logits, C_logits, _ = compute_sentence_greedy_semantic(
                    question=question,
                    greedy_text=item["greedy_results"]["answer"] if item["greedy_results"]["answer"] is not None else "",
                    sample_texts=sample_texts,
                    tokenizer=tokenizer,
                    nli_model=nli_model
                )

                item["sentence_greedy_semantic_entail_logits"] = E_logits.tolist()
                item["sentence_greedy_semantic_contra_logits"] = C_logits.tolist()

            with open(path, "w", encoding="utf-8") as f:
                json.dump(dataset, f, ensure_ascii=False, indent=4)

        scores = []
        score_lists = []
        for item in tqdm(dataset):
            E = np.asarray(item["sentence_greedy_semantic_entail_logits"], dtype=np.float32)
            C = np.asarray(item["sentence_greedy_semantic_contra_logits"], dtype=np.float32)

            if E.ndim != 2 or E.shape[1] <= valid_num:
                score_lists.append(None)
                scores.append(None)
                continue

            per_sent = self.selfcheck_score(E, C)         
            score_lists.append(per_sent)
            scores.append(float(np.mean(per_sent)))

        return scores, score_lists

    def calculate(self):
        sample_test_path, _ = build_sample_path(self.save_path)
        test_scores, test_scores_lists = self.calculate_scores(sample_test_path)

        with open(sample_test_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        output = []
        for item, selfcheck_score, selfcheck_score_list in zip(dataset, test_scores, test_scores_lists):
            greedy_ans = item["greedy_results"].get("answer", None)

            if selfcheck_score is not None and greedy_ans is not None:
                conf = 1.0 - selfcheck_score
                output.append({
                    "id": item["id"],
                    "question": item["question"],
                    "description": item.get("description", ""),
                    "ground_truth": item.get("ground_truth", None),
                    "pred_answer": greedy_ans,
                    "selfcheck_score_list": selfcheck_score_list,
                    "selfcheck_score": float(selfcheck_score),
                    "pred_score": float(conf)
                })
            else:
                output.append({
                    "id": item["id"],
                    "question": item["question"],
                    "description": item.get("description", ""),
                    "ground_truth": item.get("ground_truth", None),
                    "pred_answer": greedy_ans,
                    "selfcheck_score_list": None,
                    "selfcheck_score": None,
                    "pred_score": None
                })

            if "story" in item.keys():
                output[-1]["story"] = item["story"]

        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=4)
