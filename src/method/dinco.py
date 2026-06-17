from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import os
import re
import numpy as np
import scipy.linalg
import copy
import json
import torch
from tqdm import tqdm
from src import config
from src.utils import compute_greedy_semantic, sample_generate, build_sample_path, extract_final_answer, load_nli_model, compute_sample_semantic


class DiNCo:

    def __init__(
        self,
        model_name,
        model,
        dataset_name, 
        dev_dataset,
        test_dataset,
        sample_num, 
        temperature,
        threshold
    ):
        self.system_prompt = """Read the following question and reason step-by-step to formulate your final answer. {description}

Output ONLY a single JSON object with fields in this exact order:
{{
  "reasoning": "Your step-by-step analysis",
  "final_answer": "Your final answer"
}}"""
        self.beam_system_prompt = """Read the following question and reason step-by-step to formulate 5 most possible answers. {description} Then, reason about the confidence in each possible answer.

Output ONLY a single JSON object with fields in this exact order:
{{
  "reasoning": "Your step-by-step analysis",
  "answers": [
    {{"candidate": "first most possible answer", "confidence": 0.0-1.0}},
    {{"candidate": "second most possible answer", "confidence": 0.0-1.0}},
    ...
    {{"candidate": "fifth most possible answer", "confidence": 0.0-1.0}}
  ]
}}"""
        self.threshold = threshold
        self.model_name = model_name
        self.model = model
        self.dataset_name = dataset_name
        self.dataset = test_dataset
        self.sample_num = sample_num
        self.temperature = temperature
        self.save_path = f"{config.OUTPUT_DIR}/dinco/{dataset_name}/{model_name}.json"
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        self.show = True

    def build_system_prompt(self, item):
        return self.sample_system_prompt.format(description=item["description"]) 

    def build_beam_system_prompt(self, item):
        return self.beam_system_prompt.format(description=item["description"]) 

    def build_user_prompt(self, item: Dict[str, Any]) -> str:
        return f"Question: {item['question']}"

    def build_coqa_user_prompt(self, item):
        return f"Context: {item['story']}\nQuestion: {item['question']}"


    def _to_conf(self, x: Any) -> Optional[float]:
        if x is None:
            return None
        if isinstance(x, str):
            s = x.strip()
            s = s.rstrip("%").strip()
            try:
                v = float(s)
            except Exception:
                return None
        else:
            try:
                v = float(x)
            except Exception:
                return None

        if v > 1.0:
            v = v / 100.0
        if v < 0.0:
            v = 0.0
        if v > 1.0:
            v = 1.0
        return float(v)

    def extract_candidates_confidences(self, text: str):
        if not isinstance(text, str) or not text.strip():
            return []

        raw = text.strip()
        blocks: List[str] = []
        for m in re.finditer(r"```json\s*(\[[\s\S]*?\])\s*```", raw, flags=re.IGNORECASE):
            blocks.append(m.group(1))

        for m in re.finditer(r"(\[\s*\{[\s\S]*?\"candidate\"[\s\S]*?\}\s*\])", raw):
            blocks.append(m.group(1))

        for blk in reversed(blocks):
            try:
                arr = json.loads(blk)
            except Exception:
                continue
            if not isinstance(arr, list):
                continue

            out: List[Dict[str, Any]] = []
            for obj in arr:
                if not isinstance(obj, dict):
                    continue
                cand = obj.get("candidate", None)
                conf = obj.get("confidence", None)

                if not isinstance(cand, str):
                    continue
                cand = cand.strip()
                if not cand:
                    continue

                conf_f = self._to_conf(conf)
                if conf_f is None:
                    continue

                out.append({"candidate": cand, "confidence": conf_f})

            if out:
                return out

        pairs = re.findall(
            r"\"candidate\"\s*:\s*\"(.*?)\"\s*,\s*\"confidence\"\s*:\s*([0-9]*\.?[0-9]+%?)",
            raw,
            flags=re.DOTALL,
        )
        if not pairs:
            return []

        out = []
        for cand, conf in pairs:
            cand = (cand or "").strip()
            conf_f = self._to_conf(conf)
            if cand and conf_f is not None:
                out.append({"candidate": cand, "confidence": conf_f})
        return out

    def beam_score(self, item) -> float:
        bs = item["beam_search_results"]
        conf_d = bs["confidence"]
        ans_d = bs["answer"]

        E = np.asarray(item["beam_semantic_entail"], dtype=np.float32)   
        C = np.asarray(item["beam_semantic_contra"], dtype=np.float32)
        if E.size == 0:
            return None
        B = E.shape[0]

        p = np.full((B,), 0.0, dtype=np.float32)
        for i in range(B):
            key = str(i)
            if key in conf_d:
                p[i] = float(conf_d[key])

        main = 0
        p0 = float(p[main])

        contra_w = (C + C.T) / 2.0

        E_no_diag = E.copy()
        np.fill_diagonal(E_no_diag, -1.0)
        degrees = np.sum(np.maximum(0.0, E_no_diag), axis=0) + 1.0
        numerator = p0
        denominator = numerator

        for i in range(B):
            if i == main:
                continue

            denom_i = float(degrees[i] - E[main, i])
            if denom_i <= 1e-12:
                continue

            denominator += float(p[i]) * float(contra_w[main, i]) / denom_i

        if denominator > 1.0:
            nvc = numerator / denominator
        else:
            nvc = numerator

        return float(nvc)

    def sample_score(self, sample_item, item) -> float:
        
        valid_num = min(self.sample_num // 2, 1)
        E = item["anchor_semantic_entail"]
        E_re = item["anchor_semantic_entail_re"]

        E = np.asarray(E, dtype=np.float32).reshape(-1)
        E_re = np.asarray(E_re, dtype=np.float32).reshape(-1)

        if E.size <=valid_num or E_re.size <=valid_num:
            return None

        sims = (E + E_re) / 2.0
        matches = (sims > self.threshold).astype(np.float32)
        matches = np.concatenate([matches, np.ones((1,), dtype=np.float32)], axis=0)

        return float(np.mean(matches))

    def generate(self, **kwargs: Any):

        sample_test_path, _ =  build_sample_path(self.save_path)   
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

        messages = []
        for item in self.dataset:
            system = self.build_beam_system_prompt(item)
            if self.dataset_name == "coqa":
                user = self.build_coqa_user_prompt(item)
            else:
                user = self.build_user_prompt(item)
            messages.append([
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ])

        if self.show:
            print(messages[0])

        texts = self.model.generate_batch(messages, temperature=0.1, **kwargs)

        for idx, item in enumerate(self.dataset):
            item["beam_search_results"] = {"text": texts[idx]}

        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(self.dataset, f, ensure_ascii=False, indent=4)

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


        with open(self.save_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        for item in tqdm(dataset):
            text = item["beam_search_results"]["text"]
            ans_conf = self.extract_candidates_confidences(text)
            ans_conf_sorted = sorted(
                ans_conf,
                key=lambda d: float(d.get("confidence", 0.0) if d.get("confidence", None) is not None else 0.0),
                reverse=True,
            )
            answers = {str(i): d["candidate"] for i, d in enumerate(ans_conf_sorted)}
            confs   = {str(i): d["confidence"] for i, d in enumerate(ans_conf_sorted)}
            item["beam_search_results"]["answer"] = answers
            item["beam_search_results"]["confidence"] = confs

        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)

        return dataset

    def calculate(self):

        sample_test_path, _ =  build_sample_path(self.save_path)
        with open(sample_test_path, "r", encoding="utf-8") as f:
            dataset1 = json.load(f)
        
        with open(self.save_path, "r", encoding="utf-8") as f:
            dataset2 = json.load(f)  
        
        tokenizer, nli_model = load_nli_model(config.NLI_MODEL_PATH)

        for item1, item2 in tqdm(zip(dataset1, dataset2), total=len(dataset1)):
            question = item1.get("question", "")
            sample_texts = []
            for idx, s in enumerate(item1["sample_results"][:self.sample_num]):
                ans = s["answer"]
                if ans is None:
                    continue
                if not isinstance(ans, str):
                    ans = str(ans)
                ans_strip = ans.strip()
                if not ans_strip:
                    continue
                sample_texts.append(ans_strip)

            greedy_text = item2["beam_search_results"]["answer"]["0"] if item2["beam_search_results"]["answer"] and item2["beam_search_results"]["answer"]["0"] is not None else ""
            E, C, E_logits, C_logits, E_re, C_re, E_logits_re, C_logits_re, _ = compute_greedy_semantic(
                    question=question,
                    sample_texts = sample_texts,
                    greedy_text = greedy_text,
                    tokenizer=tokenizer,
                    nli_model=nli_model
            )

            item2["anchor_semantic_entail"] = E.tolist()
            item2["anchor_semantic_entail_re"] = E_re.tolist()

        for item in tqdm(dataset2):
            question = item.get("question", "")
            sample_texts = []
            for idx, ans in enumerate(item["beam_search_results"]["answer"].values()):
                if ans is None:
                    continue
                if not isinstance(ans, str):
                    ans = str(ans)
                ans_strip = ans.strip()
                if not ans_strip:
                    continue
                sample_texts.append(ans_strip)

            E, C, N, E_logits, C_logits, N_logits, class_mat, meta = compute_sample_semantic(
                    question=question,
                    sample_texts=sample_texts,
                    tokenizer=tokenizer,
                    nli_model=nli_model
            )

            item["beam_semantic_entail"] = E.tolist()
            item["beam_semantic_contra"] = C.tolist()

        output = []
        dinco_scores = []
        for item1, item2 in zip(dataset1, dataset2):
            beam_score = self.beam_score(item2)
            sample_score = self.sample_score(item1, item2)
            if beam_score == None and sample_score is not None:
                dinco_scores.append(sample_score)
            elif beam_score is not None and sample_score == None:
                dinco_scores.append(beam_score)
            elif beam_score is not None and sample_score is not None:
                dinco_scores.append((beam_score+sample_score)/2)
            else:
                dinco_scores.append(None)

        for item, dinco_score in zip(dataset2, dinco_scores):
            if item["beam_search_results"]["answer"] != {}:
                item["pred_answer"] = item["beam_search_results"]["answer"]["0"]
            else:
                item["pred_answer"] = None
            
            item["pred_score"] = dinco_score
            if item["pred_answer"] is None:
                item["pred_score"] = None

        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(dataset2, f, ensure_ascii=False, indent=4)
