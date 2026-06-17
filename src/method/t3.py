from typing import Any, Dict, List, Optional, Tuple
import json
import ast
from tqdm import tqdm
import numpy as np
from src import config
import os

class T3:

    def __init__(self, model_name, model, dataset_name, dev_dataset, test_dataset, temperature = 0.1):
        self.system_prompt = """Read the following question and explanations for each option, and reason step-by-step to formulate 2 best guesses and probability that each is correct. {description}

Output ONLY a single JSON object with fields in this exact order:
{{
  "reasoning": "Your step-by-step analysis",
  "answers": [
    {{"candidate": "first most likely guess", "probability": 0.0-1.0}},
    {{"candidate": "second most likely guess", "probability": 0.0-1.0}}
  ]
}}"""
        self.model_name = model_name
        self.model = model
        self.dataset_name = dataset_name
        self.dataset = test_dataset
        self.temperature = temperature
        self.save_path = f"{config.OUTPUT_DIR}/t3/{dataset_name}/{model_name}.json"
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        self.show = True

    def build_system_prompt(self, item):
        return self.system_prompt.format(description=item["description"]) 

    def build_user_prompt_explanation(self, item, idx):
        return f"Question: {item['question']}\nThe answer is {chr(ord('A') + idx)}\nPlease generate an brief explanation to try to justify the answer judgment."
        
    def build_user_prompt(self, item):
        explanation = item["explanations"]
        exp = [f"Possible explanation {i+1}: {e}" for i,e in enumerate(explanation)]
        exp_str = "\n".join(exp)
        return f"Question: {item['question']}\n{exp_str}"

    def _normalize_confidence(self, conf: Optional[str]) -> Optional[float]:
        if conf is None:
            return None
        try:
            v = float(conf)
        except Exception:
            return None
        if v > 1.0:
            v = v / 100.0
        if v < 0.0:
            v = 0.0
        if v > 1.0:
            v = 1.0
        return v

    def parse_answer_confidence(self, text: str) -> Tuple[Dict[str, str], Dict[str, float]]:
        answers: Dict[str, str] = {}
        confs: Dict[str, float] = {}

        if not isinstance(text, str):
            return answers, confs

        raw0 = text.strip()
        if not raw0:
            return answers, confs

        def _to_candidate_str(cand: Any) -> Optional[str]:
            if cand is None:
                return None
            try:
                s = str(cand).strip()
            except Exception:
                return None
            if not s:
                return None
            if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
                s = s[1:-1].strip()
            return s if s else None

        def _fill_from_answers_array(arr: Any) -> bool:
            if not isinstance(arr, list):
                return False

            pairs: List[Tuple[str, float]] = []
            for obj in arr:
                if not isinstance(obj, dict):
                    continue

                cand_raw = obj.get("candidate", None)
                prob_raw = obj.get("probability", None)
                if prob_raw is None:
                    prob_raw = obj.get("confidence", None)

                cand = _to_candidate_str(cand_raw)
                if not cand:
                    continue

                prob_f = self._normalize_confidence(prob_raw)
                if prob_f is None:
                    continue

                pairs.append((cand, float(prob_f)))

            if not pairs:
                return False

            pairs.sort(key=lambda x: x[1], reverse=True)
            answers.clear()
            confs.clear()
            for i, (a, c) in enumerate(pairs):
                answers[str(i)] = a
                confs[str(i)] = c
            return True

        for parser in (json.loads, ast.literal_eval):
            try:
                obj = parser(raw0)
            except Exception:
                continue

            if isinstance(obj, dict) and _fill_from_answers_array(obj.get("answers", None)):
                return answers, confs

            if isinstance(obj, str) and obj.strip():
                try:
                    obj2 = parser(obj.strip())
                except Exception:
                    continue
                if isinstance(obj2, dict) and _fill_from_answers_array(obj2.get("answers", None)):
                    return answers, confs

        return answers, confs


    def generate(self, **kwargs: Any):
        messages = []
        for item in self.dataset:
            options_num = item["options_num"]
            for i in range(options_num):
                user = self.build_user_prompt_explanation(item, i)
                messages.append([{"role": "user", "content": user}])

        if self.show:
            print(messages[0])

        texts = self.model.generate_batch(messages, temperature=0.1, **kwargs)

        idx = 0
        for item in self.dataset:
            exps = []
            options_num = item["options_num"]
            for _ in range(options_num):
                t = texts[idx]
                idx += 1
                exps.append(t)
            item["explanations"] = exps
            
        messages = []
        for item in self.dataset:
            system = self.build_system_prompt(item)
            user = self.build_user_prompt(item)
            messages.append([{"role": "system", "content": system},
                {"role": "user", "content": user}])

        if self.show:
            print(messages[0])

        texts = self.model.generate_batch(messages, temperature=0.1, **kwargs)

        for i, item in enumerate(self.dataset):
            item["greedy_results"] = {"text": texts[i]}

        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(self.dataset, f, ensure_ascii=False, indent=4)
    
    def extract(self):
        with open(self.save_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        for item in tqdm(dataset):
            text = item["greedy_results"]["text"]
            ans, conf = self.parse_answer_confidence(text)
            item["greedy_results"]["answer"] = ans
            item["greedy_results"]["confidence"] = conf

        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)

        return dataset

    def calculate(self):
        with open(self.save_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        for item in tqdm(dataset):
            label = [o["label"] for o in item["options"]]
            ans = item["greedy_results"]["answer"]
            if "0" in ans.keys():
                greedy_ans = ans["0"]
                if greedy_ans.upper() in label:
                    item["pred_answer"] = greedy_ans.upper()
                else:
                    if greedy_ans[1] == '.' and greedy_ans[0].upper() in label:
                        item["pred_answer"] = greedy_ans[0].upper()
                    else:
                        item["pred_answer"] = None

                item["pred_score"] = item["greedy_results"]["confidence"]["0"]
            else:
                item["pred_score"] = None
                item["pred_answer"] = None
                
        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)

        return dataset
