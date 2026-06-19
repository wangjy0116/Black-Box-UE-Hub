from typing import Any, Dict, List, Optional, Tuple
import re
import json
from tqdm import tqdm
import numpy as np
from src.utils import compute_sample_semantic, norm_mc, load_nli_model
from src import config
import os
import ast

class VPD:

    def __init__(self, model_name, model, dataset_name, dev_dataset, test_dataset, sample_num, temperature):

        self.system_prompt_open = """Read the following question and reason step-by-step to formulate your answer. You may propose multiple possible answers (fewer than five). {description} Always include 'None of the above' as a possible answer. Reason about the confidence in each possible answer and assign confidence scores from a probability distribution (they must sum to 1.0).

Output ONLY a single JSON object with fields in this exact order:
{{
  "reasoning": "Your step-by-step analysis",
  "answers": [
    {{"candidate": "Candidate 1", "confidence": 0.0-1.0}},
    {{"candidate": "Candidate 2", "confidence": 0.0-1.0}},
    ...
    {{"candidate": "None of the above", "confidence": 0.0-1.0}}
  ]
}}"""

        self.system_prompt = """Read the following question and reason step-by-step to formulate your answer. You may propose multiple possible answers (fewer than five). {description} Reason about the confidence in each possible answer and assign confidence scores from a probability distribution(they must sum to 1.0).

Output ONLY a single JSON object with fields in this exact order:
{{
  "reasoning": "Your step-by-step analysis",
  "answers": [
    {{"candidate": "Candidate 1", "confidence": 0.0-1.0}},
    {{"candidate": "Candidate 2", "confidence": 0.0-1.0}},
    ...
  ]
}}"""
        self.model_name = model_name
        self.model = model
        self.dataset_name = dataset_name
        self.dataset = test_dataset
        self.sample_num = sample_num
        self.temperature = temperature
        self.save_path = f"{config.OUTPUT_DIR}/vpd/{dataset_name}/{model_name}.json"
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        self.show = True

    def build_system_prompt_open(self, item):
        return self.system_prompt_open.format(description=item["description"]) 

    def build_system_prompt(self, item):
        return self.system_prompt.format(description=item["description"]) 

    def build_user_prompt(self, item):
        return f"Question: {item['question']}"

    def build_coqa_user_prompt(self, item):
        return f"Context: {item['story']}\nQuestion: {item['question']}"

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

    def generate(self, **kwargs: Any,):
        messages = []
        for item in self.dataset:
            if self.dataset_name == "truthfulqa_mc":
                system = self.build_system_prompt(item)
            else:
                system = self.build_system_prompt_open(item)
            if self.dataset_name == "coqa":
                user = self.build_coqa_user_prompt(item)
            else:
                user = self.build_user_prompt(item)
            for _ in range(int(self.sample_num)):
                messages.append([{"role": "system", "content": system},
                                 {"role": "user", "content": user}])

        if self.show:
            print(messages[0])

        texts = self.model.generate_batch(messages, temperature=self.temperature, **kwargs)

        idx = 0
        for i, item in enumerate(self.dataset):
            samples = []
            for _ in range(int(self.sample_num)):
                samples.append({"text": texts[idx]})
                idx += 1
            item["sample_results"] = samples
 
        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(self.dataset, f, ensure_ascii=False, indent=4)


    def extract(self):
        with open(self.save_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        for item in tqdm(dataset):
            samples = item.get("sample_results", [])
            for s in samples:
                text = s.get("text", "")
                if "answer" not in s.keys():
                    ans, conf = self.parse_answer_confidence(text)
                    if self.dataset_name == "truthfulqa_mc":
                        s["answer"] = {}
                        s["confidence"] = {}
                        for k in ans.keys():
                            norm_ans = norm_mc(ans[k], item)
                            if norm_ans is not None:
                                s["answer"][k] = norm_ans
                                s["confidence"][k] = conf[k]
                    else:
                        s["answer"] = ans
                        s["confidence"] = conf

        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)

    def calculate(self):
        with open(self.save_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        def _is_none_of_above(s: str) -> bool:
            if not isinstance(s, str):
                return False
            return "none of the above" in s.strip().lower()

        valid_num = self.sample_num // 2
        tokenizer, nli_model = load_nli_model(config.NLI_MODEL_PATH)
        for item in tqdm(dataset):
            samples = item["sample_results"][:self.sample_num]

            all_texts = []
            all_confs = []

            vaild = 0
            for sample in samples:
                ans_d = sample.get("answer", {}) or {}
                conf_d = sample.get("confidence", {}) or {}
                if ans_d != {}:
                    vaild += 1
                sum_c = 0.0
                for k, a in ans_d.items():
                    c = conf_d.get(k, None)
                    if not isinstance(a, str) or c is None:
                        continue
                    a = a.strip()
                    if not a:
                        continue
                    sum_c += float(c)
                for k, a in ans_d.items():
                    c = conf_d.get(k, None)
                    if not isinstance(a, str) or c is None:
                        continue
                    a = a.strip()
                    if not a:
                        continue
                    all_texts.append(a)
                    if sum_c != 0.0:
                        all_confs.append(float(c)/sum_c)
                    else:
                        all_confs.append(float(c))

            if not all_texts or vaild<=valid_num:
                item["pred_answer"] = None
                item["pred_score"] = None
                item["dedup_answers"] = []
                item["cluster"] = []
                item["pred_dict"] = {}
                continue

            has_non_nota = any((isinstance(a, str) and (not _is_none_of_above(a))) for a in all_texts)
            if not has_non_nota:
                item["pred_answer"] = "None of the above"
                item["pred_score"] = None
                item["dedup_answers"] = ["None of the above"]
                item["cluster"] = [[0]]
                item["pred_dict"] = {"None of the above": 1.0}
                continue

            dedup_texts = []
            text2id = {}

            for ii, key in enumerate(all_texts):
                if key not in text2id:
                    text2id[key] = [ii]
                    dedup_texts.append(key)
                else:
                    text2id[key].append(ii)

            item["dedup_answers"] = dedup_texts

            if len(dedup_texts) == 1:
                item["cluster"] = [[0]]
                rep_k = int(np.argmax(all_confs))
                rep_ans = all_texts[rep_k]
                if _is_none_of_above(rep_ans):
                    item["pred_answer"] = "None of the above"
                    item["pred_score"] = None
                else:
                    item["pred_answer"] = rep_ans
                    item["pred_score"] = 1.0
                continue

            if self.dataset_name == "truthfulqa_mc":
                clusters = [[did] for did in range(len(dedup_texts))]
                item["cluster"] = clusters
                dedup2cid = {did: text2id[dedup_texts[did]] for did in range(len(dedup_texts))}
            else:
                question = item["question"]
                _, _, _, _,_, _, class_mat, id_map = compute_sample_semantic(
                    question=question,
                    sample_texts=dedup_texts,
                    tokenizer=tokenizer,
                    nli_model=nli_model
                )
                entail_id = id_map["entail_id"]

                n = len(dedup_texts)
                visited = [False] * n
                clusters = []

                for i in range(n):
                    if visited[i]:
                        continue
                    cluster = [i]
                    visited[i] = True
                    for j in range(n):
                        if visited[j]:
                            continue
                        if class_mat[i, j] == entail_id and class_mat[j, i] == entail_id:
                            cluster.append(j)
                            visited[j] = True
                    clusters.append(cluster)

                item["cluster"] = clusters

                dedup2cid = {}
                for cid, clu in enumerate(clusters): 
                    dedup2cid[cid] = []
                    for did in clu:
                        dedup2cid[cid].extend(text2id[dedup_texts[did]])


            avg_conf_dict = {}
            total_conf = 0.0
            for k, v in dedup2cid.items():
                sum_conf = 0
                max_id = -1
                max_conf = -1.0
                for vv in v:
                    sum_conf += all_confs[vv]
                    if all_confs[vv]>max_conf:
                        max_conf = all_confs[vv]
                        max_id = vv
                avg_conf_dict[all_texts[max_id]] = sum_conf
                total_conf += sum_conf

            if total_conf <= 0:
                item["pred_dict"] = {}
                item["pred_answer"] = None
                item["pred_score"] = None
                continue

            scaled = {k: v / total_conf for k, v in avg_conf_dict.items()}
            sorted_scaled = dict(sorted(scaled.items(), key=lambda kv: kv[1], reverse=True))

            item["pred_dict"] = sorted_scaled
            it = iter(sorted_scaled.items())
            ans, conf = next(it)
            if _is_none_of_above(ans):
                if len(sorted_scaled) == 1:
                    item["pred_answer"] = "None of the above"
                    item["pred_score"] = None
                else:
                    ans2, conf2 = next(it)
                    item["pred_answer"] = ans2
                    item["pred_score"] = conf2
            else:
                item["pred_answer"] = ans
                item["pred_score"] = conf

        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)

        return dataset