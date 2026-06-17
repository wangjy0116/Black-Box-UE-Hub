from typing import Any, Dict, List, Optional, Tuple
import json
import ast
from src.utils import compute_sample_semantic, norm_mc, load_nli_model
from tqdm import tqdm
import numpy as np
from src import config
import os

class Ling:

    def __init__(self, model_name, model, dataset_name, dev_dataset, test_dataset, sample_num, temperature):
        self.confidence_map = {"Certain": 1.0, "Almost Certain": 0.95, "Highly Likely": 0.9, "Very Good Chance": 0.8,
        "We Believe": 0.75, "Probably": 0.7, "Probable": 0.7, "Likely": 0.7, "Better than Even": 0.6, 
        "About Even": 0.5, "Probably Not": 0.25, "We Doubt": 0.2, "Unlikely": 0.2, "Little Chance": 0.1,
        "Chances are Slight": 0.1, "Improbable": 0.1, "Highly Unlikely": 0.05, "Almost No Chance": 0.02, "Impossible": 0.0}
        self.system_prompt = """Read the following question and reason step-by-step to formulate 2 best guesses and describe how likely it is that your guess is correct as one of the following expressions: {expression}. {description}

Output ONLY a single JSON object with fields in this exact order:
{{
  "reasoning": "Your step-by-step analysis",
  "answers": [
    {{"candidate": "first most likely guess", "confidence": description of confidence}},
    {{"candidate": "second most likely guess", "confidence": description of confidence}}
  ]
}}"""
        self.model_name = model_name
        self.model = model
        self.dataset_name = dataset_name
        self.dataset = test_dataset
        self.sample_num = sample_num
        self.temperature = temperature
        self.save_path = f"{config.OUTPUT_DIR}/ling/{dataset_name}/{model_name}.json"
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        self.show = True

    def build_system_prompt(self, item):
        expression = '["Certain", "Almost Certain", "Highly Likely", "Very Good Chance", "We Believe", "Probably", "Probable", "Likely", "Better than Even", "About Even", "Probably Not", "We Doubt", "Unlikely", "Little Chance", "Chances are Slight", "Improbable", "Highly Unlikely", "Almost No Chance", "Impossible"]'
        return self.system_prompt.format(expression = expression, description=item["description"]) 

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

    def _normalize_answer(self, ans: Any) -> Optional[str]:
        """
        Normalize final_answer into a string (to keep downstream logic intact).
        - str -> str
        - list -> if len==1 return the single element; else JSON-dump list
        - others -> stringify
        """
        if ans is None:
            return None
        if isinstance(ans, str):
            s = ans.strip()
            return s if s else None
        if isinstance(ans, list):
            cleaned = []
            for x in ans:
                if isinstance(x, str):
                    cleaned.append(x.strip())
                else:
                    cleaned.append(str(x))
            cleaned = [x for x in cleaned if x]  # drop empty strings
            if len(cleaned) == 0:
                return None
            if len(cleaned) == 1:
                return cleaned[0]
            return json.dumps(cleaned, ensure_ascii=False)
        s = str(ans).strip()
        return s if s else None

    def parse_answer_confidence(self, text: str) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, float]]:
        answers: Dict[str, str] = {}
        conf_texts: Dict[str, str] = {}
        confs: Dict[str, float] = {}

        if not isinstance(text, str):
            return answers, conf_texts, confs

        raw = text.strip()
        if not raw:
            return answers, conf_texts, confs

        conf_map_lower = {k.strip().lower(): float(v) for k, v in self.confidence_map.items()}

        def _normalize_conf_any(conf: Any) -> Optional[float]:
            if conf is None:
                return None

            try:
                v = self._normalize_confidence(conf)
                if isinstance(v, float):
                    return v
            except Exception:
                pass

            if isinstance(conf, str):
                t = conf.strip()
                if t in self.confidence_map:
                    return float(self.confidence_map[t])
                tl = t.lower()
                if tl in conf_map_lower:
                    return float(conf_map_lower[tl])

            return None

        def _pack_topk(pairs: List[Tuple[str, str, float]]) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, float]]:
            pairs = sorted(pairs, key=lambda x: x[2], reverse=True)
            ans_d = {str(i): a for i, (a, _, _) in enumerate(pairs)}
            txt_d = {str(i): t for i, (_, t, _) in enumerate(pairs)}
            val_d = {str(i): c for i, (_, _, c) in enumerate(pairs)}
            return ans_d, txt_d, val_d

        def _extract_pairs(obj: Any) -> Optional[Tuple[Dict[str, str], Dict[str, str], Dict[str, float]]]:
            if not isinstance(obj, dict):
                return None
            arr = obj.get("answers", None)
            if not isinstance(arr, list):
                return None

            pairs: List[Tuple[str, str, float]] = []
            for it in arr:
                if not isinstance(it, dict):
                    continue

                cand = self._normalize_answer(it.get("candidate"))
                if not cand:
                    continue

                conf_raw = it.get("confidence")
                if conf_raw is None:
                    conf_raw = it.get("probability")

                conf_val = _normalize_conf_any(conf_raw)
                if conf_val is None:
                    continue

                if isinstance(conf_raw, str):
                    conf_text = conf_raw.strip()
                else:
                    conf_text = str(conf_raw).strip()

                if not conf_text:
                    continue

                pairs.append((cand, conf_text, float(conf_val)))

            if not pairs:
                return None
            return _pack_topk(pairs)

        for parser in (json.loads, ast.literal_eval):
            try:
                obj = parser(raw)
            except Exception:
                continue

            parsed = _extract_pairs(obj)
            if parsed is not None:
                return parsed

            if isinstance(obj, str) and obj.strip():
                try:
                    obj2 = parser(obj.strip())
                except Exception:
                    continue
                parsed = _extract_pairs(obj2)
                if parsed is not None:
                    return parsed

        return answers, conf_texts, confs

    def generate(self, **kwargs: Any,):
        messages = []
        for item in self.dataset:
            system = self.build_system_prompt(item)
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
                if "answer" not in s.keys():
                    text = s.get("text", "")
                    ans, conf_t, conf_d = self.parse_answer_confidence(text)
                    if self.dataset_name == "truthfulqa_mc":
                        s["answer"] = {}
                        s["confidence_text"] = {}
                        s["confidence"] = {}
                        for k in ans.keys():
                            norm_ans = norm_mc(ans[k], item)
                            if norm_ans is not None:
                                s["answer"][k] = norm_ans
                                s["confidence_text"][k] = conf_t[k]
                                s["confidence"][k] = conf_d[k]
                                
                    else:
                        s["answer"] = ans
                        s["confidence_text"] = conf_t
                        s["confidence"] = conf_d

        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)

    def calculate(self):
        with open(self.save_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        valid_num = min(self.sample_num // 2, 1)
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
                for k, a in ans_d.items():
                    c = conf_d.get(k, None)
                    if not isinstance(a, str) or c is None:
                        continue
                    a = a.strip()
                    if not a:
                        continue
                    all_texts.append(a)
                    all_confs.append(float(c))

            if not all_texts or vaild<=valid_num:
                item["pred_answer"] = None
                item["pred_score"] = None
                item["dedup_answers"] = []
                item["cluster"] = []
                item["pred_dict"] = {}
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

                item["pred_answer"] = rep_ans
                item["pred_score"] = sum(all_confs) / vaild
                if item["pred_score"] > 1.0:
                    item["pred_score"] = 1.0
                continue

            if self.dataset_name == "truthfulqa_mc":
                clusters = [[did] for did in range(len(dedup_texts))]
                item["cluster"] = clusters
                dedup2cid = {did: text2id[dedup_texts[did]] for did in range(len(dedup_texts))}
            else:
                question = item["question"]
                _, _, _,_, _, _, class_mat, id_map = compute_sample_semantic(
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

            if total_conf < 0:
                item["pred_dict"] = {}
                item["pred_answer"] = None
                item["pred_score"] = None
                continue

            scaled = {k: v / vaild for k, v in avg_conf_dict.items()}
            sorted_scaled = dict(sorted(scaled.items(), key=lambda kv: kv[1], reverse=True))

            item["pred_dict"] = sorted_scaled
            it = iter(sorted_scaled.items())
            ans, conf = next(it)
            item["pred_answer"] = ans
            if conf > 1.0:
                conf = 1.0
            item["pred_score"] = conf

        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)