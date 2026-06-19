from typing import Any, Dict, List, Optional, Tuple
import os
import json
from src.utils import compute_sample_semantic, norm_mc, load_nli_model
from tqdm import tqdm
import numpy as np
import ast
from src import config

class CoT:
    def __init__(self, model_name, model, dataset_name, dev_dataset, test_dataset, sample_num, temperature):
        self.system_prompt = """Read the following question and reason step-by-step to formulate your final answer. {description} Then, reason about the confidence in your answer.

Output ONLY a single JSON object with fields in this exact order:
{{
  "reasoning": "Your step-by-step analysis",
  "final_answer": "Your final answer",
  "confidence": 0.0-1.0
}}"""
        self.model_name = model_name
        self.model = model
        self.dataset_name = dataset_name
        self.dataset = test_dataset
        self.sample_num = sample_num
        self.temperature = temperature
        self.save_path = f"{config.OUTPUT_DIR}/cot/{dataset_name}/{model_name}.json"
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        self.show = True


    def build_system_prompt(self, item):
        return self.system_prompt.format(description=item["description"]) 

    def build_user_prompt(self, item):
        return f"Question: {item['question']}"
        
    def build_coqa_user_prompt(self, item):
        return f"Context: {item['story']}\nQuestion: {item['question']}"

    def _normalize_confidence(self, conf):
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

    def _normalize_answer(self, ans):
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
            cleaned = [x for x in cleaned if x]
            if len(cleaned) == 0:
                return None
            if len(cleaned) == 1:
                return cleaned[0]
            return json.dumps(cleaned, ensure_ascii=False)
        s = str(ans).strip()
        return s if s else None

    def parse_answer_confidence(self, text):
        if not isinstance(text, str):
            return None, None
        raw = text.strip()
        if not raw:
            return None, None

        def _extract(obj):
            if not isinstance(obj, dict):
                return None, None
            ans = obj.get("final_answer", obj.get("answer"))
            conf = obj.get("confidence", obj.get("probability"))
            ans = self._normalize_answer(ans)
            conf = self._normalize_confidence(conf)
            if ans is None or conf is None:
                return None, None
            return ans, conf

        for parser in (json.loads, ast.literal_eval):
            try:
                obj = parser(raw)
            except Exception:
                continue
            ans, conf = _extract(obj)
            if ans is not None and conf is not None:
                return ans, conf

        return None, None

    def generate(self, **kwargs):
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
                    ans, conf = self.parse_answer_confidence(text)
                    if self.dataset_name == "truthfulqa_mc":
                        s["answer"] = norm_mc(ans, item)
                    else:
                        s["answer"] = ans
                    s["confidence"] = conf

        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)


    def calculate(self):
        with open(self.save_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        valid_num = self.sample_num // 2
        tokenizer, nli_model = load_nli_model(config.NLI_MODEL_PATH)
        for item in tqdm(dataset):
            samples = item["sample_results"][:self.sample_num]
            answers = [s.get("answer") for s in samples]
            confidences = [s.get("confidence") for s in samples]

            valid_idxs = [
                i for i, a in enumerate(answers)
                if isinstance(a, str) and confidences[i] is not None
            ]
            if len(valid_idxs) <= valid_num:
                item["pred_answer"] = None
                item["pred_score"] = None
                item["dedup_answers"] = []
                item["cluster"] = []
                continue

            valid_answers = [answers[i].strip() for i in valid_idxs]
            valid_confs = [float(confidences[i]) for i in valid_idxs]

            if self.dataset_name == "truthfulqa_mc":
                dedup_answers = []
                ans2did = {}
                orig2dedup = []

                for a in valid_answers:
                    key = a
                    if key not in ans2did:
                        ans2did[key] = len(dedup_answers)
                        dedup_answers.append(key)
                    orig2dedup.append(ans2did[key])

                item["dedup_answers"] = dedup_answers

                clusters = [[did] for did in range(len(dedup_answers))]
                item["cluster"] = clusters

                cluster_conf = [0.0 for _ in range(len(clusters))]
                for k, did in enumerate(orig2dedup):
                    cluster_conf[did] += float(valid_confs[k])

                total_conf = float(sum(cluster_conf))
                if total_conf < 0:
                    item["pred_answer"] = None
                    item["pred_score"] = None
                    continue

                norm_conf = [c / len(valid_idxs) for c in cluster_conf]
                best_cid = int(np.argmax(norm_conf))

                rep_k = None
                rep_conf = -1.0
                for k, did in enumerate(orig2dedup):
                    if did == best_cid and valid_confs[k] > rep_conf:
                        rep_conf = valid_confs[k]
                        rep_k = k

                item["pred_answer"] = valid_answers[rep_k] if rep_k is not None else None
                item["pred_score"] = float(norm_conf[best_cid])
                continue

            dedup_answers = []
            ans2did = {}
            orig2dedup = []
            for a in valid_answers:
                if a not in ans2did:
                    ans2did[a] = len(dedup_answers)
                    dedup_answers.append(a)
                orig2dedup.append(ans2did[a])

            item["dedup_answers"] = dedup_answers

            if len(dedup_answers) == 1:
                clusters = [[0]]
                item["cluster"] = clusters
                rep_k = int(np.argmax(valid_confs))
                item["pred_answer"] = valid_answers[rep_k]
                item["pred_score"] = float(np.mean(valid_confs))
                continue
            
            question = item["question"]
            _, _, _, _, _, _, class_mat, id_map = compute_sample_semantic(
                question=question,
                sample_texts=dedup_answers,
                tokenizer=tokenizer,
                nli_model=nli_model
            )
            entail_id = id_map["entail_id"]

            n = len(dedup_answers)
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
                for did in clu:
                    dedup2cid[did] = cid

            cluster_conf = [0.0 for _ in range(len(clusters))]
            for k, did in enumerate(orig2dedup):
                cid = dedup2cid.get(did, None)
                if cid is None:
                    continue
                cluster_conf[cid] += float(valid_confs[k])

            total_conf = float(sum(cluster_conf))
            if total_conf <= 0:
                item["pred_answer"] = None
                item["pred_score"] = None
                continue

            norm_conf = [c / len(valid_idxs) for c in cluster_conf]
            best_cid = int(np.argmax(norm_conf))

            best_cluster_dedup_ids = set(clusters[best_cid])
            rep_k = None
            rep_conf = -1.0
            for k, did in enumerate(orig2dedup):
                if did in best_cluster_dedup_ids and valid_confs[k] > rep_conf:
                    rep_conf = valid_confs[k]
                    rep_k = k

            item["pred_answer"] = valid_answers[rep_k] if rep_k is not None else None
            item["pred_score"] = float(norm_conf[best_cid])

        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)

