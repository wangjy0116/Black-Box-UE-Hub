from typing import Any, Dict, List, Optional, Tuple
import re
import json
import ast
from collections import Counter
import numpy as np
from tqdm import tqdm
from src.utils import compute_sample_semantic, norm_mc, load_nli_model
from src import config
import os

class SteerConf:

    def __init__(self, model_name, model, dataset_name, dev_dataset, test_dataset, temperature):

        self.model_name = model_name
        self.model = model
        self.dataset_name = dataset_name
        self.dataset = test_dataset
        self.temperature = temperature
        self.save_path = f"{config.OUTPUT_DIR}/steerconf/{dataset_name}/{model_name}.json"
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        self.show = True

        self.system_prompt = """Read the following question and reason step-by-step to formulate your final answer. {description} Then, reason about the confidence in your answer. {steer_level}

Output ONLY a single JSON object with fields in this exact order:
{{
  'reasoning': 'Your step-by-step analysis',
  'final_answer': 'Your final answer',
  'confidence': 0.0-1.0
}}"""
        self.show = True

        self.steering_instruction = {
            "very_cautious": "You are making important decisions, thus you should avoid giving a wrong answer with high confidence. You should be very cautious, and tend to give low confidence on almost all of the answers.",
            "cautious": "You are making important decisions, thus you should avoid giving a wrong answer with high confidence.",
            "vanilla": "",
            "confident": "You are making important decisions, thus you should avoid giving a right answer with low confidence.",
            "very_confident": "You are making important decisions, thus you should avoid giving a right answer with low confidence. You should be very confident, and tend to give high confidence on almost all of the answers",
        }

        self.steering_levels = [
            "very_cautious",
            "cautious",
            "vanilla",
            "confident",
            "very_confident",
        ]

        self.level_order = {
            "very_cautious": 0,
            "cautious": 1,
            "vanilla": 2,
            "confident": 3,
            "very_confident": 4,
        }

    def build_system_prompt(self, item, steer_prompt):
        return self.system_prompt.format(description=item.get("description", ""), steer_level = steer_prompt)

    def build_user_prompt(self, item):
        return f"Question: {item['question']}"

    def build_coqa_user_prompt(self, item):
        return f"Context: {item['story']}\nQuestion: {item['question']}"

    def _normalize_confidence(self, conf: Optional[Any]) -> Optional[float]:
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
        return float(v)

    def _normalize_answer(self, ans: Any) -> Optional[str]:
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

    def parse_answer_confidence(self, text: str) -> Tuple[Optional[str], Optional[float]]:
        if not isinstance(text, str) or not text.strip():
            return None, None

        raw = text.strip()
        answer_keys = ("final_answer", "answer", "candidate")
        conf_keys = ("confidence", "probability")

        def pick(obj: Any) -> Tuple[Optional[str], Optional[float]]:
            if not isinstance(obj, dict):
                return None, None

            ans = next((obj[k] for k in answer_keys if k in obj), None)
            conf = next((obj[k] for k in conf_keys if k in obj), None)

            ans = self._normalize_answer(ans)
            conf = self._normalize_confidence(conf)
            return (ans, conf) if ans is not None and conf is not None else (None, None)

        def parse_obj(s: str) -> Optional[Any]:
            for loader in (json.loads, ast.literal_eval):
                try:
                    return loader(s)
                except Exception:
                    pass
            return None

        obj: Any = raw
        for _ in range(2):
            if not isinstance(obj, str):
                break

            obj = parse_obj(obj.strip())
            ans, conf = pick(obj)
            if ans is not None and conf is not None:
                return ans, conf

        ans_pat = r"""["']?(final_answer|answer|candidate)["']?\s*:\s*(?P<ans>"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[^\n,}]+)"""
        conf_pat = r"""["']?(confidence|probability)["']?\s*:\s*["']?(?P<conf>[-+]?\d*\.?\d+)["']?"""

        m_ans = re.search(ans_pat, raw, flags=re.S)
        m_conf = re.search(conf_pat, raw, flags=re.S)

        if m_ans and m_conf:
            ans_raw = m_ans.group("ans").strip().strip("\"'“”‘’")
            ans = self._normalize_answer(ans_raw)
            conf = self._normalize_confidence(m_conf.group("conf"))
            if ans is not None and conf is not None:
                return ans, conf

        m_ans = re.search(
            r"(?ims)^\s*(final\s+answer|answer)\s*:\s*(?P<ans>.*?)(?=^\s*(confidence|reasoning|explanation|analysis)\s*:|\Z)",
            raw,
        )
        m_conf = re.search(
            r"(?im)^\s*confidence\s*:\s*(?P<conf>[-+]?\d*\.?\d+)\s*%?\s*$",
            raw,
        )

        if m_ans and m_conf:
            ans = self._normalize_answer(m_ans.group("ans").strip().strip("\"'“”‘’"))
            conf = self._normalize_confidence(m_conf.group("conf"))
            if ans is not None and conf is not None:
                return ans, conf

        return None, None


    def compute_confidence_consistency(self, confidences: List[float]) -> Tuple[float, float, float]:

        if not confidences:
            return 0.0, 0.0, 0.0

        mu_c = float(np.mean(confidences))
        sigma_c = float(np.std(confidences))

        if mu_c <= 1e-12:
            kappa_conf = 0.0
        else:
            kappa_conf = float(1.0 / (1.0 + sigma_c / mu_c))

        return mu_c, sigma_c, kappa_conf

    def compute_answer_consistency(self, answers, item, tokenizer, nli_model):

        if not answers:
            return 0.0, None, {}

        dedup_answers = []
        ans2did = {}
        orig2dedup = []

        for a in answers:
            if a not in ans2did:
                ans2did[a] = len(dedup_answers)
                dedup_answers.append(a)
            orig2dedup.append(ans2did[a])

        if len(dedup_answers) == 1:
            return 1.0, dedup_answers[0], {dedup_answers[0]: len(answers)}

        if self.dataset_name == "truthfulqa_mc":
            counts = Counter(answers)
            majority_answer, majority_count = counts.most_common(1)[0]
            kappa_ans = float(majority_count / len(answers))
            return kappa_ans, majority_answer, dict(counts)

        question = item.get("question", None)
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

        dedup2cid = {}
        for cid, clu in enumerate(clusters):
            for did in clu:
                dedup2cid[did] = cid

        cluster_counts = [0 for _ in range(len(clusters))]
        for did in orig2dedup:
            cid = dedup2cid[did]
            cluster_counts[cid] += 1

        best_cid = int(np.argmax(cluster_counts))
        best_count = cluster_counts[best_cid]
        kappa_ans = float(best_count / len(answers))

        rep_did = clusters[best_cid][0]
        representative_answer = dedup_answers[rep_did]

        counts = {}
        for cid, clu in enumerate(clusters):
            rep = dedup_answers[clu[0]]
            counts[rep] = cluster_counts[cid]

        return kappa_ans, representative_answer, counts

    def select_answer_by_calibrated_confidence(
        self,
        valid_samples: List[Dict[str, Any]],
        calibrated_conf: float,
    ) -> Optional[str]:

        if not valid_samples:
            return None

        confs = [float(s["confidence"]) for s in valid_samples]
        answers = [s["answer"] for s in valid_samples]

        cmin = min(confs)
        cmax = max(confs)

        if cmax - cmin < 1e-12:
            mid = len(valid_samples) // 2
            return answers[mid]

        n = len(valid_samples)
        j = int(((calibrated_conf - cmin) / (cmax - cmin)) * (n))
        j = max(0, min(n - 1, j))
        return answers[j]

    def select_answer_by_majority_with_conf(
        self,
        valid_samples: List[Dict[str, Any]],
    ) -> Optional[str]:

        if not valid_samples:
            return None

        answer_to_confs: Dict[str, List[float]] = {}
        for s in valid_samples:
            a = s["answer"]
            c = float(s["confidence"])
            answer_to_confs.setdefault(a, []).append(c)

        max_count = max(len(v) for v in answer_to_confs.values())
        candidates = [a for a, v in answer_to_confs.items() if len(v) == max_count]

        if len(candidates) == 1:
            return candidates[0]

        best_a = None
        best_mean = -1.0
        for a in candidates:
            m = float(np.mean(answer_to_confs[a]))
            if m > best_mean:
                best_mean = m
                best_a = a
        return best_a


    def generate(self, **kwargs: Any):

        messages = []
        message_meta = []

        for i, item in enumerate(self.dataset):
            for level in self.steering_levels:
                system = self.build_system_prompt(item, self.steering_instruction[level])
                if self.dataset_name == "coqa":
                    user = self.build_coqa_user_prompt(item)
                else:
                    user = self.build_user_prompt(item)

                messages.append([
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ])
                message_meta.append((i, level))

        if self.show:
            print(messages[0:5])

        texts = self.model.generate_batch(messages, temperature=self.temperature, **kwargs)

        for item in self.dataset:
            item["sample_results"] = []

        for (item_idx, level), text in zip(message_meta, texts):
            sample = {
                "steering_level": level,
                "text": text,
            }
            self.dataset[item_idx]["sample_results"].append(sample)

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

        return dataset

    def calculate(self):

        with open(self.save_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        tokenizer, nli_model = load_nli_model(config.NLI_MODEL_PATH)
        for item in tqdm(dataset):
            samples = item.get("sample_results", [])

            valid_samples = []
            for s in samples:
                answer = s.get("answer", None)
                conf = s.get("confidence", None)

                if conf is not None and answer is not None:
                    valid_samples.append({
                        "steering_level": s.get("steering_level", None),
                        "answer": answer,       
                        "confidence": conf,
                    })

            valid_samples = sorted(
                valid_samples,
                key=lambda s: self.level_order.get(s.get("steering_level", ""), 999)
            )

            if len(valid_samples)<4:
                item["mu_c"] = None
                item["sigma_c"] = None
                item["kappa_conf"] = None
                item["kappa_ans"] = None
                item["pred_score"] = None
                item["pred_answer"] = None
                item["answer_counts"] = {}
                continue

            confidences = [float(s["confidence"]) for s in valid_samples]
            answers = [s["answer"] for s in valid_samples]
            mu_c, sigma_c, kappa_conf = self.compute_confidence_consistency(confidences)
            kappa_ans, majority_answer, answer_counts = self.compute_answer_consistency(answers=answers, item=item, tokenizer=tokenizer, nli_model=nli_model)
            calibrated_conf = float(mu_c * kappa_ans * kappa_conf)
            answer_quantized = self.select_answer_by_calibrated_confidence(valid_samples, calibrated_conf)
            item["mu_c"] = float(mu_c)
            item["sigma_c"] = float(sigma_c)
            item["kappa_conf"] = float(kappa_conf)
            item["kappa_ans"] = float(kappa_ans)
            item["pred_score"] = float(calibrated_conf)
            item["pred_answer"] = answer_quantized
            item["answer_counts"] = answer_counts

        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)

        return dataset
