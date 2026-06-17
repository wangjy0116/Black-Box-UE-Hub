from typing import Any, Dict, List, Optional, Tuple
import ast
import os
import re
import json
from tqdm import tqdm
from copy import deepcopy
from sentence_transformers import SentenceTransformer
import numpy as np
from src import config 
from src.utils import build_sample_path, extract_final_answer, sample_generate

class SPUQ:

    def __init__(self, model_name, model, dataset_name, dev_dataset, test_dataset, sample_num, temperature):
        self.model_name = model_name
        self.model = model
        self.dataset_name = dataset_name
        self.dataset = test_dataset
        self.sample_num = sample_num
        self.temperature = temperature
        self.embedder = SentenceTransformer(config.SBERT_MODEL_PATH)
        self.save_path = f"{config.OUTPUT_DIR}/spuq/{dataset_name}/{model_name}.json"
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

    def build_user_prompt(self, item):
        return f"Question: {item['question']}"
        
    def build_coqa_user_prompt(self, item):
        return f"Context: {item['story']}\nQuestion: {item['question']}"

    def text_sim(self, a: str, b: str) -> float:
        embs = self.embedder.encode([a, b])
        norm = np.sqrt((embs * embs).sum(-1))
        norm_embs = embs / norm.reshape(-1, 1)
        cos_sim = (norm_embs[0] * norm_embs[1]).sum(-1)
        return cos_sim

    def build_paraphrase_messages(self, item, para_num) -> List[Dict[str, str]]:
        system = "You are a helpful assistant that paraphrases questions."
        question = item["question"]
        user = f'Provide {para_num} paraphrases for this question: {question}. Do NOT answer the question. Return ONLY a valid JSON array of strings.'
        if self.dataset_name == "coqa":
            user = f"Context: {item['story']}\n" + user
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    
    def _post_process_paraphrases(self, texts: List[str], item: Dict[str, Any], para_num: int):
        original_question = item["question"].strip()
        cleaned = []

        for t in texts:
            if not isinstance(t, str):
                continue
            t = t.strip()

            t = re.sub(r'^(sentence|paraphrase)\s*:\s*', '', t, flags=re.IGNORECASE).strip()

            if not t:
                continue
            if t == original_question:
                continue
            if len(t) < 3:
                continue

            cleaned.append(t)

        # dedup keep order
        seen = set()
        out = []
        for t in cleaned:
            key = t.lower()
            if key not in seen:
                seen.add(key)
                out.append(t)

        return out[:para_num]

    def parse_paraphrase_output(self, text: str, item: Dict[str, Any], para_num: int) -> List[str]:
        if not isinstance(text, str):
            return []

        raw = text.strip()
        if not raw:
            return []

        try:
            obj = json.loads(raw)
            if isinstance(obj, list):
                out = [str(x).strip() for x in obj if str(x).strip()]
                return self._post_process_paraphrases(out, item, para_num)
        except Exception:
            pass

        try:
            obj = ast.literal_eval(raw)
            if isinstance(obj, list):
                out = [str(x).strip() for x in obj if str(x).strip()]
                return self._post_process_paraphrases(out, item, para_num)
        except Exception:
            pass

        list_candidates = re.findall(r"\[[\s\S]*?\]", raw)
        for cand in list_candidates:
            # try json first
            try:
                obj = json.loads(cand)
                if isinstance(obj, list):
                    out = [str(x).strip() for x in obj if str(x).strip()]
                    parsed = self._post_process_paraphrases(out, item, para_num)
                    if parsed:
                        return parsed
            except Exception:
                pass

            try:
                obj = ast.literal_eval(cand)
                if isinstance(obj, list):
                    out = [str(x).strip() for x in obj if str(x).strip()]
                    parsed = self._post_process_paraphrases(out, item, para_num)
                    if parsed:
                        return parsed
            except Exception:
                pass

        raw2 = raw
        raw2 = re.sub(
            r'^\s*(sure[,! ]*)?(here are|below are|the paraphrases are|paraphrases)\s*:?\s*',
            '',
            raw2,
            flags=re.IGNORECASE,
        ).strip()

        raw2 = re.sub(
            r'^\s*I will provide\s+\d+\s+paraphrases\s*:?\s*',
            '',
            raw2,
            flags=re.IGNORECASE,
        ).strip()

        lines = [x.strip() for x in raw2.split("\n") if x.strip()]
        out = []

        for line in lines:
            line = re.sub(r'^\s*\d+[\.\)\-:]\s*', '', line)
            line = re.sub(r'^\s*[\-\*\•]\s*', '', line)

            lower = line.lower().strip()

            if lower in {
                "paraphrases:",
                "here are the paraphrases:",
                "sure, here are the paraphrases:",
                "here are some paraphrases:",
            }:
                continue

            line = line.strip().strip('"').strip("'").strip()
            if line:
                out.append(line)

        parsed = self._post_process_paraphrases(out, item, para_num)
        if parsed:
            return parsed


        if ";" in raw2:
            parts = [p.strip().strip('"').strip("'").strip() for p in raw2.split(";")]
            parsed = self._post_process_paraphrases(parts, item)
            if parsed:
                return parsed

        quoted = re.findall(r'"([^"\n]+)"|\'([^\'\n]+)\'', raw2)
        if quoted:
            flat = []
            for a, b in quoted:
                s = a if a else b
                s = s.strip()
                if s:
                    flat.append(s)
            parsed = self._post_process_paraphrases(flat, item)
            if parsed:
                return parsed

        parts = re.split(r'\s+(?=\d+[\.\)])', raw2)
        if len(parts) > 1:
            cleaned = []
            for p in parts:
                p = re.sub(r'^\s*\d+[\.\)]\s*', '', p).strip()
                p = p.strip('"').strip("'").strip()
                if p:
                    cleaned.append(p)
            parsed = self._post_process_paraphrases(cleaned, item)
            if parsed:
                return parsed

        return []

    def generate(self, **kwargs: Any):
        para_num = self.sample_num - 1
        sample_test_path, _ = build_sample_path(self.save_path)

        if not os.path.exists(sample_test_path):
            build_user = self.build_coqa_user_prompt if self.dataset_name == "coqa" else self.build_user_prompt
            sample_generate(
                model=self.model,
                dataset=self.dataset,
                save_path=sample_test_path,
                build_system=self.build_system_prompt,
                build_user=build_user,
                sample_num=self.sample_num,
                temperature=self.temperature,
                **kwargs,
            )

        with open(sample_test_path, "r", encoding="utf-8") as f:
            sample_test_dataset = json.load(f)

        if "perturbed_results" not in sample_test_dataset[0]:
            para_messages_batch = [self.build_paraphrase_messages(item, para_num) for item in self.dataset]

            if self.show:
                print(para_messages_batch[0])

            para_outputs = self.model.generate_batch(para_messages_batch, temperature=0.1, **kwargs)

            for item, para in zip(sample_test_dataset, para_outputs):
                item["para_outputs"] = para
                paraphrases = self.parse_paraphrase_output(para, item, para_num)
                paraphrases = paraphrases[:para_num]
                item["perturbations"] = [item["question"]] + paraphrases

            answer_messages_batch = []
            answer_meta = []
            for idx, item in enumerate(sample_test_dataset):
                for pert in item["perturbations"]:
                    if self.dataset_name == "coqa":
                        prompt_text = f"Context: {item['story']}\nQuestion: {pert}"
                    else:
                        prompt_text = f"Question: {pert}"

                    msg = [
                        {"role": "system", "content": self.build_system_prompt(item)},
                        {"role": "user", "content": prompt_text},
                    ]
                    answer_messages_batch.append(msg)
                    answer_meta.append((idx, pert))

            if self.show:
                print(answer_messages_batch[0])

            answer_outputs = self.model.generate_batch(answer_messages_batch, temperature=self.temperature, **kwargs)

            for item in sample_test_dataset:
                item["perturbed_results"] = []

            for answer, (idx, pert_question) in zip(answer_outputs, answer_meta):
                rec = {
                    "perturbed_question": pert_question,
                    "text": answer,
                }
                sample_test_dataset[idx]["perturbed_results"].append(rec)

            with open(sample_test_path, "w", encoding="utf-8") as f:
                json.dump(sample_test_dataset, f, ensure_ascii=False, indent=4)
           
    def extract(self):
        sample_test_path, _ = build_sample_path(self.save_path)

        with open(sample_test_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        for item in tqdm(dataset):
            samples = item.get("perturbed_results", [])
            for s in samples:
                if "answer" not in s.keys():
                    text = s.get("text", "")
                    ans = extract_final_answer(text)
                    s["answer"] = ans
            
            if "answer" not in item["greedy_results"].keys():
                text = item["greedy_results"].get("text", "")
                ans = extract_final_answer(text)
                item["greedy_results"]["answer"] = ans

        with open(sample_test_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)

    def calculate(self):
        sample_test_path, _ = build_sample_path(self.save_path)
        with open(sample_test_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        output = []
        spuq_scores = []
        for item in tqdm(dataset):
            sample_results = item["perturbed_results"]
            origin_question = item["question"]

            sum_wt = 0.0
            sum_conf = 0.0
            for s in sample_results:
                perturbed_question = s.get("perturbed_question")
                perturbed_answer = s.get("answer")

                if perturbed_question is None or perturbed_answer is None:
                    continue

                wt = self.text_sim(origin_question, perturbed_question)
                conf = self.text_sim(item["greedy_results"]["answer"], perturbed_answer)
                sum_conf += conf * wt
                sum_wt += wt

            spuq_score = (sum_conf / sum_wt) if sum_wt > 0 else None
            spuq_scores.append(spuq_score)

        for item, spuq_score in zip(dataset, spuq_scores):
            if spuq_score is not None and item["greedy_results"]["answer"] is not None:
                output.append({"id": item["id"], "question": item["question"],
                    "description": item["description"], "ground_truth": item["ground_truth"],
                    "pred_answer": item["greedy_results"]["answer"],
                    "pred_score": spuq_score})
            else:
                output.append({"id": item["id"], "question": item["question"],
                    "description": item["description"], "ground_truth": item["ground_truth"],
                    "pred_answer": item["greedy_results"]["answer"],
                    "pred_score": None})

            if "story" in item.keys():
                output[-1]["story"] = item["story"]


        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=4)

