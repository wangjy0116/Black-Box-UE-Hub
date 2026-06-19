from typing import Any, Dict, List, Optional, Tuple
import ast
import json
import os
import re
import numpy as np
from tqdm import tqdm
from scipy.stats import wasserstein_distance
from src import config
from src.utils import (
    build_sample_path,
    extract_final_answer,
    sample_generate
)
from sentence_transformers import SentenceTransformer
import torch


class InvE:
    def __init__(
        self,
        model_name: str,
        model: Any,
        dataset_name: str,
        dev_dataset: List[Dict[str, Any]],
        test_dataset: List[Dict[str, Any]],
        sample_num: int,
        temperature: float,
    ):
        self.model_name = model_name
        self.model = model
        self.dataset_name = dataset_name
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.sample_num = sample_num
        self.temperature = temperature
        self.dataset = test_dataset
        self.save_path = f"{config.OUTPUT_DIR}/inve/{dataset_name}/{model_name}.json"
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

        self.show = True
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedder = SentenceTransformer(config.SBERT_MODEL_PATH)
        self.embedder = self.embedder.to(self.device)
        self.system_prompt = """Read the following question and reason step-by-step to formulate your final answer. {description}

Output ONLY a single JSON object with fields in this exact order:
{{
  "reasoning": "Your step-by-step analysis",
  "final_answer": "Your final answer"
}}"""

    def build_system_prompt(self, item: Dict[str, Any]) -> str:
        return self.system_prompt.format(description=item.get("description", ""))

    def build_user_prompt(self, item: Dict[str, Any]) -> str:
        return f"Question: {item['question']}"

    def build_coqa_user_prompt(self, item: Dict[str, Any]) -> str:
        return f"Context: {item['story']}\nQuestion: {item['question']}"

    def build_paraphrase_messages(self, item: Dict[str, Any], para_num: int) -> List[Dict[str, str]]:
        system = "You are a helpful assistant that paraphrases questions."
        question = item["question"]

        user = (
            f"Provide {para_num} paraphrases for this question: {question}. "
            "Do NOT answer the question. Return ONLY a valid JSON array of strings."
        )

        if self.dataset_name == "coqa":
            user = f"Context: {item['story']}\n{user}"

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def parse_matrix(self,matrix_str):
        return np.array(ast.literal_eval(matrix_str))

    def row_normalize(self, matrix):
        row_sums = matrix.sum(axis=1, keepdims=True)
        return matrix / row_sums

    def compute_entropy(self, prob_vector):
        prob_vector = prob_vector[prob_vector > 0]  
        return -np.sum(prob_vector * np.log2(prob_vector))

    def compute_conditional_entropy(self, prob_matrix, axis):
        entropy_values = np.zeros(prob_matrix.shape[axis])
        
        if axis == 1:  
            for i in range(prob_matrix.shape[0]):
                row = prob_matrix[i, :]
                entropy_values[i] = self.compute_entropy(row)
        else:  
            for j in range(prob_matrix.shape[1]):
                col = prob_matrix[:, j]
                entropy_values[j] = self.compute_entropy(col)
        
        return np.sum(entropy_values)  

    def compute_joint_entropy(self, pxy):
        pxy_nonzero = pxy[pxy > 0]  
        return -np.sum(pxy_nonzero * np.log2(pxy_nonzero))

    def kl_divergence(self, p, q):
        p = p.flatten()
        q = q.flatten()
        mask = (p > 0) & (q > 0)  
        return np.sum(p[mask] * np.log2(p[mask] / q[mask]))

    def js_divergence(self, p, q):
        m = 0.5 * (p + q)
        return 0.5 * (self.kl_divergence(p, m) + self.kl_divergence(q, m))

    def compute_cosine_matrix(self, texts):
        if len(texts) == 0:
            return np.zeros((0, 0), dtype=np.float32)

        texts = [str(t).strip() for t in texts]

        if len(texts) == 1:
            return np.ones((1, 1), dtype=np.float32)

        unique_texts = list(dict.fromkeys(texts))

        emb_unique = self.embedder.encode(
            unique_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

        text2emb = {
            text: emb_unique[i]
            for i, text in enumerate(unique_texts)
        }

        n = len(texts)
        sim = np.zeros((n, n), dtype=np.float32)

        for i in range(n):
            for j in range(i, n):
                similarity = float(np.dot(text2emb[texts[i]], text2emb[texts[j]]))
                similarity = abs(similarity)
                sim[i, j] = similarity
                sim[j, i] = similarity

        np.fill_diagonal(sim, 1.0)
        return sim

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
            parsed = self._post_process_paraphrases(parts, item, para_num)
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
            parsed = self._post_process_paraphrases(flat, item, para_num)
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
            parsed = self._post_process_paraphrases(cleaned, item, para_num)
            if parsed:
                return parsed

        return []

    def generate(self, **kwargs: Any) -> None:
        para_num = self.sample_num - 1
        sample_test_path, sample_dev_path = build_sample_path(self.save_path)

        build_user = self.build_coqa_user_prompt if self.dataset_name == "coqa" else self.build_user_prompt

        def process_split(
            split_dataset: List[Dict[str, Any]],
            save_path: str,
        ) -> List[Dict[str, Any]]:
            if not os.path.exists(save_path):
                sample_generate(
                    model=self.model,
                    dataset=split_dataset,
                    save_path=save_path,
                    build_system=self.build_system_prompt,
                    build_user=build_user,
                    sample_num=self.sample_num,
                    temperature=self.temperature,
                    **kwargs,
                )

            with open(save_path, "r", encoding="utf-8") as f:
                sample_dataset = json.load(f)

            if "perturbed_results" in sample_dataset[0]:
                return sample_dataset

            para_messages_batch = [self.build_paraphrase_messages(item, para_num) for item in split_dataset]

            if self.show:
                print(para_messages_batch[0])

            para_outputs = self.model.generate_batch(para_messages_batch, temperature=0.1, **kwargs)

            for item, para in zip(sample_dataset, para_outputs):
                item["para_outputs"] = para
                paraphrases = self.parse_paraphrase_output(para, item, para_num)
                paraphrases = paraphrases[:para_num]
                item["perturbations"] = [item["question"]] + paraphrases

            answer_messages_batch = []
            answer_meta = []

            for idx, item in enumerate(sample_dataset):
                for pert in item.get("perturbations", []):
                    if self.dataset_name == "coqa":
                        prompt_text = f"Context: {item['story']}\nQuestion: {pert}"
                    else:
                        prompt_text = f"Question: {pert}"

                    message = [
                        {"role": "system", "content": self.build_system_prompt(item)},
                        {"role": "user", "content": prompt_text}
                    ]
                    answer_messages_batch.append(message)
                    answer_meta.append((idx, pert))

            if self.show:
                print(answer_messages_batch[0])

            answer_outputs = self.model.generate_batch(answer_messages_batch, temperature=self.temperature, **kwargs)

            for item in sample_dataset:
                item["perturbed_results"] = []

            for answer, (idx, pert_question) in zip(answer_outputs, answer_meta):
                record = {
                    "perturbed_question": pert_question,
                    "text": answer,
                }
                sample_dataset[idx]["perturbed_results"].append(record)

            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(sample_dataset, f, ensure_ascii=False, indent=4)

            return sample_dataset

        process_split(self.test_dataset, sample_test_path)
        process_split(self.dev_dataset, sample_dev_path)

    def extract(self) -> None:
        sample_test_path, sample_dev_path = build_sample_path(self.save_path)

        def process_cache_file(cache_path: str):
            with open(cache_path, "r", encoding="utf-8") as f:
                dataset = json.load(f)

            for item in tqdm(dataset):
                greedy = item.get("greedy_results", None)
                if greedy is not None and "answer" not in greedy:
                    text = greedy.get("text", "")
                    greedy["answer"] = extract_final_answer(text)

                perturbed_results = item.get("perturbed_results", [])
                for perturbed in perturbed_results:
                    if "answer" not in perturbed:
                        text = perturbed.get("text", "")
                        perturbed["answer"] = extract_final_answer(text)

            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(dataset, f, ensure_ascii=False, indent=4)
        
        process_cache_file(sample_test_path)
        process_cache_file(sample_dev_path)


    def calculate_scores(self, path):

        with open(path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        results = []
        
        for item in tqdm(dataset):
            
            x = []  # question
            y = []  # answer
            
            for s in item["perturbed_results"]:
                question = s["perturbed_question"]
                answer = s["answer"]
                if answer is None:
                    answer = "none"
                x.append(question)
                y.append(answer)
            
            # compute similarity matrix
            simX = self.compute_cosine_matrix(x)
            simY = self.compute_cosine_matrix(y)

            # Calculate row normalization matrix
            px = self.row_normalize(simX)
            py = self.row_normalize(simY)
            
            # calculate n
            n = px.shape[0]
            
            # construct pi_uniform
            pi_uniform = np.full((1, n), 1/n)
            
            # compute px_marginal, px_y, py_marginal
            py_marginal_I = (pi_uniform @ py).T
            px_y_I = py @ px
            px_marginal_I = (pi_uniform @ py @ py @ px).T

            # compute py_x by baye's rule
            py_x_I = (px_y_I * py_marginal_I) / px_marginal_I.T

            # compute pxy
            pxy_I  = px_y_I * py_marginal_I

            #The index of x determines the row, and the index of y determines the column.
            px_y_I = px_y_I.T
            py_x_I = py_x_I.T
            pxy_I = pxy_I.T

            WD_px_py_I = wasserstein_distance(px_marginal_I.flatten(), py_marginal_I.flatten())
            entropy_y_x_I = -np.sum(np.diag(py_x_I) * np.log(np.diag(py_x_I)))
            entropy_x_y_I = -np.sum(np.diag(px_y_I) * np.log(np.diag(px_y_I)))
            max_y_I = max(py_marginal_I).item()
            results.append(entropy_x_y_I)

        return results


    def calculate(self):
        sample_test_path, sample_dev_path = build_sample_path(self.save_path)

        test_scores = self.calculate_scores(sample_test_path)
        dev_scores = self.calculate_scores(sample_dev_path)

        with open(sample_test_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        valid_scores_dev = [float(x) for x in dev_scores if x is not None and not np.isnan(x)]
        if len(valid_scores_dev) == 0:
            u_min, u_max = 0.0, 1.0
        else:
            u_min = float(np.min(valid_scores_dev))
            u_max = float(np.max(valid_scores_dev))
        denom = u_max - u_min

        def minmax(x: float) -> float:
            if denom <= 1e-12:
                return 0.0
            v = (x - u_min) / denom
            if v < 0.0:
                v = 0.0
            if v > 1.0:
                v = 1.0
            return float(v)

        output = []

        for item, inv_entropy in zip(dataset, test_scores):
            greedy_ans = item.get("greedy_results", {}).get("answer", None)

            record = {
                "id": item.get("id", None),
                "question": item.get("question", ""),
                "description": item.get("description", ""),
                "ground_truth": item.get("ground_truth", None),
                "pred_answer": greedy_ans,
                "inve_score": None,
                "pred_score": None,
            }

            if inv_entropy is not None and greedy_ans is not None:
                inv_entropy_norm = minmax(float(inv_entropy))
                record["inve_score"] = float(inv_entropy)
                record["pred_score"] = float(1.0 - inv_entropy_norm)

            if "story" in item:
                record["story"] = item["story"]

            output.append(record)

        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=4)

        return output