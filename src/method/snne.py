from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple
import os
import re
import json
import copy
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from openai import OpenAI
from judge import api_base, api_key, build_judge_prompt, judge_one, match_string
from src.utils import compute_sample_semantic, build_sample_path, sample_generate, extract_final_answer, load_nli_model
from src import config

class SNNE:

    def __init__(
        self,
        model_name, 
        model, 
        dataset_name,
        dev_dataset,
        test_dataset,
        sample_num, 
        temperature,
        variant: str = "only_denom",
        selfsim: bool = True,           
        weighted: bool = False,        
        jitter: float = 0.0,            
    ):
        self.variant = str(variant)
        self.selfsim = bool(selfsim)
        self.weighted = bool(weighted)
        self.jitter = float(jitter)
        self.model_name = model_name
        self.model = model
        self.dataset_name = dataset_name
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.sample_num = sample_num
        self.temperature = temperature
        self.save_path = f"{config.OUTPUT_DIR}/snne/{dataset_name}/{model_name}.json"
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        self.show = True
        self.system_prompt = """Read the following question and reason step-by-step to formulate your final answer. {description}

Output ONLY a single JSON object with fields in this exact order:
{{
  "reasoning": "Your step-by-step analysis",
  "final_answer": "Your final answer"
}}"""

    def build_system_prompt(self, item: Dict[str, Any]) -> str:
        return self.system_prompt.format(description=item["description"])

    def build_user_prompt(self, item: Dict[str, Any]) -> str:
        return f"Question: {item['question']}"

    def build_coqa_user_prompt(self, item: Dict[str, Any]) -> str:
        return f"Context: {item['story']}\nQuestion: {item['question']}"

    def judge_and_cache_labels(
        self,
        cache_path: str,
        eval_model: str = "gpt-5.1",
        reasoning_effort: str = "low",
        max_workers: int = 32,
        max_retries: int = 5,
        timeout: int = 60,
        base_backoff: float = 1.0,
    ) -> None:
        if not os.path.exists(cache_path):
            return

        with open(cache_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        api_results: Dict[int, float] = {}
        tasks: List[Tuple[int, str]] = []
        client = None

        for idx, rec in enumerate(dataset):
            pred_answer = rec["greedy_results"].get("answer", None)
            if pred_answer is None or pred_answer == "":
                continue

            if "label" in rec.keys() and rec["label"] is not None:
                continue

            ground_truth = rec.get("ground_truth", [])
            if isinstance(ground_truth, str):
                ground_truth = [ground_truth]

            if self.dataset_name == "truthfulqa_mc":
                rec["label"] = 1.0 if match_string(pred_answer, ground_truth) else 0.0
                continue

            if match_string(pred_answer, ground_truth):
                rec["label"] = 1.0
                continue

            prompt = build_judge_prompt(rec, ground_truth, pred_answer)
            tasks.append((idx, prompt))

        if tasks:
            client = OpenAI(
                api_key=api_key,
                base_url=api_base,
                timeout=timeout,
            )

            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                fut2idx = {}
                for idx, prompt in tasks:
                    fut = ex.submit(
                        judge_one,
                        client,
                        eval_model,
                        reasoning_effort,
                        prompt,
                        max_retries,
                        base_backoff,
                    )
                    fut2idx[fut] = idx

                for fut in tqdm(as_completed(fut2idx), total=len(fut2idx)):
                    idx = fut2idx[fut]
                    try:
                        txt = fut.result()
                    except Exception:
                        txt = None

                    if not txt:
                        continue

                    reply = txt.strip().lower()
                    api_results[idx] = 1.0 if reply.startswith("yes") else 0.0

        for idx, rec in enumerate(dataset):
            if rec.get("label", None) is None:
                rec["label"] = api_results.get(idx, rec.get("label", None))

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)

    @staticmethod
    def _logsumexp(x: np.ndarray, axis: int = -1) -> np.ndarray:
        m = np.max(x, axis=axis, keepdims=True)
        y = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True) + 1e-12)
        return np.squeeze(y, axis=axis)

    @staticmethod
    def uf_labels_from_classmat(class_mat: np.ndarray) -> np.ndarray:
        K = class_mat.shape[0]
        parent = list(range(K))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i in range(K):
            for j in range(i + 1, K):
                if class_mat[i, j] == 0:
                    union(i, j)

        roots = [find(i) for i in range(K)]
        uniq = {}
        labels = np.zeros(K, dtype=np.int64)
        cur = 0
        for i, r in enumerate(roots):
            if r not in uniq:
                uniq[r] = cur
                cur += 1
            labels[i] = uniq[r]
        return labels

    def snne_score_from_similarity(
        self,
        S: np.ndarray,
        labels: np.ndarray,
        temperature: float,
        variant: str,
        selfsim: bool,
    ) -> float:
        """
        Compute (a variant of) SNNE loss from similarity matrix S and cluster labels.
        Return a scalar uncertainty score (higher typically = more uncertain, depending on variant).
        """
        assert S.ndim == 2 and S.shape[0] == S.shape[1]
        K = S.shape[0]
        if K <= 1:
            return 0.0
        Z = S / max(1e-12, float(temperature))

        if not selfsim:
            Z = Z.copy()
            np.fill_diagonal(Z, -1e9)

        denom = self._logsumexp(Z, axis=1)
        num = np.full((K,), -1e9, dtype=np.float64)
        for i in range(K):
            mask = (labels == labels[i])
            if not selfsim:
                mask = mask & (np.arange(K) != i)
            if np.any(mask):
                num[i] = self._logsumexp(Z[i, mask], axis=0)
            else:
                num[i] = -1e9

        if variant == "full" or variant == "num_minus_denom":
            loss = -(num - denom).mean()
        elif variant == "only_num":
            loss = -num.mean()
        elif variant == "only_denom":
            loss = -denom.mean()
        else:
            raise ValueError(f"Unknown variant: {variant}")

        return float(loss)

    def compute_and_cache_semantics(
        self,
        dataset: List[Dict[str, Any]],
        path: str
    ) -> None:

        if "semantic_matrix_entail" in dataset[0].keys():
            return
        tokenizer, nli_model = load_nli_model(config.NLI_MODEL_PATH)
        valid_num = self.sample_num // 2
        for item in tqdm(dataset):
            question = item.get("question", "")

            kept_sample_indices = []
            sample_texts = []

            for idx, s in enumerate(item.get("sample_results", [])[: self.sample_num]):
                ans = s.get("answer", None)
                if ans is None:
                    continue
                if not isinstance(ans, str):
                    ans = str(ans)
                ans_strip = ans.strip()
                if not ans_strip:
                    continue

                kept_sample_indices.append(idx)
                sample_texts.append(ans_strip)

            item["sample_indices"] = kept_sample_indices

            if len(sample_texts) <= valid_num:
                item["semantic_matrix_entail"] = []
                item["semantic_matrix_contra"] = []
                item["class_mat"] = []
                continue

            E, C, N, E_logits, C_logits, N_logits, class_mat, _ = compute_sample_semantic(
                question = question,
                sample_texts = sample_texts,
                tokenizer = tokenizer,
                nli_model = nli_model
            )

            item["semantic_matrix_entail"] = E.tolist()
            item["semantic_matrix_contra"] = C.tolist()
            item["class_mat"] = class_mat.tolist()


    def cal_auroc(self, y_true: List[int], y_score: List[float]) -> float:
        y_true = np.asarray(y_true, dtype=np.int64)
        y_score = np.asarray(y_score, dtype=np.float64)
        return float(roc_auc_score(y_true, y_score))

    def compute_snne_for_dataset(
        self,
        dataset: List[Dict[str, Any]],
        path: str,
        temperature: float,
        variant: str,
        exclude_diagonal: bool,
    ) -> List[Optional[float]]:
        scores: List[Optional[float]] = []

        for item in dataset:
            E_list = item.get("semantic_matrix_entail", [])
            C_list = item.get("semantic_matrix_contra", [])
            class_list = item.get("class_mat", [])

            if not E_list or not C_list or not class_list:
                scores.append(None)
                continue

            E = np.asarray(E_list, dtype=np.float32)
            C = np.asarray(C_list, dtype=np.float32)
            class_mat = np.asarray(class_list, dtype=np.float32)

            if E.ndim != 2 or E.shape[0] <= min(1, self.sample_num // 2) or E.shape[0] != E.shape[1]:
                scores.append(None)
                continue

            S = (E+E.T)/2
            np.fill_diagonal(S, 1.0)
            labels = self.uf_labels_from_classmat(class_mat)

            base = self.snne_score_from_similarity(
                S=S, labels=labels, temperature=temperature, variant=variant, selfsim=exclude_diagonal
            )
            scores.append(float(base))

        return scores

    def tune_on_dev(
        self,
        dev_path: str,
        temperature_grid: List[float],
        variant_grid: List[str],
        selfsim_grid: List[bool],
        metric: str = "auroc",
    ) -> Dict[str, Any]:

        with open(dev_path, "r", encoding="utf-8") as f:
            dev_data = json.load(f)

        self.compute_and_cache_semantics(dev_data, dev_path)
        with open(dev_path, "w", encoding="utf-8") as f:
            json.dump(dev_data, f, ensure_ascii=False, indent=4)

        best = None
        best_cfg = None

        for T in temperature_grid:
            for var in variant_grid:
                for ss in selfsim_grid:
                    scores = self.compute_snne_for_dataset(
                            dev_data, dev_path, temperature=T, variant=var, exclude_diagonal=not ss
                        )
                    y_score = []
                    y_label = []
                    for item, sc in zip(dev_data, scores):
                        if sc is None or np.isnan(sc):
                            continue
                        y = item.get("label", None)
                        if y is None:
                            continue

                        y_score.append(float(sc))           
                        y_label.append(int(1-y))

                    if len(set(y_label)) < 2:
                        continue

                    if metric == "auroc":
                        val = self.cal_auroc(y_label, y_score)
                    else:
                        raise ValueError(f"Unsupported metric: {metric}")

                    if (best is None) or (val > best):
                        best = val
                        best_cfg = dict(temperature=T, variant=var, selfsim=ss)

        return {"best_metric": best, "best_cfg": best_cfg, "metric": metric}

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

    def calculate(self) -> None:
        temperature_choice = [0.1, 1, 10, 100]
        variant_choice = ['only_denom']
        selfsim_choice = [True]
        metric = "auroc"

        sample_test_path, sample_dev_path = build_sample_path(self.save_path)

        self.judge_and_cache_labels(sample_dev_path)

        tune_res = self.tune_on_dev(
            dev_path=sample_dev_path,
            temperature_grid=temperature_choice,
            variant_grid=variant_choice,
            selfsim_grid=selfsim_choice,
            metric=metric,
        )

        best_cfg = tune_res["best_cfg"]
        if best_cfg is None:
            best_cfg = dict(temperature=1.0, variant="only_denom", selfsim=True)


        with open(sample_test_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)

        self.compute_and_cache_semantics(test_data, sample_test_path)
        with open(sample_test_path, "w", encoding="utf-8") as f:
            json.dump(test_data, f, ensure_ascii=False, indent=4)

        snne_test = self.compute_snne_for_dataset(
            test_data,
            sample_test_path,
            temperature=best_cfg["temperature"],
            variant=best_cfg["variant"],
            exclude_diagonal=not best_cfg["selfsim"]
        )

        with open(sample_dev_path, "r", encoding="utf-8") as f:
            dev_data = json.load(f)

        snne_dev = self.compute_snne_for_dataset(
            dev_data,
            sample_dev_path,
            temperature=best_cfg["temperature"],
            variant=best_cfg["variant"],
            exclude_diagonal= not best_cfg["selfsim"]
        )

        valid_dev = [float(x) for x in snne_dev if x is not None and not np.isnan(x)]
        if len(valid_dev) == 0:
            u_min, u_max = 0.0, 1.0
        else:
            u_min, u_max = float(np.min(valid_dev)), float(np.max(valid_dev))
        denom = (u_max - u_min)

        def minmax_u(u: float) -> float:
            if denom <= 1e-12:
                return 0.0
            v = (u - u_min) / denom
            if v < 0.0:
                v = 0.0
            if v > 1.0:
                v = 1.0
            return float(v)

        output = []
        for item, u in zip(test_data, snne_test):
            greedy_ans = None
            if isinstance(item.get("greedy_results", None), dict):
                greedy_ans = item["greedy_results"].get("answer", None)

            if u is not None and greedy_ans is not None and not (isinstance(greedy_ans, str) and greedy_ans.strip() == ""):
                u_norm = minmax_u(float(u))     
                conf = 1.0 - u_norm            
            else:
                conf = None

            row = {
                "id": item.get("id"),
                "question": item.get("question"),
                "description": item.get("description"),
                "ground_truth": item.get("ground_truth"),
                "pred_answer": greedy_ans,
                "snne_uncertainty": u,
                "pred_score": conf,
                "snne_dev_tuning": tune_res,
            }
            if self.dataset_name == "coqa":
                row["story"] = item["story"]
            output.append(row)

        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=4)