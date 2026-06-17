from typing import Any, Dict, List, Optional, Tuple
import os
import re
import json
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from src import config
from src.utils import build_sample_path, extract_final_answer, norm_mc, sample_generate


class UF:
    def __init__(
        self,
        model_name, 
        model, 
        dataset_name,
        dev_dataset,
        test_dataset,
        sample_num, 
        temperature,
        max_workers,
        tau: float = 2.0,
        max_chain_steps: int = 32
    ):
        self.tau = tau
        self.max_chain_steps = max_chain_steps
        self.model_name = model_name
        self.model = model
        self.dataset_name = dataset_name
        self.dataset = test_dataset
        self.sample_num = sample_num
        self.temperature = temperature
        self.max_workers = max_workers
        self.save_path = f"{config.OUTPUT_DIR}/uf/{dataset_name}/{model_name}.json"
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        self.show = True
        self.system_prompt = """Read the following question and reason step-by-step to formulate your final answer. {description}

Output ONLY a single JSON object with fields in this exact order:
{{
  "reasoning": "Your step-by-step analysis",
  "final_answer": "Your final answer"
}}"""

    def format_options(self, options: List[Dict[str, str]]) -> str:
        lines = []
        for opt in options:
            lab = str(opt["label"]).strip()
            txt = str(opt["text"]).strip()
            lines.append(f"{lab}. {txt}")
        return "\n".join(lines)

    def build_system_prompt(self, item):
        return self.system_prompt.format(description=item["description"]) 

    def build_user_prompt(self, item: Dict[str, Any], options_override: Optional[List[Dict[str, str]]] = None) -> str:
        if options_override is None:
            opt_block = self.format_options(item["options"])
        else:
            opt_block = self.format_options(options_override)
        q = item["clean_question"]
        return f"Question: {q}\nOptions:\n{opt_block}"

    def entropy_uncertainty(self, p: np.ndarray, options_num) -> float:
        M = int(len(p))
        if M <= 1:
            return 0.0
        eps = 1e-12
        p_safe = np.clip(p, eps, 1.0)
        H = float(-(p_safe * np.log(p_safe)).sum())
        return float(H / np.log(options_num))

    def fidelity_weights_for_chain(self, chain: List[str]) -> Dict[str, float]:
        if not chain:
            return {}

        weights = {}
        denom = 0.0
        for i in range(1, len(chain) + 1):
            denom += (self.tau ** i)

        for idx_from_right, lab in enumerate(reversed(chain), start=1):
            weights[lab] = (self.tau ** idx_from_right) / denom
        return weights

    def elicit_fidelity_chain(
        self,
        item: Dict[str, Any],
        target_label: str,
        labels_all 
    ) -> List[str]:
        original_opts = item["options"]
        
        def make_opts(excluded: set) -> List[Dict[str, str]]:
            out = []
            for o in original_opts:
                lab = str(o["label"]).strip().upper()
                if lab in excluded:
                    continue
                txt = str(o["text"]).strip()
                if lab == target_label:
                    txt = "All other options are wrong."
                out.append({"label": lab, "text": txt})
            return out

        excluded = set()
        chain = []

        if target_label not in labels_all:
            return chain

        outs = []
        for _step in range(self.max_chain_steps):
            opts = make_opts(excluded)
            valid_labels = [o["label"] for o in opts]
            if target_label not in valid_labels:
                break
            if len(valid_labels) <= 0:
                break

            sys = self.build_system_prompt(item)
            user = self.build_user_prompt(item, options_override=opts)
            messages = [{"role": "system", "content": sys},
                         {"role": "user", "content": user}]

            out = self.model.generate_one(messages, temperature=0.1)
            outs.append(out)
            pick = extract_final_answer(out)
            pick = norm_mc(pick, item)
            if pick in labels_all:
                chain.append(pick)
            else:
                break

            if pick == target_label:
                break

            excluded.add(pick)
            if len(excluded) >= (len(labels_all) - 1):
                continue

        return chain, outs

    def _process_one_item(self, idx: int, item: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        labels = [e["label"] for e in item["options"]]

        samples = item.get("sample_results", [])
        picks = [s.get("answer") for s in samples]
        picks = [p for p in picks if isinstance(p, str) and p.upper() in labels]
        k = len(picks)
        valid_num = min(self.sample_num // 2, 1)

        patch: Dict[str, Any] = {}

        if k <= valid_num:
            patch["sampled_counts"] = {lab: 0 for lab in labels}
            patch["sampled_probs"] = {lab: 0.0 for lab in labels}
            patch["uncertainty"] = None
            patch["fidelity_scores"] = {lab: None for lab in labels}
            patch["confidence_scores"] = {lab: None for lab in labels}
            patch["pred_answer"] = None
            patch["uf_score"] = None
            patch["fidelity_chains"] = {}
            patch["fidelity_texts"] = {}
            return idx, patch

        counts = {lab: 0 for lab in labels}
        for p in picks:
            counts[p.upper()] += 1

        p_vec = np.array([counts[lab] / k for lab in labels], dtype=np.float64)
        patch["sampled_counts"] = counts
        patch["sampled_probs"] = {lab: float(counts[lab] / k) for lab in labels}
        patch["pred_answer"] = max(labels, key=lambda lab: counts[lab])

        patch["uncertainty"] = float(self.entropy_uncertainty(p_vec, int(item["options_num"])))

        distinct_sampled = [lab for lab in labels if counts[lab] > 0]
        fidelity_chains: Dict[str, List[str]] = {}
        fidelity_outs: Dict[str, Any] = {}

        for a in distinct_sampled:
            chain, outs = self.elicit_fidelity_chain(item, a, labels)
            fidelity_chains[a] = chain
            fidelity_outs[a] = outs

        patch["fidelity_chains"] = fidelity_chains
        patch["fidelity_texts"] = fidelity_outs

        fidelity_scores = {lab: 0.0 for lab in labels}
        for a in distinct_sampled:
            p_chain = float(counts[a] / k)
            chain = fidelity_chains.get(a, [])
            weights = self.fidelity_weights_for_chain(chain)
            for lab in labels:
                fidelity_scores[lab] += p_chain * float(weights.get(lab, 0.0))

        patch["fidelity_scores"] = {lab: float(fidelity_scores[lab]) for lab in labels}

        return idx, patch

    def generate(self, **kwargs: Any):
        sample_test_path, _ = build_sample_path(self.save_path)

        if not os.path.exists(sample_test_path):
            sample_generate(
                model=self.model,
                dataset=self.test_dataset,
                save_path=sample_test_path,
                build_system=self.build_system_prompt,
                build_user=self.build_user_prompt,
                sample_num=self.sample_num,
                temperature=self.temperature,
                **kwargs
            )

        with open(sample_test_path, "r", encoding="utf-8") as f:
            sample_dataset = json.load(f)

        for item in tqdm(sample_dataset):
            samples = item.get("sample_results", [])
            for s in samples:
                if "answer" not in s.keys():
                    text = s.get("text", "")
                    ans = extract_final_answer(text)
                    s["answer"] = norm_mc(ans, item)

        with open(sample_test_path, "w", encoding="utf-8") as f:
            json.dump(sample_dataset, f, ensure_ascii=False, indent=4)
            
        for item, sample_item in zip(self.dataset, sample_dataset):
            item["sample_results"] = sample_item["sample_results"][: self.sample_num]

        if "qwen3-4b" in self.model_name:
            self.max_workers = 1

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = [
                ex.submit(self._process_one_item, i, self.dataset[i])
                for i in range(len(self.dataset))
            ]
            for fu in tqdm(as_completed(futures), total=len(futures)):
                idx, patch = fu.result()
                self.dataset[idx].update(patch)

        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(self.dataset, f, ensure_ascii=False, indent=4)

    def extract(self):
        return

    def calculate(self):
        
        with open(self.save_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        for item in tqdm(dataset):
            labels = [e["label"] for e in item["options"]]
            unc = item["uncertainty"]
            if unc is None:
                continue
            F = item["fidelity_scores"]
            conf_scores = {}
            for lab in labels:
                c = (1.0 - float(unc)) * float(F[lab])
                conf_scores[lab] = float(c)

            item["confidence_scores"] = conf_scores
            item["pred_score"] = float(conf_scores.get(item["pred_answer"], 0.0))

        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)

        return
