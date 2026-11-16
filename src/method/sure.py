from __future__ import annotations

import ast
import copy
import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src import config
from src.utils import (
    build_sample_path,
    extract_final_answer,
    load_nli_model,
    sample_generate,
)


class SURE:
    """SURE using anchors and perturbed-question probe answers."""

    def __init__(
        self,
        model_name: str,
        model: Any,
        dataset_name: str,
        dev_dataset: List[Dict[str, Any]],
        test_dataset: List[Dict[str, Any]],
        sample_num: int,
        temperature: float,
        anchor_count: int = 3,
        anchor_temperature: float = 0.2,
        candidate_answers_per_paraphrase: int = 2,
    ) -> None:
        del dev_dataset
        self.model_name = model_name
        self.model = model
        self.dataset_name = dataset_name
        self.dataset = test_dataset
        self.sample_num = int(sample_num)
        self.temperature = float(temperature)
        self.anchor_count = int(anchor_count)
        self.anchor_temperature = float(anchor_temperature)
        self.candidate_answers_per_paraphrase = int(
            candidate_answers_per_paraphrase
        )
        self.save_path = (
            f"{config.OUTPUT_DIR}/sure/{dataset_name}/{model_name}.json"
        )
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        self.system_prompt = """Read the following question and reason step-by-step to formulate your final answer. {description}

Output ONLY a single JSON object with fields in this exact order:
{{
  "reasoning": "Your step-by-step analysis",
  "final_answer": "Your final answer"
}}"""

    def build_system_prompt(self, item: Dict[str, Any]) -> str:
        return self.system_prompt.format(description=item.get("description", ""))

    def build_user_prompt(
        self,
        item: Dict[str, Any],
        question: Optional[str] = None,
    ) -> str:
        selected_question = question or item.get("question", "")
        if self.dataset_name == "coqa" and "story" in item:
            return f"Context: {item['story']}\nQuestion: {selected_question}"
        return f"Question: {selected_question}"

    def build_paraphrase_messages(
        self, item: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        user = (
            f"Provide {self.sample_num} paraphrases for this question: "
            f"{item['question']}. Do NOT answer the question. Return ONLY a "
            "valid JSON array of strings."
        )
        if self.dataset_name == "coqa" and "story" in item:
            user = f"Context: {item['story']}\n{user}"
        return [
            {
                "role": "system",
                "content": "You are a helpful assistant that paraphrases questions.",
            },
            {"role": "user", "content": user},
        ]

    def _clean_paraphrases(
        self, values: Sequence[Any], item: Dict[str, Any]
    ) -> List[str]:
        original = str(item.get("question", "")).strip()
        output: List[str] = []
        seen = set()
        for value in values:
            if not isinstance(value, str):
                continue
            text = re.sub(
                r"^(sentence|paraphrase)\s*:\s*",
                "",
                value.strip(),
                flags=re.IGNORECASE,
            ).strip()
            key = text.casefold()
            if (
                not text
                or len(text) < 3
                or text == original
                or key in seen
            ):
                continue
            seen.add(key)
            output.append(text)
            if len(output) >= self.sample_num:
                break
        return output

    def parse_paraphrase_output(
        self, text: Any, item: Dict[str, Any]
    ) -> List[str]:
        if not isinstance(text, str) or not text.strip():
            return []
        raw = text.strip()

        for parser in (json.loads, ast.literal_eval):
            try:
                value = parser(raw)
            except Exception:
                continue
            if isinstance(value, list):
                return self._clean_paraphrases(value, item)

        for candidate in re.findall(r"\[[\s\S]*?\]", raw):
            for parser in (json.loads, ast.literal_eval):
                try:
                    value = parser(candidate)
                except Exception:
                    continue
                if isinstance(value, list):
                    parsed = self._clean_paraphrases(value, item)
                    if parsed:
                        return parsed

        lines = []
        for line in raw.splitlines():
            line = re.sub(r"^\s*\d+[\.\)\-:]\s*", "", line.strip())
            line = re.sub(r"^\s*[\-\*\u2022]\s*", "", line)
            line = line.strip().strip('"').strip("'").strip()
            if line:
                lines.append(line)
        return self._clean_paraphrases(lines, item)

    @staticmethod
    def _load_json(path: str) -> List[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def _save_json(path: str, data: List[Dict[str, Any]]) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

    def _prepare_sample_cache(self, **kwargs: Any) -> List[Dict[str, Any]]:
        sample_test_path, _ = build_sample_path(self.save_path)
        if not os.path.exists(sample_test_path):
            sample_generate(
                model=self.model,
                dataset=self.dataset,
                save_path=sample_test_path,
                build_system=self.build_system_prompt,
                build_user=self.build_user_prompt,
                sample_num=self.sample_num,
                temperature=self.temperature,
                **kwargs,
            )

        records = self._load_json(sample_test_path)
        if not records:
            return records

        missing_paraphrases = [
            (index, item)
            for index, item in enumerate(records)
            if not item.get("perturbations")
        ]
        if missing_paraphrases:
            outputs = self.model.generate_batch(
                [
                    self.build_paraphrase_messages(item)
                    for _, item in missing_paraphrases
                ],
                temperature=0.1,
                **kwargs,
            )
            for (item_index, item), output in zip(missing_paraphrases, outputs):
                raw = output if isinstance(output, str) else ""
                records[item_index]["para_outputs"] = raw
                records[item_index]["perturbations"] = self.parse_paraphrase_output(
                    raw, item
                )

        messages: List[List[Dict[str, str]]] = []
        targets: List[Tuple[int, str]] = []
        for item_index, item in enumerate(records):
            perturbed_results = item.get("perturbed_results")
            if not isinstance(perturbed_results, list):
                perturbed_results = []
                item["perturbed_results"] = perturbed_results

            for paraphrase in (item.get("perturbations") or [])[: self.sample_num]:
                if not isinstance(paraphrase, str) or not paraphrase.strip():
                    continue
                question = paraphrase.strip()
                existing_num = sum(
                    1
                    for result in perturbed_results
                    if result.get("perturbed_question") == question
                    and str(result.get("text") or "").strip()
                )
                message = [
                    {
                        "role": "system",
                        "content": self.build_system_prompt(item),
                    },
                    {
                        "role": "user",
                        "content": self.build_user_prompt(item, question),
                    },
                ]
                for _ in range(
                    self.candidate_answers_per_paraphrase - existing_num
                ):
                    messages.append(copy.deepcopy(message))
                    targets.append((item_index, question))

        if messages:
            outputs = self.model.generate_batch(
                messages,
                temperature=self.temperature,
                **kwargs,
            )
            for (item_index, question), output in zip(targets, outputs):
                records[item_index]["perturbed_results"].append(
                    {
                        "perturbed_question": question,
                        "text": output if isinstance(output, str) else "",
                    }
                )
        self._save_json(sample_test_path, records)
        return records

    def _prepare_anchor_records(self, **kwargs: Any) -> List[Dict[str, Any]]:
        if os.path.exists(self.save_path):
            records = self._load_json(self.save_path)
        else:
            records = copy.deepcopy(self.dataset)

        messages: List[List[Dict[str, str]]] = []
        targets: List[Tuple[int, int]] = []
        for item_index, item in enumerate(records):
            anchors = item.get("anchor_results")
            if not isinstance(anchors, list):
                anchors = []
            anchors = anchors[: self.anchor_count]
            while len(anchors) < self.anchor_count:
                anchors.append({})
            item["anchor_results"] = anchors

            message = [
                {"role": "system", "content": self.build_system_prompt(item)},
                {"role": "user", "content": self.build_user_prompt(item)},
            ]
            for anchor_index, anchor in enumerate(anchors):
                if isinstance(anchor, dict) and str(anchor.get("text") or "").strip():
                    continue
                messages.append(copy.deepcopy(message))
                targets.append((item_index, anchor_index))

        if messages:
            outputs = self.model.generate_batch(
                messages,
                temperature=self.anchor_temperature,
                **kwargs,
            )
            for (item_index, anchor_index), output in zip(targets, outputs):
                records[item_index]["anchor_results"][anchor_index] = {
                    "text": output if isinstance(output, str) else ""
                }
        self._save_json(self.save_path, records)
        return records

    def generate(self, **kwargs: Any) -> None:
        self._prepare_sample_cache(**kwargs)
        self._prepare_anchor_records(**kwargs)

    def extract(self) -> None:
        sample_test_path, _ = build_sample_path(self.save_path)
        sample_records = self._load_json(sample_test_path)
        for item in tqdm(sample_records):
            greedy = item.get("greedy_results")
            if isinstance(greedy, dict) and "answer" not in greedy:
                greedy["answer"] = extract_final_answer(greedy.get("text", ""))
            for result in item.get("perturbed_results") or []:
                if isinstance(result, dict) and "answer" not in result:
                    result["answer"] = extract_final_answer(result.get("text", ""))
        self._save_json(sample_test_path, sample_records)

        records = self._load_json(self.save_path)
        for item in tqdm(records):
            for anchor in item.get("anchor_results") or []:
                if isinstance(anchor, dict) and "answer" not in anchor:
                    anchor["answer"] = extract_final_answer(anchor.get("text", ""))
        self._save_json(self.save_path, records)

    @staticmethod
    def _valid_answer(record: Any) -> Optional[str]:
        if not isinstance(record, dict):
            return None
        answer = record.get("answer")
        if answer is None:
            return None
        answer = str(answer).strip()
        return answer or None

    def _select_anchors(self, item: Dict[str, Any]) -> List[str]:
        anchors = []
        for record in (item.get("anchor_results") or [])[: self.anchor_count]:
            answer = self._valid_answer(record)
            if answer is not None:
                anchors.append(answer)
        return anchors

    def _select_probes(
        self, item: Dict[str, Any]
    ) -> Tuple[List[str], List[str], List[int]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for record in item.get("perturbed_results") or []:
            if not isinstance(record, dict):
                continue
            question = str(record.get("perturbed_question") or "").strip()
            if question:
                grouped.setdefault(question, []).append(record)

        questions: List[str] = []
        probes: List[str] = []
        group_indices: List[int] = []
        for group_index, question in enumerate(
            (item.get("perturbations") or [])[: self.sample_num]
        ):
            if not isinstance(question, str) or not question.strip():
                continue
            question = question.strip()
            records_for_question = [
                record
                for record in grouped.get(question, [])
                if self._valid_answer(record) is not None
            ]
            for record in records_for_question[: self.candidate_answers_per_paraphrase]:
                answer = self._valid_answer(record)
                questions.append(question)
                probes.append(answer)
                group_indices.append(group_index)
        return questions, probes, group_indices

    @staticmethod
    def _entailment_label_id(nli_model: Any) -> int:
        for index, label in nli_model.config.id2label.items():
            if "entail" in str(label).lower():
                return int(index)

    def _entailment_probabilities(
        self,
        premises: Sequence[str],
        hypotheses: Sequence[str],
        tokenizer: Any,
        nli_model: Any,
        batch_size_nli: int = 8,
    ) -> np.ndarray:
        if not premises:
            return np.zeros((0,), dtype=np.float32)

        entailment_id = self._entailment_label_id(nli_model)
        device = next(nli_model.parameters()).device
        scores = np.empty(len(premises), dtype=np.float32)
        bs = max(1, int(batch_size_nli))
        for start in range(0, len(premises), bs):
            end = min(len(premises), start + bs)
            encoded = tokenizer(
                list(premises[start:end]),
                list(hypotheses[start:end]),
                truncation=True,
                max_length=512,
                padding=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                logits = nli_model(**encoded).logits
                probabilities = F.softmax(logits, dim=-1)
            scores[start:end] = (
                probabilities[:, entailment_id]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
        return scores

    def _bidirectional_entailment_matrix(
        self,
        rows: Sequence[str],
        columns: Sequence[str],
        tokenizer: Any,
        nli_model: Any,
    ) -> np.ndarray:
        row_count = len(rows)
        column_count = len(columns)
        if row_count == 0 or column_count == 0:
            return np.zeros((row_count, column_count), dtype=np.float32)

        premises = [row for row in rows for _ in columns]
        hypotheses = [column for _ in rows for column in columns]
        forward = self._entailment_probabilities(
            premises, hypotheses, tokenizer, nli_model
        )
        backward = self._entailment_probabilities(
            hypotheses, premises, tokenizer, nli_model
        )
        shape = (row_count, column_count)
        return np.minimum(
            forward.reshape(shape), backward.reshape(shape)
        ).astype(np.float32)

    @staticmethod
    def _spectral_scores(
        matrix: np.ndarray,
    ) -> Tuple[float, float, List[float], float]:
        singular_values = np.linalg.svd(
            matrix.astype(np.float64), compute_uv=False
        )
        squared_energy = float(np.sum(np.square(singular_values)))
        confidence = float(
            np.clip(
                squared_energy / float(matrix.shape[0] * matrix.shape[1]),
                0.0,
                1.0,
            )
        )
        return confidence, 1.0 - confidence, singular_values.tolist(), squared_energy

    @staticmethod
    def _question_answer_text(question: str, answer: str) -> str:
        return f"{question.strip()} {answer.strip()}".strip()

    def calculate(self) -> List[Dict[str, Any]]:
        self.extract()
        sample_test_path, _ = build_sample_path(self.save_path)
        sample_records = self._load_json(sample_test_path)
        anchor_records = self._load_json(self.save_path)

        output: List[Dict[str, Any]] = []
        valid: List[Tuple[Dict[str, Any], List[str], List[str]]] = []
        valid_num = (
            self.sample_num * self.candidate_answers_per_paraphrase
        ) // 2
        for anchor_item, sample_item in zip(anchor_records, sample_records):
            anchors = self._select_anchors(anchor_item)
            probe_questions, probes, group_indices = self._select_probes(sample_item)
            greedy = sample_item.get("greedy_results")
            greedy_answer = self._valid_answer(greedy)

            record: Dict[str, Any] = {
                "id": sample_item.get("id"),
                "question": sample_item.get("question", ""),
                "description": sample_item.get("description", ""),
                "ground_truth": sample_item.get("ground_truth"),
                "pred_answer": greedy_answer,
                "pred_score": None,
                "anchor_results": anchor_item.get("anchor_results", []),
                "anchor_answers": anchors,
                "probe_questions": probe_questions,
                "probe_answers": probes,
                "probe_group_indices": group_indices,
                "similarity_matrix": None,
                "singular_values": None,
                "squared_spectral_energy": None,
                "sure_confidence": None,
                "sure_uncertainty": None,
                "similarity_matrix_answer": None,
                "singular_values_answer": None,
                "squared_spectral_energy_answer": None,
                "sure_confidence_answer": None,
                "sure_uncertainty_answer": None,
            }
            if "story" in sample_item:
                record["story"] = sample_item["story"]
            if "options" in sample_item:
                record["options"] = sample_item["options"]
            output.append(record)

            if (
                greedy_answer is not None
                and len(anchors) == self.anchor_count
                and len(probes) > valid_num
            ):
                valid.append((record, anchors, probes))

        if valid:
            tokenizer, nli_model = load_nli_model(config.NLI_MODEL_PATH)
            for record, anchors, probes in tqdm(valid):
                original_question = str(record.get("question", ""))
                anchor_texts = [
                    self._question_answer_text(original_question, answer)
                    for answer in anchors
                ]
                probe_texts = [
                    self._question_answer_text(question, answer)
                    for question, answer in zip(record["probe_questions"], probes)
                ]
                matrix = self._bidirectional_entailment_matrix(
                    anchor_texts, probe_texts, tokenizer, nli_model
                )
                confidence, uncertainty, singular_values, squared_energy = (
                    self._spectral_scores(matrix)
                )
                record["similarity_matrix"] = matrix.tolist()
                record["singular_values"] = singular_values
                record["squared_spectral_energy"] = squared_energy
                record["sure_confidence"] = confidence
                record["sure_uncertainty"] = uncertainty
                record["pred_score"] = confidence

                answer_matrix = self._bidirectional_entailment_matrix(
                    anchors, probes, tokenizer, nli_model
                )
                (
                    answer_confidence,
                    answer_uncertainty,
                    answer_singular_values,
                    answer_squared_energy,
                ) = self._spectral_scores(answer_matrix)
                record["similarity_matrix_answer"] = answer_matrix.tolist()
                record["singular_values_answer"] = answer_singular_values
                record["squared_spectral_energy_answer"] = answer_squared_energy
                record["sure_confidence_answer"] = answer_confidence
                record["sure_uncertainty_answer"] = answer_uncertainty

        self._save_json(self.save_path, output)
        return output
