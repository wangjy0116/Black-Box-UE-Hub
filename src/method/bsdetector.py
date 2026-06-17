from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import os
import re
from src.utils import compute_greedy_semantic, sample_generate, build_sample_path, extract_final_answer, load_nli_model
import numpy as np
import scipy.linalg
import copy
import json
import torch
from tqdm import tqdm
from src import config


class BSDetector:

    def __init__(self, model_name, model, dataset_name, dev_dataset, test_dataset, sample_num, temperature):

        self.sample_system_prompt = """Read the following question, reason step-by-step to formulate your final answer. {description} Output your reasoning and a JSON object that states the final answer in it: {{"final_answer": "Your final answer"}}"""
        self.system_prompt = """Read the following question and reason step-by-step to formulate your final answer. {description}

Output ONLY a single JSON object with fields in this exact order:
{{
  "reasoning": "Your step-by-step analysis",
  "final_answer": "Your final answer"
}}"""
        self.reflect_prompt1 = """Question: {question} Proposed Answer: {greedy_answer}. Is the proposed answer: (A) Correct (B) Incorrect (C) I am not sure.\nThe output should strictly use the following template:\nExplanation: [Your step-by-step analysis]\nAnswer: [Please select one of the following options and only provide that choice: (A) Correct, (B) Incorrect, or (C) I am not sure.]""" 
        self.reflect_prompt2 = """Question: {question} Proposed Answer: {greedy_answer}. Are you really sure the proposed answer iscorrect? Choose again: (A) Correct (B) Incorrect (C) I amnot sure.\nThe output should strictly use the following template\nExplanation: [Your step-by-step analysis]\nAnswer: [Please select one of the following options and only provide that choice: (A) Correct, (B) Incorrect, or (C) I am not sure.]"""
        self.model_name = model_name
        self.model = model
        self.dataset_name = dataset_name
        self.dataset = test_dataset
        self.sample_num = sample_num
        self.temperature = temperature
        self.save_path = f"{config.OUTPUT_DIR}/bsdetector/{dataset_name}/{model_name}.json"
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        self.show = True

    def extract_choice_abc(self, text: str) -> Optional[str]:
        ANSWER_RE = re.compile(
            r"answer\s*:\s*(?P<ans>.+?)(?=\n|$)",
            flags=re.IGNORECASE,
        )
        if not isinstance(text, str):
            return None

        matches = list(ANSWER_RE.finditer(text))
        if not matches:
            return None

        raw_ans = matches[-1].group("ans").strip()
        raw_ans = raw_ans.strip().strip('"').strip("'").strip()
        m = re.search(r"\b([ABC])\b", raw_ans, flags=re.IGNORECASE)
        if m:
            return m.group(1).lower()
        low = raw_ans.lower()
        if "correct" in low:
            return "a"  
        if "incorrect" in low:
            return "b"
        if "not sure" in low or "i am not sure" in low:
            return "c"

        return None

    def build_sample_system_prompt(self, item):
        return self.sample_system_prompt.format(description=item["description"]) 

    def build_sample_user_prompt(self, item: Dict[str, Any]) -> str:
        return f"Question: {item['question']}"

    def build_coqa_sample_user_prompt(self, item):
        return f"Context: {item['story']}\nQuestion: {item['question']}"

    def build_reflect_user_prompt(self, prompt, item: Dict[str, Any]) -> str:
        return prompt.format(question=item["question"], greedy_answer = item["greedy_results"]["answer"] if item["greedy_results"]["answer"] is not None else "")

    def build_coqa_reflect_user_prompt(self, prompt, item: Dict[str, Any]) -> str:
        return f"Context: {item['story']}\n" + prompt.format(question=item["question"], greedy_answer = item["greedy_results"]["answer"] if item["greedy_results"]["answer"] is not None else "")

    def bsdetector_score(self, sample, contra, contra_re, greedy_results):
        score = {"a": 1.0, "b": 0.0, "c": 0.5}
        cnt = 0
        for i in range(self.sample_num):
            if sample[i]["answer"] is not None:
                cnt+=1
        
        if greedy_results["reflect1_answer"] and  greedy_results["reflect2_answer"]:
            reflect_score = (score[greedy_results["reflect1_answer"]] + score[greedy_results["reflect2_answer"]])/2
        elif greedy_results["reflect1_answer"]:
            reflect_score = score[greedy_results["reflect1_answer"]] 
        elif greedy_results["reflect2_answer"]:
            reflect_score = score[greedy_results["reflect2_answer"]]
        else: reflect_score = None
        if cnt == 0: 
            return None
        
        sample_score = float(np.mean(1.0 - 0.5 * (contra[:cnt] + contra_re[:cnt])))
        if reflect_score is None:
            return sample_score
        else:
            return sample_score*0.7 + reflect_score*0.3

    def generate(self, **kwargs: Any) -> List[Dict[str, Any]]:

        sample_test_path, _ =  build_sample_path(self.save_path)   
        if self.dataset_name == "coqa":
            build_user=self.build_coqa_sample_user_prompt
        else:
            build_user=self.build_sample_user_prompt

        if not os.path.exists(sample_test_path):
            sample_generate(
                model=self.model,
                dataset=self.dataset,
                save_path=sample_test_path,
                build_system=self.build_sample_system_prompt,
                build_user=build_user,
                sample_num=self.sample_num,
                temperature=self.temperature,
                **kwargs
            )
        
        with open(sample_test_path, "r", encoding="utf-8") as f:
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

        output = []
        messages = []
        for item in dataset:
            if self.dataset_name != "coqa":
                user = self.build_reflect_user_prompt(self.reflect_prompt1, item)
            else:
                user = self.build_coqa_reflect_user_prompt(self.reflect_prompt1, item)
            messages.append([{"role": "user", "content": user}])
            output.append({
                "id": item["id"],
                "question": item["question"],
                "description": item["description"],
                "ground_truth": item["ground_truth"],
                "greedy_results": item["greedy_results"]
            })

        if self.show:
            print(messages[0])

        texts1 = self.model.generate_batch(messages, temperature=0.1, **kwargs)
        idx = 0
        for item, text, message in zip(dataset, texts1, messages):
            if self.dataset_name != "coqa":
                user = self.build_reflect_user_prompt(self.reflect_prompt2, item)
            else:
                user = self.build_coqa_reflect_user_prompt(self.reflect_prompt2, item)
            message.append({"role": "assistant", "content": text})
            message.append({"role": "user", "content": user})

        if self.show:
            print(messages[0])        

        texts2 = self.model.generate_batch(messages, temperature=0.1, **kwargs)
        for idx, item in enumerate(output):
            item["greedy_results"]["reflect1_text"] = texts1[idx]
            item["greedy_results"]["reflect2_text"] = texts2[idx]

        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=4)


    def extract(self):

        with open(self.save_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        for item in tqdm(dataset):
            greedy = item["greedy_results"]
            greedy["reflect1_answer"] = self.extract_choice_abc(greedy["reflect1_text"])
            greedy["reflect2_answer"] = self.extract_choice_abc(greedy["reflect2_text"])

        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)

    def calculate(self):

        sample_test_path, _ =  build_sample_path(self.save_path)
        with open(sample_test_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        
        valid_num = min(self.sample_num // 2, 1)
        tokenizer, nli_model = load_nli_model(config.NLI_MODEL_PATH) 

        if "greedy_semantic_entail" not in dataset[0].keys():
            for item in tqdm(dataset):
                question = item["question"]
                sample_texts = []
                for idx, s in enumerate(item["sample_results"][:self.sample_num]):
                    ans = s.get("answer", None)
                    if ans is None:
                        continue
                    if not isinstance(ans, str):
                        ans = str(ans)

                    ans_strip = ans.strip()
                    if not ans_strip:
                        continue

                    sample_texts.append(ans_strip)
                
                E, C, E_logits, C_logits, E_re, C_re, E_logits_re, C_logits_re, _ = compute_greedy_semantic(
                    question=question,
                    greedy_text=item["greedy_results"]["answer"] if item["greedy_results"]["answer"] is not None else "",
                    sample_texts=sample_texts,
                    tokenizer=tokenizer,
                    nli_model=nli_model
                )

                item["greedy_semantic_entail"] = E.tolist()
                item["greedy_semantic_contra"] = C.tolist()
                item["greedy_semantic_entail_re"] = E_re.tolist()
                item["greedy_semantic_contra_re"] = C_re.tolist()

            with open(sample_test_path, "w", encoding="utf-8") as f:
                json.dump(dataset, f, ensure_ascii=False, indent=4)
        
        with open(self.save_path, "r", encoding="utf-8") as f:
            bs_dataset = json.load(f)

        bsdetector_scores = []
        for item1, item2 in zip(dataset, bs_dataset):
            C = np.asarray(item1["greedy_semantic_contra"], dtype=np.float32)
            C_re = np.asarray(item1["greedy_semantic_contra_re"], dtype=np.float32)
            if C.size <=valid_num or C_re.size <=valid_num:
                bsdetector_scores.append(None)
            else:
                bsdetector_scores.append(self.bsdetector_score(item1["sample_results"], C, C_re, item2["greedy_results"]))

        for item, bsdetector_score in zip(bs_dataset, bsdetector_scores): 
            item["pred_answer"] = item["greedy_results"]["answer"]
            if item["pred_answer"] is None:
                item["pred_score"] = None
            else:
                item["pred_score"] = bsdetector_score
            
        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(bs_dataset, f, ensure_ascii=False, indent=4)

