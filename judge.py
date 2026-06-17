import argparse
import json
import os
import time
import re
import string
from typing import Dict, Any, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
api_base = os.getenv("OPENROUTER_API_BASE")

def normalize_for_match(s: str) -> str:
    if s is None:
        return ""
    s = str(s).lower()
    s = s.replace("_", " ")

    exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`"]))
    s = "".join(ch if ch not in exclude else " " for ch in s)
    return s

def match_string(resp_text: str, ground: List[str]) -> bool:
    r = normalize_for_match(resp_text)
    if not r:
        return False
    for g in ground:
        if normalize_for_match(g) == r:
            return True
    return False

def build_judge_prompt(rec: Dict[str, Any], ground: List[str], resp: str) -> str:
    q = rec.get("question", "")
    ref_text = "\n".join(f"{i+1}. {g}" for i, g in enumerate(ground))
    prompt = (
        f"Question:\n{q}\n\n"
        f"Ground Truth Answer(s):\n{ref_text}\n\n"
        f"Based strictly on the ground truth provided above, is the following answer correct?\n"
        f"Do not use any external knowledge.\n"
        f"Respond only with 'Yes' or 'No'.\n\n"
        f"Answer: {resp}"
    )
    if "story" in rec.keys():
        prompt = "Content:\n" + rec["story"] + "\n\n" + prompt
    return prompt

def judge_one(
    client: OpenAI,
    model: str,
    reasoning_effort: str,
    prompt: str,
    max_retries: int = 5,
    base_backoff: float = 1.0,
):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                reasoning_effort=reasoning_effort,
            )
            response = response.choices[0].message.content.strip()
            if response:
                return response
            return None
        except Exception as e:
            print(e)
            time.sleep(base_backoff * (2 ** attempt))
    return None

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate answers using threaded calls to /v1/responses with local string-match shortcut. "
                    "TruthfulQA-MC uses local-only matching (no API)."
    )
    parser.add_argument("--input_dir", type=str, default="output")
    parser.add_argument("--models", nargs="+", required=True, help="Model names to run")
    parser.add_argument("--datasets", nargs="+", required=True, help="Dataset names to run")
    parser.add_argument("--methods", nargs="+", required=True, help="Methods to run")
    parser.add_argument("--eval_model", type=str, default="gpt-5.1")
    parser.add_argument("--reasoning_effort", type=str, default="low", choices=["low", "medium", "high"])
    parser.add_argument("--max_workers", type=int, default=32)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--base_backoff", type=float, default=1.0)
    args = parser.parse_args()

    for method_name in args.methods:
        for dataset_name in args.datasets:
            for model_name in args.models:
                path = f"{args.input_dir}/{method_name}/{dataset_name}/{model_name}.json"
                with open(path, "r", encoding="utf-8") as f:
                    records = json.load(f)
                if dataset_name != "truthfulqa_mc":
                    client = OpenAI(
                        api_key=api_key,
                        base_url=api_base,
                        timeout=args.timeout,
                    )

                tasks: List[Tuple[str, str, str]] = []
                api_results: Dict[str, float] = {}

                for rec in tqdm(records):
                    pred_answer = rec.get("pred_answer", None)
                    if pred_answer is None or pred_answer == "":
                        continue

                    sid = rec.get("id", "")
                    ground_truth = rec.get("ground_truth", [])
                    if isinstance(ground_truth, str):
                        ground_truth = [ground_truth]

                    if dataset_name == "truthfulqa_mc":
                        rec["label"] = 1.0 if match_string(pred_answer, ground_truth) else 0.0
                        continue

                    if match_string(pred_answer, ground_truth):
                        rec["label"] = 1.0
                        continue

                    prompt = build_judge_prompt(rec, ground_truth, pred_answer)
                    tasks.append((sid, pred_answer, prompt))

                if tasks:
                    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
                        fut2key = {}
                        for sid, pred_answer, prompt in tasks:
                            fut = ex.submit(
                                judge_one,
                                client,
                                args.eval_model,
                                args.reasoning_effort,
                                prompt,
                                args.max_retries,
                                args.base_backoff,
                            )
                            fut2key[fut] = sid

                        for fut in tqdm(as_completed(fut2key), total=len(fut2key)):
                            sid = fut2key[fut]
                            try:
                                txt = fut.result()
                            except Exception:
                                txt = None

                            if not txt:
                                continue

                            reply = txt.strip().lower()
                            score = 1.0 if reply.startswith("yes") else 0.0
                            api_results[sid] = score
                
                for rec in tqdm(records):
                    sid = rec.get("id", "")
                    rec["label"] = api_results.get(sid, rec.get("label", None))
                
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(records, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()