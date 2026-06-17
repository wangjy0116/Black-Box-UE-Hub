import ast
import json
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple
from tqdm import tqdm
import numpy as np
import spacy
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from vllm import LLM, SamplingParams
import src.config as config

def extract_final_answer(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    raw = text.strip()
    if not raw:
        return None

    def _clean(ans: Any) -> Optional[str]:
        if ans is None:
            return None
        s = str(ans).strip()
        if not s:
            return None
        if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
            s = s[1:-1].strip()
        return s if s else None

    def _extract_from_obj(obj: Any) -> Optional[str]:
        if not isinstance(obj, dict):
            return None
        ans = obj.get("final_answer")
        if ans is None:
            ans = obj.get("answer")
        return _clean(ans)

    for parser in (json.loads, ast.literal_eval):
        try:
            obj = parser(raw)
        except Exception:
            continue

        ans = _extract_from_obj(obj)
        if ans is not None:
            return ans

        if isinstance(obj, str) and obj.strip():
            try:
                obj2 = parser(obj.strip())
            except Exception:
                continue
            ans = _extract_from_obj(obj2)
            if ans is not None:
                return ans

    return None

def sample_generate(
    model: Any,
    dataset: List[Dict[str, Any]],
    save_path: str,
    build_system: Callable[[Dict[str, Any]], str],
    build_user: Callable[[Dict[str, Any]], str],
    sample_num: int,
    temperature: float,
    sample_key: str = "sample_results",
    greedy_key: str = "greedy_results",
    show: bool = True,
    **kwargs: Any,
) -> List[Dict[str, Any]]:

    out = [dict(x) for x in dataset]

    sample_pairs = []
    greedy_pairs = []

    for item in out:
        system = build_system(item)
        user = build_user(item)
        greedy_pairs.append([{"role":"system", "content": system}, {"role":"user", "content":user}])
        for _ in range(int(sample_num)):
            sample_pairs.append([{"role":"system", "content": system}, {"role":"user", "content":user}])
    if show:
        print(sample_pairs[0])

    sample_texts = model.generate_batch(sample_pairs, temperature=temperature, **kwargs)
    greedy_texts = model.generate_batch(greedy_pairs, temperature=0.1, **kwargs)

    idx = 0
    for i, item in enumerate(out):
        samples = []
        for _ in range(int(sample_num)):
            samples.append({"text": sample_texts[idx]})
            idx += 1
        item[sample_key] = samples
        item[greedy_key] = {"text": greedy_texts[i]}

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=4)

def build_sample_path(output_path: str) -> str:
    parts = os.path.normpath(output_path).split(os.sep)
    keep = parts[-2:] if len(parts) >= 2 else parts
    rest = os.path.join(*keep) if keep else os.path.basename(output_path)
    return os.path.join("sample_test", rest), os.path.join("sample_dev", rest)

def norm_mc(s: str, item) -> str:
    if not isinstance(s, str):
        return ""
    labels = [o["label"] for o in item["options"]]
    texts = []
    for o in item.get("options", []):
        t = o.get("text", "")
        t = "" if t is None else str(t)
        t = t.rstrip()  
        if t.endswith("."):
            t = t[:-1]
        t = t.upper()
        texts.append(t)

    s = s.upper()
    if s in labels:
        return s
    if len(s)>1 and s[1]=='.' and s[0] in labels:
        return s[0]
    if s.endswith("."):
        s = s[:-1]
    if s in texts:
        return labels[texts.index(s)]
    return None


def load_nli_model(nli_model_path):
    tokenizer = AutoTokenizer.from_pretrained(nli_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(nli_model_path)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return tokenizer, model

def check_entailment(question: str, ref: str, ans: str, max_length: int = 512) -> bool:
    tokenizer, nli_model = load_nli_model(config.NLI_MODEL_PATH)
    id2label = getattr(nli_model.config, "id2label", None)
    _id2label = {}
    for k, v in id2label.items():
        try:
            ki = int(k)
        except Exception:
            ki = k
        _id2label[ki] = str(v).lower()

    def _find_label_id(substrs):
        for i, lab in _id2label.items():
            for s in substrs:
                if s in lab:
                    return int(i)
        return None

    entail_id = _find_label_id(["entail"])
    contra_id = _find_label_id(["contra", "contradict"])

    if question is not None:
        prem = f"{question} {ref}".strip()
        hypo = f"{question} {ans}".strip()
    else:
        prem = f"{ref}".strip()
        hypo = f"{ans}".strip()

    enc = tokenizer(
        prem,
        hypo,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )

    device = next(nli_model.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}
    
    nli_model.eval()
    with torch.no_grad():
        out = nli_model(**enc)
        logits = out.logits
        pred = int(logits.argmax(dim=-1).item())

    return 1 if pred == entail_id else 0

def compute_sample_semantic(
    question: str,
    sample_texts: List[str],
    tokenizer,
    nli_model,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size_nli: int = 8,
    max_length: int = 512
):
    id2label = getattr(nli_model.config, "id2label", None)
    _id2label = {}
    for k, v in id2label.items():
        try:
            ki = int(k)
        except Exception:
            ki = k
        _id2label[ki] = str(v).lower()

    def _find_label_id(substrs):
        for i, lab in _id2label.items():
            for s in substrs:
                if s in lab:
                    return int(i)
        return None

    entail_id = _find_label_id(["entail"])
    contra_id = _find_label_id(["contra", "contradict"])
    neutral_id = _find_label_id(["neutral"]) 

    n = len(sample_texts)
    if n == 0:
        z = np.zeros((0, 0), dtype=np.float32)
        return z, z, z, z, z, z, z, {}

    if question is None:
        responses = [str(t).strip() for t in sample_texts]
    else:
        responses = [f"{question} {t}".strip() for t in sample_texts]

    premises = []
    hypotheses = []
    pairs_ij = []

    for i in range(n):
        for j in range(n):
            pairs_ij.append((i, j))
            premises.append(responses[i])
            hypotheses.append(responses[j])

    entail_probs = np.zeros((n, n), dtype=np.float32)
    contra_probs = np.zeros((n, n), dtype=np.float32)
    neutral_probs = np.zeros((n, n), dtype=np.float32)
    class_mat = np.zeros((n, n), dtype=int)
    entail_logits = np.zeros((n, n), dtype=np.float32)
    contra_logits = np.zeros((n, n), dtype=np.float32)
    neutral_logits = np.zeros((n, n), dtype=np.float32)

    bs = max(1, int(batch_size_nli))
    for start in range(0, len(pairs_ij), bs):
        end = min(len(pairs_ij), start + bs)

        enc = tokenizer(
            premises[start:end],
            hypotheses[start:end],
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            out = nli_model(**enc)
            logits = out.logits 
            probs = F.softmax(logits, dim=-1)

        # probs
        batch_entail_probs = probs[:, entail_id].detach().cpu().numpy()
        batch_contra_probs = probs[:, contra_id].detach().cpu().numpy()
        batch_neutral_probs = probs[:, neutral_id].detach().cpu().numpy()
        class_preds = probs.argmax(dim=-1).detach().cpu().numpy()

        # logits
        batch_entail_log = logits[:, entail_id].detach().cpu().numpy()
        batch_contra_log = logits[:, contra_id].detach().cpu().numpy()
        batch_neutral_log = logits[:, neutral_id].detach().cpu().numpy()

        for k, (i, j) in enumerate(pairs_ij[start:end]):
            entail_probs[i, j] = float(batch_entail_probs[k])
            contra_probs[i, j] = float(batch_contra_probs[k])
            neutral_probs[i, j] = float(batch_neutral_probs[k])
            class_mat[i, j] = int(class_preds[k])
            entail_logits[i, j] = float(batch_entail_log[k])
            contra_logits[i, j] = float(batch_contra_log[k])
            neutral_logits[i, j] = float(batch_neutral_log[k])
    id_map = {"entail_id": entail_id, "contra_id": contra_id, "neutral_id": neutral_id}
    return entail_probs, contra_probs, neutral_probs, entail_logits, contra_logits, neutral_logits, class_mat, id_map


def compute_greedy_semantic(
    question: str,
    greedy_text: str,
    sample_texts: List[str],
    tokenizer,
    nli_model,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size_nli: int = 8,
    max_length: int = 512
):
    id2label = getattr(nli_model.config, "id2label", None)
    _id2label = {}
    for k, v in id2label.items():
        try:
            ki = int(k)
        except Exception:
            ki = k
        _id2label[ki] = str(v).lower()

    def _find_label_id(substrs):
        for i, lab in _id2label.items():
            for s in substrs:
                if s in lab:
                    return int(i)
        return None

    entail_id = _find_label_id(["entail"])
    contra_id = _find_label_id(["contra", "contradict"])
    neutral_id = _find_label_id(["neutral"])

    n = len(sample_texts)
    if n == 0:
        z = np.zeros((0,), dtype=np.float32)
        return z, z, z, z, z, z, z, z, {}

    if question is not None:
        responses = [f"{question} {t}".strip() for t in sample_texts]
        greedy = f"{question} {greedy_text}".strip()
    else:
        responses = [f"{t}".strip() for t in sample_texts]
        greedy = f"{greedy_text}".strip()

    premises = responses
    hypotheses = [greedy] * n

    entail = np.zeros((n,), dtype=np.float32)
    contra = np.zeros((n,), dtype=np.float32)
    entail_logits = np.zeros((n,), dtype=np.float32)
    contra_logits = np.zeros((n,), dtype=np.float32)


    entail_re = np.zeros((n,), dtype=np.float32)
    contra_re = np.zeros((n,), dtype=np.float32)
    entail_logits_re = np.zeros((n,), dtype=np.float32)
    contra_logits_re = np.zeros((n,), dtype=np.float32)

    bs = max(1, int(batch_size_nli))
    for start in range(0, n, bs):
        end = min(n, start + bs)

        enc = tokenizer(
            premises[start:end],
            hypotheses[start:end],
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            out = nli_model(**enc)
            logits = out.logits  
            probs = F.softmax(logits, dim=-1)  

        entail_probs = probs[:, entail_id].detach().cpu().numpy()
        contra_probs = probs[:, contra_id].detach().cpu().numpy()

        entail_log = logits[:, entail_id].detach().cpu().numpy()
        contra_log = logits[:, contra_id].detach().cpu().numpy()

        entail[start:end] = entail_probs.astype(np.float32)
        contra[start:end] = contra_probs.astype(np.float32)
        entail_logits[start:end] = entail_log.astype(np.float32)
        contra_logits[start:end] = contra_log.astype(np.float32)


    bs = max(1, int(batch_size_nli))
    for start in range(0, n, bs):
        end = min(n, start + bs)

        enc = tokenizer(
            hypotheses[start:end],
            premises[start:end],
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            out = nli_model(**enc)
            logits = out.logits  
            probs = F.softmax(logits, dim=-1)

        entail_probs_re = probs[:, entail_id].detach().cpu().numpy()
        contra_probs_re = probs[:, contra_id].detach().cpu().numpy()

        entail_log_re = logits[:, entail_id].detach().cpu().numpy()
        contra_log_re = logits[:, contra_id].detach().cpu().numpy()

        entail_re[start:end] = entail_probs_re.astype(np.float32)
        contra_re[start:end] = contra_probs_re.astype(np.float32)
        entail_logits_re[start:end] = entail_log_re.astype(np.float32)
        contra_logits_re[start:end] = contra_log_re.astype(np.float32)

    id_map = {"entail_id": entail_id, "contra_id": contra_id, "neutral_id": neutral_id}
    
    return entail, contra, entail_logits, contra_logits, entail_re, contra_re, entail_logits_re, contra_logits_re, id_map


def compute_sentence_greedy_semantic(
    question: str,               
    greedy_text: str,
    sample_texts: List[str],
    tokenizer,
    nli_model,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size_nli: int = 8,
    max_length: int = 512,
):
    nlp = spacy.load("en_core_web_sm")
    id2label = getattr(nli_model.config, "id2label", None)
    _id2label: Dict[int, str] = {}
    for k, v in id2label.items():
        try:
            ki = int(k)
        except Exception:
            ki = k
        _id2label[int(ki)] = str(v).lower()

    def _find_label_id(substrs):
        for i, lab in _id2label.items():
            for s in substrs:
                if s in lab:
                    return int(i)
        return None

    entail_id = _find_label_id(["entail"])
    contra_id = _find_label_id(["contra", "contradict"])
    neutral_id = _find_label_id(["neutral"])  

    n = len(sample_texts)
    if n == 0:
        z = np.zeros((0,), dtype=np.float32)
        return z, z, z, z, {}

    if question is not None:
        responses = [f"{question} {t}".strip() for t in sample_texts]
        greedy_sentences = [f"{question} {sent.text.strip()}".strip() for sent in nlp(greedy_text).sents]
    else:
        responses = [t.strip() for t in sample_texts]
        greedy_sentences = [sent.text.strip() for sent in nlp(greedy_text).sents]

    m = len(greedy_sentences)
    if m == 0:
        z = np.zeros((0,), dtype=np.float32)
        return z, z, z, z, {}

    premises: List[str] = []
    hypotheses: List[str] = []
    pairs: List[Tuple[int, int]] = []

    for i in range(m):
        h = greedy_sentences[i]
        for j in range(n):
            pairs.append((i, j))
            premises.append(responses[j])   
            hypotheses.append(h)            

    entail = np.zeros((m, n), dtype=np.float32)
    contra = np.zeros((m, n), dtype=np.float32)
    entail_logits = np.zeros((m, n), dtype=np.float32)
    contra_logits = np.zeros((m, n), dtype=np.float32)

    bs = max(1, int(batch_size_nli))
    for start in range(0, len(pairs), bs):
        end = min(len(pairs), start + bs)

        enc = tokenizer(
            premises[start:end],
            hypotheses[start:end],
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            out = nli_model(**enc)
            logits = out.logits
            probs = F.softmax(logits, dim=-1)

        ent_p = probs[:, entail_id].detach().cpu().numpy()
        con_p = probs[:, contra_id].detach().cpu().numpy()
        ent_l = logits[:, entail_id].detach().cpu().numpy()
        con_l = logits[:, contra_id].detach().cpu().numpy()

        for k, (i, j) in enumerate(pairs[start:end]):
            entail[i, j] = float(ent_p[k])
            contra[i, j] = float(con_p[k])
            entail_logits[i, j] = float(ent_l[k])
            contra_logits[i, j] = float(con_l[k])

    meta = {
        "entail_id": entail_id,
        "contra_id": contra_id,
        "neutral_id": neutral_id
    }

    return entail, contra, entail_logits, contra_logits, meta


def compute_sample_cosine( 
    question: str,
    sample_texts: List[str],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size_cos: int = 8,
    max_length: int = 512,
) -> np.ndarray:

    M = len(sample_texts)
    if M == 0:
        return np.zeros((0, 0), dtype=np.float32)
    if M == 1:
        return np.ones((1, 1), dtype=np.float32)

    sbert = SentenceTransformer(config.SBERT_MODEL_PATH)
    sbert = sbert.to(device)

    if question is not None:
        texts = [f"{question} {a.strip()}" for a in sample_texts]
    else:
        texts = [f"{a.strip()}" for a in sample_texts]

    emb = sbert.encode(
        texts,
        batch_size=batch_size_cos,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)
    
    cosine_matrix = emb @ emb.T
    np.fill_diagonal(cosine_matrix, 1.0)

    return cosine_matrix

def generate_steps(dataset, num):
    cot_extractor_model_path = config.COT_EXTRACTOR_MODEL_PATH
    
    model = LLM(
        model=cot_extractor_model_path,
        tensor_parallel_size=1,
        trust_remote_code=True,
        dtype="float16",
        max_model_len=32768,
    )

    generation_config = SamplingParams(
        top_p=1.0,
        temperature=0.0,
        max_tokens=2048,
    )

    with open(f"{config.PROJECT_PATH}/system_prompt.txt") as f:
        sys_prompt = f.read()

    messages: List[List[Dict[str, str]]] = []
    refs: List[Tuple[int, str, Optional[int]]] = []

    for i, item in enumerate(dataset):
        question = item["question"]

        greedy = item.get("greedy_results", {}) or {}
        final_answer = greedy.get("answer")
        think = greedy.get("reasoning")
        if final_answer and think:
            messages.append([
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"## Question\n{question}\n\n## Thought\n{think}\n\n## Final Answer\n{final_answer}"},
            ])
            refs.append((i, "greedy", None))
        else:
            dataset[i]["greedy_results"]["cots"] = []

        for j in range(num):
            sample = item.get("sample_results", [])[j]
            final_answer = sample.get("answer")
            think = sample.get("reasoning")
            if final_answer and think:
                messages.append([
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": f"## Question\n{question}\n\n## Thought\n{think}\n\n## Final Answer\n{final_answer}"},
                ])
                refs.append((i, "sample", j))
            else:
                sample["cots"] = []

    if not messages:
        return dataset

    outputs = model.chat(messages=messages, sampling_params=generation_config)

    for ref, out in zip(refs, outputs):
        i, kind, j = ref
        response = (out.outputs[0].text or "").strip()
        cots = [c.strip() for c in response.split("[STEP]") if c.strip()]
        if kind == "greedy":
            dataset[i]["greedy_results"]["cots"] = cots
        else:
            dataset[i]["sample_results"][j]["cots"] = cots

    return dataset