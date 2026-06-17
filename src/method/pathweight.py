from typing import Any, Dict, List, Optional, Tuple
import os
import json
import math
from collections import defaultdict
import numpy as np
import networkx as nx
from tqdm import tqdm
import random
from src import config
from src.utils import compute_sample_semantic, build_sample_path, sample_generate, extract_final_answer, generate_steps, load_nli_model

class PathWeight:

    def __init__(
        self,
        model_name, 
        model,
        dataset_name,
        dev_dataset,
        test_dataset,
        sample_num,
        temperature,
        alpha_katz: float = 0.1,
        beta_katz: float = 1.0,
        pathconv_M: int = 200,
        pathconv_L: int = 12,
        pathconv_gamma: float = 1.0,
        pathweight_L: int = 12,
        seed: int = 42,
        merge_equivalent_steps: bool = False,
    ):
        self.alpha_katz = alpha_katz
        self.beta_katz = beta_katz
        self.pathconv_M = pathconv_M
        self.pathconv_L = pathconv_L
        self.pathconv_gamma = pathconv_gamma
        self.pathweight_L = pathweight_L
        self.seed = seed
        self.merge_equivalent_steps = merge_equivalent_steps
        self.model = model
        self.dataset_name = dataset_name
        self.dataset = test_dataset
        self.sample_num = sample_num
        self.temperature = temperature
        self.save_path = f"{config.OUTPUT_DIR}/pathweight/{dataset_name}/{model_name}.json"
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

    def build_user_prompt(self, item: Dict[str, Any]) -> str:
        return f"Question: {item['question']}"

    def build_coqa_user_prompt(self, item):
        return f"Context: {item['story']}\nQuestion: {item['question']}"

    def parse_cot(self, text: Any):
        if not isinstance(text, str):
            return None

        raw = text.strip()
        if not raw:
            return None

        def _normalize_reasoning(x: Any) -> Optional[str]:
            if x is None:
                return None
            if isinstance(x, str):
                s = x.strip()
                return s if s else None
            if isinstance(x, list):
                parts = [str(v).strip() for v in x if str(v).strip()]
                return "\n".join(parts) if parts else None
            s = str(x).strip()
            return s if s else None

        def _extract_from_obj(obj: Any) -> Optional[str]:
            if not isinstance(obj, dict):
                return None
            reasoning = obj.get("reasoning")
            if reasoning is None:
                reasoning = obj.get("cot")
            return _normalize_reasoning(reasoning)

        for parser in (json.loads, ast.literal_eval):
            try:
                obj = parser(raw)
            except Exception:
                continue

            reasoning = _extract_from_obj(obj)
            if reasoning is not None:
                return reasoning

            if isinstance(obj, str) and obj.strip():
                try:
                    obj2 = parser(obj.strip())
                except Exception:
                    continue
                reasoning = _extract_from_obj(obj2)
                if reasoning is not None:
                    return reasoning

        return None

    def generate(self, **kwargs: Any):
        sample_test_path, _ = build_sample_path(self.save_path)
        if self.dataset_name == "coqa":
            build_user=self.build_coqa_user_prompt
        else:
            build_user=self.build_user_prompt

        if not os.path.exists(sample_test_path):
            sample_generate(
                model=self.model,
                dataset=self.dataset,
                save_path=sample_test_path,
                build_system=self.build_system_prompt,
                build_user=build_user,
                sample_num=self.sample_num,
                temperature=self.temperature,
                **kwargs
            )

    def extract(self):
        sample_test_path, _ = build_sample_path(self.save_path)
        with open(sample_test_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        for item in tqdm(dataset, desc="Extracting steps and answers"):
            for sample in item.get("sample_results", []):
                if "answer" not in sample.keys():
                    text = sample.get("text", "")
                    sample["answer"] = extract_final_answer(text)
                if "reasoning" not in sample.keys():
                    text = sample.get("text", "")
                    cot = self.parse_cot(text)
                    sample["reasoning"] = cot

            if "answer" not in item["greedy_results"].keys():
                text = item["greedy_results"].get("text", "")
                item["greedy_results"]["answer"] = extract_final_answer(text)
            if "reasoning" not in item["greedy_results"].keys():
                text = item["greedy_results"].get("text", "")
                cot = self.parse_cot(text)
                item["greedy_results"]["reasoning"] = cot

        with open(sample_test_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)


    def _normalize_answer_for_graph(self, ans: Any) -> Optional[str]:
        if ans is None:
            return None
        s = str(ans).strip()
        return s if s else None

    def _cluster_from_eq_mat(self, eq_mat: np.ndarray) -> List[List[int]]:
        n = eq_mat.shape[0]
        if n == 0:
            return []
        visited = [False] * n
        clusters = []

        for i in range(n):
            if visited[i]:
                continue
            stack = [i]
            visited[i] = True
            comp = []

            while stack:
                u = stack.pop()
                comp.append(u)
                for v in range(n):
                    if not visited[v] and eq_mat[u, v]:
                        visited[v] = True
                        stack.append(v)

            clusters.append(sorted(comp))
        return clusters


    def _clip_text_by_tokens(self, text: str, tokenizer, max_tokens: int = 512) -> str:
        if not isinstance(text, str):
            return ""
        try:
            ids = tokenizer.encode(text, add_special_tokens=False)
            ids = ids[:max_tokens]
            return tokenizer.decode(ids, skip_special_tokens=True)
        except Exception:
            return text[:1000]


    def semantic_equiv_matrix(
        self,
        question: Optional[str],
        texts: List[str],
        tokenizer,
        nli_model
    ) -> Tuple[np.ndarray, Any]:

        n = len(texts)
        if n == 0:
            return np.zeros((0, 0), dtype=bool), {}
        if n == 1:
            return np.ones((1, 1), dtype=bool), {"entail_id": 1}

        _, _, _, _, _, _, class_mat, id_map = compute_sample_semantic(
            question=question,
            sample_texts=texts,
            tokenizer=tokenizer,
            nli_model=nli_model
        )
        entail_id = id_map["entail_id"]
        eq_mat = np.zeros((n, n), dtype=bool)
        for i in range(n):
            eq_mat[i, i] = True
            for j in range(i + 1, n):
                same = (class_mat[i, j] == entail_id) and (class_mat[j, i] == entail_id)
                eq_mat[i, j] = same
                eq_mat[j, i] = same
        return eq_mat, id_map


    def add_step_nodes_and_intra_edges(
        self,
        G: nx.DiGraph,
        valid_samples: List[Dict[str, Any]],
    ):
        all_step_texts = []
        all_step_meta = []
        chain_last_step_nodes = []
        answer_texts = []

        for i, s in enumerate(valid_samples):
            prev = "Q"

            for j, st in enumerate(s["cots"]):
                node_id = f"S_{i}_{j}"
                G.add_node(
                    node_id,
                    node_type="step",
                    text=st,
                    sample_id=i,
                    step_id=j,
                    weight=1,
                )

                G.add_edge(prev, node_id, edge_type="intra")
                prev = node_id

                all_step_texts.append(st)
                all_step_meta.append((i, j, node_id, st))

            chain_last_step_nodes.append(prev)
            answer_texts.append(s["answer"])

        return all_step_texts, all_step_meta, chain_last_step_nodes, answer_texts

    def add_inter_edges_between_equivalent_steps(
        self,
        G: nx.DiGraph,
        item: Dict[str, Any],
        all_step_texts: List[str],
        all_step_meta: List[Tuple[int, int, str, str]],
        tokenizer,
        nli_model
    ):
        step_components = []
        step_node_to_cluster = {}

        if len(all_step_texts) == 0:
            return step_components, step_node_to_cluster

        if len(all_step_texts) == 1:
            _, _, node_id, _ = all_step_meta[0]
            step_components = [[0]]
            step_node_to_cluster[node_id] = 0
            G.nodes[node_id]["weight"] = 1
            return step_components, step_node_to_cluster

        question = item.get("question", "")

        step_eq_mat, _ = self.semantic_equiv_matrix( 
            question=question,
            texts=all_step_texts,
            tokenizer = tokenizer,
            nli_model = nli_model
        )
        for a in range(len(all_step_meta)):
            ci, sj, ni, ti = all_step_meta[a]
            for b in range(a + 1, len(all_step_meta)):
                ck, sl, nj, tj = all_step_meta[b]
                if ci == ck:
                    continue
                if step_eq_mat[a, b]:
                    G.add_edge(ni, nj, edge_type="inter")
                    G.add_edge(nj, ni, edge_type="inter")

        step_components = self._cluster_from_eq_mat(step_eq_mat)
        for cid, comp in enumerate(step_components):
            comp_size = len(comp)
            for idx in comp:
                _, _, node_id, _ = all_step_meta[idx]
                step_node_to_cluster[node_id] = cid
                G.nodes[node_id]["weight"] = comp_size

        return step_components, step_node_to_cluster

    def add_merged_answer_nodes(
        self,
        G: nx.DiGraph,
        item: Dict[str, Any],
        answer_texts: List[str],
        chain_last_step_nodes: List[str],
        tokenizer,
        nli_model
    ):

        answer_nodes = []
        answer_text_to_node = {}
        chain_answer_node_ids = []

        if len(answer_texts) == 0:
            return answer_nodes, answer_text_to_node, chain_answer_node_ids

        question = item.get("question", "")

        if len(answer_texts) == 1:
            ans_node = "A_0"
            G.add_node(
                ans_node,
                node_type="answer",
                text=answer_texts[0],
                weight=1,
            )
            for last_step_node in chain_last_step_nodes:
                G.add_edge(last_step_node, ans_node, edge_type="intra")
            return [ans_node], {answer_texts[0]: ans_node}, [ans_node]

        ans_eq_mat, _ = self.semantic_equiv_matrix(  
            question=question,
            texts=answer_texts,
            tokenizer=tokenizer,
            nli_model = nli_model
        )

        answer_components = self._cluster_from_eq_mat(ans_eq_mat)

        idx_to_answer_node = {}

        for cid, comp in enumerate(answer_components):
            rep_text = answer_texts[comp[0]]
            ans_node = f"A_{cid}"
            G.add_node(
                ans_node,
                node_type="answer",
                text=rep_text,
                weight=len(comp),  
            )
            answer_nodes.append(ans_node)
            answer_text_to_node[rep_text] = ans_node

            for idx in comp:
                idx_to_answer_node[idx] = ans_node

        for i, last_step_node in enumerate(chain_last_step_nodes):
            ans_node = idx_to_answer_node[i]
            G.add_edge(last_step_node, ans_node, edge_type="intra")
            chain_answer_node_ids.append(ans_node)

        return answer_nodes, answer_text_to_node, chain_answer_node_ids

    def build_reasoning_graph(
        self,
        item: Dict[str, Any],
        valid_samples: List[Dict[str, Any]],
        tokenizer,
        nli_model,
    ):
        G = nx.DiGraph()
        G.add_node("Q", node_type="question", text=item.get("question", ""), weight=1)

        all_step_texts, all_step_meta, chain_last_step_nodes, answer_texts = self.add_step_nodes_and_intra_edges(
                G=G,
                valid_samples=valid_samples,
            )

        step_components, step_node_to_cluster = self.add_inter_edges_between_equivalent_steps(
                G=G,
                item=item,
                all_step_texts=all_step_texts,
                all_step_meta=all_step_meta,
                tokenizer=tokenizer,
                nli_model=nli_model
            )

        answer_nodes, answer_text_to_node, chain_answer_node_ids = self.add_merged_answer_nodes(
                G=G,
                item=item,
                answer_texts=answer_texts,
                chain_last_step_nodes=chain_last_step_nodes,
                tokenizer=tokenizer,
                nli_model=nli_model
            )

        return G, {
            "answer_nodes": answer_nodes,
            "answer_text_to_node": answer_text_to_node,
            "chain_answer_node_ids": chain_answer_node_ids,
            "step_nodes": [x[2] for x in all_step_meta],
            "step_clusters": step_components,
            "step_node_to_cluster": step_node_to_cluster,
        }

    def _remove_loops_by_dropping_inter_edges(self, G: nx.DiGraph) -> nx.DiGraph:
        H = G.copy()
        max_iter = 10000
        it = 0

        while it < max_iter:
            it += 1
            try:
                cycle = next(nx.simple_cycles(H), None)
            except Exception:
                break

            if cycle is None:
                break

            removed = False
            m = len(cycle)
            for idx in range(m):
                u = cycle[idx]
                v = cycle[(idx + 1) % m]
                if H.has_edge(u, v):
                    etype = H.edges[u, v].get("edge_type", "")
                    if etype == "inter":
                        H.remove_edge(u, v)
                        removed = True
                        break

            if not removed:
                u = cycle[0]
                v = cycle[1 % len(cycle)]
                if H.has_edge(u, v):
                    H.remove_edge(u, v)

        return H


    def compute_cenconf(
        self,
        G: nx.DiGraph,
        answer_nodes: List[str],
        alpha: float = 0.1,
        beta: float = 1.0,
    ):
        if len(answer_nodes) == 0:
            return {}

        if len(answer_nodes) == 1:
            return {answer_nodes[0]: 1.0}

        try:
            katz = nx.katz_centrality_numpy(G, alpha=alpha, beta=beta)
        except Exception:
            katz = nx.katz_centrality(G, alpha=min(alpha, 0.01), beta=beta, max_iter=5000)

        scores = {a: max(float(katz.get(a, 0.0)), 0.0) for a in answer_nodes}
        s = sum(scores.values())
        if s <= 0:
            return {a: 1.0 / len(answer_nodes) for a in answer_nodes}
        return {a: v / s for a, v in scores.items()}



    def enumerate_paths_with_cutoff(self, G: nx.DiGraph, source: str, target: str, cutoff: int):
        try:
            return list(nx.all_simple_paths(G, source=source, target=target, cutoff=cutoff))
        except Exception:
            return []


    def compute_pathconv_sampling(
        self,
        G: nx.DiGraph,
        answer_nodes: List[str],
        M: int = 200,
        max_path_length: int = 12,
        gamma: float = 1.0,
    ):
        if len(answer_nodes) == 0:
            return {}

        raw_scores = {}
        for ans_node in answer_nodes:
            paths = self.enumerate_paths_with_cutoff(G, "Q", ans_node, cutoff=max_path_length)
            raw_scores[ans_node] = len(paths)


        s = sum(raw_scores.values())
        if s <= 0:
            return {a: 1.0 / len(answer_nodes) for a in answer_nodes}
        return {a: v / s for a, v in raw_scores.items()}


    def sample_one_random_path_score(
        self,
        G: nx.DiGraph,
        answer_node: str,
        max_path_length: int = 12,
        gamma: float = 1.0,
    ):
        v = "Q"
        w = 1.0

        for _ in range(max_path_length):
            if v == answer_node:
                return w

            succ = sorted(G.successors(v))
            if len(succ) == 0:
                return 0.0

            v = random.choice(succ)
            w *= gamma

        if v == answer_node:
            return w
        return 0.0
 
    def build_pathweight_merged_graph(
        self,
        G: nx.DiGraph,
        meta: Dict[str, Any],
    ) -> Tuple[nx.DiGraph, str, List[str], Dict[str, str]]:

        MG = nx.DiGraph()
        source = "MQ"
        MG.add_node(source, node_type="question", text="Q", weight=1)

        step_node_to_cluster = meta.get("step_node_to_cluster", {})
        answer_nodes = meta.get("answer_nodes", [])

        raw_to_merged = {"Q": source}

        for raw_node, cid in step_node_to_cluster.items():
            mnode = f"M{raw_node}"
            text = G.nodes[raw_node].get("text", "")
            cluster_size = sum(1 for _, c in step_node_to_cluster.items() if c == cid)

            MG.add_node(
                mnode,
                node_type="step",
                text=text,
                weight=cluster_size,
            )
            raw_to_merged[raw_node] = mnode

        merged_answer_nodes = []
        for ans_node in answer_nodes:
            mans = f"M{ans_node}"
            MG.add_node(
                mans,
                node_type="merged_answer",
                text=G.nodes[ans_node]["text"],
                weight=float(G.nodes[ans_node].get("weight", 1)),
            )
            raw_to_merged[ans_node] = mans
            merged_answer_nodes.append(mans)

        for u, v, edata in G.edges(data=True):
            if edata.get("edge_type") != "intra":
                continue
            if u not in raw_to_merged or v not in raw_to_merged:
                continue

            mu = raw_to_merged[u]
            mv = raw_to_merged[v]
            if mu == mv:
                continue

            MG.add_edge(mu, mv)

        return MG, source, merged_answer_nodes, raw_to_merged

    def compute_pathweight(
        self,
        G: nx.DiGraph,
        meta: Dict[str, Any],
        max_path_length: int = 12,
        id=None
    ):
        answer_nodes = meta.get("answer_nodes", [])
        if len(answer_nodes) == 0:
            return {}

        MG, source, merged_answer_nodes, _ = self.build_pathweight_merged_graph(G, meta)
        raw_scores = {}
        for orig_ans_node, mans in zip(answer_nodes, merged_answer_nodes):
            total = 0.0
            paths = self.enumerate_paths_with_cutoff(MG, source, mans, cutoff=max_path_length)

            for p in paths:
                score = 1.0
                for v in p:
                    if MG.nodes[v].get("node_type", "") in ["merged_answer", "answer", "raw_answer"]:
                        continue
                    score *= float(MG.nodes[v].get("weight", 1.0))
                total += score

            raw_scores[orig_ans_node] = total

        s = sum(raw_scores.values())
        if s <= 0:
            return {a: 1.0 / len(answer_nodes) for a in answer_nodes}
        return {a: v / s for a, v in raw_scores.items()}

    def calculate(self):
        sample_test_path, _ = build_sample_path(self.save_path)
        tokenizer, nli_model = load_nli_model(config.NLI_MODEL_PATH)

        with open(sample_test_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        if "cots" not in dataset[0]["sample_results"][0]:
            dataset = generate_steps(dataset, self.sample_num)

        with open(sample_test_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)

        random.seed(self.seed)
        np.random.seed(self.seed)

        output = []
        for item in dataset:
            output.append({"id": item["id"], "question": item["question"],
                    "description": item["description"], "ground_truth": item["ground_truth"],
                    "greedy_answer": item["greedy_results"]["answer"]})
            if self.dataset_name == "truthfulqa_mc":
                output[-1]["options"] = item["options"]
            if self.dataset_name == "coqa":
                output[-1]["story"] = item["story"]

        valid_num = min(self.sample_num // 2, 1)
        for sample_item, output_item in tqdm(zip(dataset, output), total=len(dataset)):

            samples = sample_item.get("sample_results", [])[:self.sample_num]
            valid_samples = []
            for s in samples:
                steps = s.get("cots", [])
                ans = self._normalize_answer_for_graph(s.get("answer"))
                if isinstance(steps, list) and len(steps) > 0 and ans is not None:
                    clean_steps = [str(x).strip() for x in steps if str(x).strip()]
                    if len(clean_steps) > 0:
                        valid_samples.append({
                            "cots": clean_steps,
                            "answer": ans,
                        })
                        
            if len(valid_samples) <= valid_num:
                output_item["cenconf_scores"] = {}
                output_item["pathconv_scores"] = {}
                output_item["pathweight_scores"] = {}
                output_item["cenconf_answer"] = None
                output_item["cenconf_conf"] = None
                output_item["pathconv_answer"] = None
                output_item["pathconv_conf"] = None
                output_item["pathweight_answer"] = None
                output_item["pathweight_conf"] = None
                output_item["pred_answer"] = None
                output_item["pred_score"] = None
                continue

            G, meta = self.build_reasoning_graph(
                item=sample_item,
                valid_samples=valid_samples,
                tokenizer=tokenizer,
                nli_model=nli_model,
            )
            G = self._remove_loops_by_dropping_inter_edges(G)
            answer_nodes = meta["answer_nodes"]
            cen_scores = self.compute_cenconf(
                G=G,
                answer_nodes=answer_nodes,
                alpha=self.alpha_katz,
                beta=self.beta_katz,
            )

            pathconv_scores = self.compute_pathconv_sampling(
                G=G,
                answer_nodes=answer_nodes,
                M=self.pathconv_M,
                max_path_length=self.pathconv_L,
                gamma=self.pathconv_gamma,
            )

            pathweight_scores = self.compute_pathweight(
                G=G,
                meta=meta,
                max_path_length=self.pathweight_L,
                id = output_item["id"],
            )

            output_item["cenconf_scores"] = cen_scores
            output_item["pathconv_scores"] = pathconv_scores
            output_item["pathweight_scores"] = pathweight_scores

            best_cen = max(cen_scores.items(), key=lambda x: x[1])
            best_conv = max(pathconv_scores.items(), key=lambda x: x[1])
            best_weight = max(pathweight_scores.items(), key=lambda x: x[1])

            output_item["cenconf_answer"] = G.nodes[best_cen[0]]["text"]
            output_item["cenconf_confidence"] = float(best_cen[1])

            output_item["pathconv_answer"] = G.nodes[best_conv[0]]["text"]
            output_item["pathconv_confidence"] = float(best_conv[1])

            output_item["pathweight_answer"] = G.nodes[best_weight[0]]["text"]
            output_item["pathweight_confidence"] = float(best_weight[1])
            output_item["pred_answer"] = output_item["pathweight_answer"]
            output_item["pred_score"] = output_item["pathweight_confidence"]

        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=4)
