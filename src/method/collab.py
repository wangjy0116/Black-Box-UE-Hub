import os
import re
import json
import math
import time
import random
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from openai import OpenAI
from tqdm import tqdm
from src.utils import check_entailment
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
from src import config
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
api_base = os.getenv("OPENROUTER_API_BASE")

def eval_ans(dataset_name, question, ans, reference, method: str = "gpt_cls"):

    if dataset_name == "truthfulqa_mc":
        labels = [o["label"] for o in item["options"]]
        if ans.upper() in labels:
            if ans.upper() == reference:
                return True
            else:
                return False
        if len(ans)>1 and ans[1] == '.' and ans[0].upper() in labels:
            if ans[0].upper() == reference:
                return True
            else:
                return False
        return False
        
    if isinstance(reference, str):
        reference = [reference]
    if method == "gpt_cls":
        for ref in reference:
            if ans == ref:
                return True

        client = OpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=60,
        )

        ref_text = "\n".join(f"{i+1}. {g}" for i, g in enumerate(reference))
        prompt = (
            f"Question:\n{question}\n\n"
            f"Ground Truth Answer(s):\n{ref_text}\n\n"
            f"Based strictly on the ground truth provided above, is the following answer correct?\n"
            f"Do not use any external knowledge.\n"
            f"Respond only with 'Yes' or 'No'.\n\n"
            f"Answer: {ans}"
        )
        if dataset_name == "coqa":
            prompt = "Context:\n" + item["story"] + "\n\n" + prompt

        response = client.chat.completions.create(
            model="openai/gpt-5.1",
            messages=[{"role": "user", "content": prompt}],
            reasoning_effort="low",
        )
        response = response.choices[0].message.content.strip()

        if response.startswith("yes"):
            return True
        else:
            return False

    elif method == "nli_cls":
        for ref in reference:
            if ref == ans:
                return True
            e1 = check_entailment(question, ans, ref)
            e2 = check_entailment(question, ref, ans)
            if e1 == e2 == 1:
                return True
        return False
    else:
        raise NotImplementedError

def extract_conf(res: str) -> float:
    res_parsed = (
        res.split("\n")[-1]
        .replace("Confidence:", "")
        .replace("Confidence score:", "")
        .replace("Confidence Score:", "")
        .replace("confidence score:", "")
        .replace("(", "")
        .replace(")", "")
        .strip()
    )
    try:
        conf = float(res_parsed)
    except ValueError:
        logging.debug(f"res {res} res_parsed: {res_parsed}")
        conf = 0.0
    return conf

def extract_ans(res: str) -> str:
    if res.find("A:") == -1 and res.find("Answer:") == -1:
        logging.debug(f"wrong output format: {res}")
        return "Abstain"
    if "<answer>" in res or "I'm sorry" in res:
        logging.debug(f"Abstain from answering: {res}")
        return "Abstain"
        
    ans = (res.split("\n")[0]
        .split("Confidence")[0]
        .split("confidence")[0]
        .replace("A:", "")
        .replace("Answer:", "")
        .replace("{", "{{")
        .replace("}", "}}")
        .strip())

    if ans == "":
        ans = (res.split("\n")[1]
        .split("Confidence")[0]
        .split("confidence")[0]
        .replace("A:", "")
        .replace("Answer:", "")
        .replace("{", "{{")
        .replace("}", "}}")
        .strip())
    return ans

def extract_conf_metrics(res: str) -> Tuple[float, float, float]:
    res_parsed = {"ambiguity": None, "complexity": None, "ability": None}
    for line in res.split("\n"):
        text = line.strip().lower()
        for k in res_parsed.keys():
            if text.startswith(f"{k}:"):              
                try:
                    text_val = re.sub("[\(\[].*?[\)\]]", "", text.split(f"{k}:")[-1])
                    metric = float(text_val.strip())
                except ValueError:
                    logging.debug(f"metric_name {k} text: {text}")
                    metric = None
                res_parsed[k] = metric
    logging.debug(f"conf_metrics parsed {res_parsed}")
    return res_parsed

def softmax(l: List[float]) -> List[float]:
    return np.exp(l) / np.sum(np.exp(l), axis=0)


class BaseAgent:

    def __init__(self, id, model, temperature):
        self.id = id
        self.model = model
        self.temperature = temperature

    def generate_argument(self, dataset_name, record, assigned_stance: str) -> str:
        q = record["question"]
        prompt = f"""
You are participating in a debate on the question: '{q}'\nYour assigned stance on the question is '{assigned_stance}'
Generate some arguments/evidence (no more than three sentences) on why your assigned stance is correct.
If the question is ambiguous, address the assumptions/interpretation associated with your assigned stance.
Be concise! Exclude anything irrelavant or unhelpful in terms of supporting the stance!
Argument:
"""     
        if dataset_name != "coqa":
            res = self.model.generate_one([{"role": "user", "content": prompt}], temperature=self.temperature)
        else:
            res = self.model.generate_one([{"role": "user", "content": f"Context: {record['story']}\n{prompt}"}], temperature=self.temperature)
        return res.replace("Argument:", "").strip()

    def generate_self_evaluation(self, dataset_name, record, assigned_stance: str) -> str:
        task = record["question"]
        prompt = f"""
For the task: '{task}'\nHere is your previous response: '{assigned_stance}'
Evaluate your response by generating the following:
List related premises/assumptions you had in the response and your associated confidences (float, from 0 to 1),
following this format: Premise:<a premise>Confidence:<confidence>.
Then, determine whether the task is ambiguous. If so, output "[Ambiguous]", and list several possible interpretations,
following this format: Interpretation:<interpretation>. Mark the interpretation that matches your response with an asterisk (*) at the end of the line.
""" 
        if dataset_name == "coqa":
            res = self.model.generate_one([{"role": "user", "content": prompt}], temperature = self.temperature)
        else:
            res = self.model.generate_one([{"role": "user", "content": f"Context: {record['story']}\n{prompt}"}], temperature=self.temperature)
        return res.replace("Evaluation:", "").strip()

    def chat(self, prompt):
        return self.model.generate_one([{"role": "user", "content": prompt}], temperature = self.temperature)

class CoTAgent:
    def __init__(self, id, model, temperature):
        self.system_msg = "You are a helpful AI agent good at multi-hop and arithmetic reasoning."
        self.default_trigger_prompt = (
            "State your answer ({description}), then"
            " rate the level of ambiguity in the input query (a float from 0 to 1);"
            " rate the level of complexity of the input query (a float from 0 to 1);"
            " rate your level of ability for solving the input query (a float from 0 to 1);"
            " Note that your uncertainty on the correctness of your answer is affected by input ambiguity, task complexity, and your own knowledge and abilities."
            " Based on this,"
            " give a float (between 0 to 1) indicating your overall confidence on how likely that your answer is correct."
            " Follow this format:\nAnswer:<answer>\nAmbiguity:<ambiguity score>\nComplexity:<complexity score>\nAbility:<ability score>\nConfidence:<confidence>"
        )
        self.temperature = temperature
        self.model = model
        self.id = id

    def self_deliberate(self, dataset_name, item):
        query = item["question"]
        user1 = f"Question: {query}\nLet's think step by step."
        if dataset_name == "coqa":
            user1 = f"Context: {item['story']}\n" + user1
        intermediate = self.model.generate_one([{"role": "system", "content": self.system_msg}, {"role": "user", "content": user1}], self.temperature)
        if intermediate is not None : intermediate.strip()


        system2 = f"Question: {query}\n{intermediate}"
        if dataset_name == "coqa":
            system2 = f"Context: {item['story']}\n" + system2
        user2 = self.default_trigger_prompt.format(description=item["description"][:-1].lower())
        raw = self.model.generate_one([{"role": "user", "content": system2+"\n\n"+user2}], self.temperature)
        
        if raw is not None : raw.strip()

        return raw


class KnowledgeAgent:    
    def __init__(self, id, model, temperature):
        self.id = id
        self.default_trigger_prompt = (
            "State your answer ({description}), then"
            " rate the level of ambiguity in the input query (a float from 0 to 1);"
            " rate the level of complexity of the input query (a float from 0 to 1);"
            " rate your level of ability for solving the input query (a float from 0 to 1);"
            " Note that your uncertainty on the correctness of your answer is affected by input ambiguity, task complexity, and your own knowledge and abilities."
            " Based on this,"
            " give a float (between 0 to 1) indicating your overall confidence on how likely that your answer is correct."
            " Follow this format:\nAnswer:<answer>\nAmbiguity:<ambiguity score>\nComplexity:<complexity score>\nAbility:<ability score>\nConfidence:<confidence>"
        )
        self.temperature = temperature
        self.model = model

    def self_deliberate(self, dataset_name, item):
        query = item["question"]
        # step 1: generate background document
        prompt_doc = (
            f"Generate a background document to answer the given question:\n{query}\n"
            f"The background should be brief, in no more than 100 words)"
        )
        if dataset_name == "coqa":
            prompt_doc = f"Context: {item['story']}\n" + prompt_doc
        doc = self.model.generate_one([{"role": "user", "content": prompt_doc}], self.temperature)

        # step 2: condition on doc then ask for final answer + confidence
        messages_2 = [
            {"role": "user", "content": prompt_doc},
            {"role": "assistant", "content": doc},
            {"role": "user", "content": self.default_trigger_prompt.format(description=item["description"][:-1].lower())},
        ]
        raw = self.model.generate_one(messages_2, self.temperature)

        return raw


PROMPTING_STRATEGY_MAPPING = {
    "cot": CoTAgent,
    "knowledge": KnowledgeAgent
}

class Collab:

    def __init__(self, model_name, model, dataset_name, dev_dataset, test_dataset, temperature,
                num_agent = 5, max_workers = 10, agent_ensemble = True):
        self.model_name = model_name
        self.model = model
        self.dataset_name = dataset_name
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.temperature = temperature
        self.num_agent = num_agent
        self.agent_ensemble = agent_ensemble
        self.max_workers = max_workers
        self.mix_temperature = False
        self.save_path = f"{config.OUTPUT_DIR}/collab/{dataset_name}/{model_name}.json"
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        self.show = True

    def call_api(self, system, content):
        if system is not None:
            message = [{"role": "system", "content": system}, {"role": "user", "content": content}]
        else:
            message = [{"role": "user", "content": content}]
        return self.model.generate_one(message, self.temperature)

    def allocate_agent_slots(self, tau: float = 0.2):
        initialization = {"cot": 1, "pot": 0, "search": 0, "knowledge": 1}
        initial_agents = self.populate_expert_agents(initialization)
        adjusted_confidence_all = {k: [] for k in initialization.keys()}

        for item in tqdm(self.dev_dataset):
            for key, agent_group in initial_agents.items():
                for agent in agent_group:
                    res = agent.self_deliberate(self.dataset_name, item)
                    if "Abstain" in res:
                        adjusted_confidence_all[key].append(0.0)
                    else:
                        correctness = eval_ans(self.dataset_name, item["question"], extract_ans(res), item["ground_truth"], method = "gpt_cls")
                        adjusted_confidence_all[key].append((2 * correctness - 1) * extract_conf(res))

        logging.debug(f"adjusted_confidence_all: {adjusted_confidence_all}")
        adjusted_confidence_mean = {
            k: float(np.mean(adjusted_confidence_all[k])) for k in initialization.keys() if adjusted_confidence_all[k]
        }
        logging.debug(f"adjusted_confidence_mean: {adjusted_confidence_mean}")

        final_allocation = {}
        confidence_filtered = {k: c for k, c in adjusted_confidence_mean.items() if c >= tau}
        logging.debug(f"confidence_filtered: {confidence_filtered} from {adjusted_confidence_mean.items()}")
        if len(confidence_filtered):
            portions = softmax(list(confidence_filtered.values())) 
            confidence_softmax = {_k: portions[i] for (i, _k) in enumerate(confidence_filtered.keys())} 
            confidence_sorted = dict(sorted(confidence_softmax.items(), key=lambda item: item[1], reverse=True))
            logging.info(f"{portions}, sorted: {confidence_sorted}")
            sorted_keys = list(confidence_sorted.keys())
            if self.num_agent == 2 and len(sorted_keys) > 1:
                final_allocation[sorted_keys[0]] = 1
                final_allocation[sorted_keys[1]] = 1
            else:
                for i, k in enumerate(sorted_keys):
                    final_allocation.update({k: int(np.floor(portions[i] * self.num_agent))})
                diff = self.num_agent - sum(final_allocation.values())
                if diff > 0:
                    final_allocation[sorted_keys[0]] += diff

            if sum(final_allocation.values()) != self.num_agent:
                logging.debug(
                    f"slots unmatched: {adjusted_confidence_mean}\n{confidence_filtered}\n{final_allocation}"
                )
                raise ValueError
        else:
            logging.info("All agents produced confidence below threshold. Check input task difficulty or initialization.")
            top_key = max(adjusted_confidence_mean, key=adjusted_confidence_mean.get)
            final_allocation[top_key] = self.num_agent

        logging.info(f"Final allocation of expert agents (for each model): {final_allocation}")
        return final_allocation


    def populate_expert_agents(self, selection: Dict[str, int]) -> Dict[str, Any]:
        pool = {k: [] for k in selection.keys()}
        pretrained_instances = {}
        for k, count in selection.items():
            for i in range(count):
                temperature = np.random.rand() + 0.5 if self.mix_temperature else self.temperature
                pool[k].append(
                    PROMPTING_STRATEGY_MAPPING[k](
                        id=f"{k}-agent_{i+1}_{self.model}", model=self.model, temperature=temperature # model_type=default_model
                    )
                )
        logging.debug(f"agents populated: {pool}")
        return pool

    def populate_general_agents(self, size: int) -> List[BaseAgent]:
        return [BaseAgent(f"general_{i+1}", self.model, self.temperature) for i in range(size)]

    def stance_generation(self, record, agents_mapping: Dict[str, Any]):
        votes = []
        for specialization, grouped_agents in agents_mapping.items():
            for agent in grouped_agents:
                res = agent.self_deliberate(self.dataset_name, record)
                vote = tuple((agent.id, extract_ans(res), extract_conf(res), extract_conf_metrics(res)))  # Tuple of four items; keep this order.
                votes.append(vote)
        logging.info(f"votes: {votes}")
        stances = self.construct_stances(votes, record)  # Build unique stances and their confidence scores.
        return votes, stances

    def generate_premises_and_ratings(self,
        argument: str, stance: str, scale: List[str], notice: str, verify: Optional[bool] = False
    ):
        verify_text = (
            "List premises/assumptions from the above argument that help determine the correctness of the stance"
            " Each premise should be as simple as possible (single-hop), relevant to the stance, and independent"
            " (avoid pronoun as the subject, use named entity instead), surrounded in square brackets, e.g. [Premise:<premise>]"
        ) if verify else ""
        prompt = (
            f"Here is an argument '{argument}' for the stance '{stance}'\n{notice}\n"
            f"{verify_text}"
            "Evaluate how good the argument is regarding logical consistency, clarity, and conciseness. "
            f"For each of the three aspects, choose one of {scale} as your rating. Do NOT provide any reasoning."
            "Follow this format: [Consistency:<rating>, Clarity:<rating>, Conciseness:<rating>]"
        )
        raw_res = self.call_api(None, prompt).split("\n")
        premises_parsed, ratings = [], {
            "Consistency": "modest",
            "Clarity": "modest",
            "Conciseness": "modest",
        }
        for row in raw_res:
            if "Premise" in row:
                premises_parsed.append(
                    row.replace("Premise", "")
                    .replace(":", "")
                    .replace("[", "")
                    .replace("]", "")
                    .strip()
                )
            elif "Consistency:" in row:
                parts = row.strip().split(",")
                if len(parts) != 3:
                    logging.debug(f"ratings wrong format {row}")
                ratings_parsed = [part[part.find(":") + 1 :].replace(".", "").replace("]", "").strip() for part in parts]
                ratings.update(zip(ratings, ratings_parsed))
        logging.debug(f"raw_res: {raw_res}")
        return premises_parsed, ratings

    def verify_premises(self, premises: List[str]):
        verification_results = [0] * 3
        cases = ""
        for premise in premises:
            q_rewrite = self.call_api(None, 
                f"Rewrite the premise '''{premise}''' to a question starting with an interrogative pronoun:"
            ).strip()        
            ans_self_ask = self.call_api(None, 
                f"Answer the question: {q_rewrite}. If you are unsure about the correct answer, simply output 'Unknown'"
            )
            if "Unknown" in ans_self_ask:
                verification_results[1] += 1
            else:
                nli_cls = check_entailment(None, ans_self_ask, premise)
                if nli_cls == -1:
                    check = self.call_api(None,
                        f"Is the statement {premise} True or False? Briefly explain why."
                    ).strip()
                    if "False" in check:
                        cases += f"For {premise}: {check}\n"
                    else:
                        nli_cls = 0
                verification_results[1 - nli_cls] += 1

                logging.debug(f"verify_premise res: {premise}; {q_rewrite}; {ans_self_ask}; {nli_cls}")

        factuality_score = (
            1 * verification_results[0] + 0.5 * verification_results[1] + 0 * verification_results[2]
        ) / sum(verification_results)

        return factuality_score, cases

    def generate_feedback(self, record, 
            answer: str, argument: str,
            rating_scales: Optional[List[str]] = ["bad", "modest", "good", "excellent"],
            theta: Optional[float] = 0.6, rater_choice: Optional[str] = None, verify: Optional[bool] = False):

        query = record["question"]
        stance = f"The answer to the question '{query}' is '{answer}'"
        notice = f"Note in earlier debate, you were {rater_choice} the answer corresponding to this argument." if rater_choice in ["supporting", "opposing"] else ""
        ans_consistency = check_entailment(query, argument, stance)
        if self.dataset_name == "coqa":
            stance = f"Context: {record['story']}\n" + stance
        premises, ratings = self.generate_premises_and_ratings(argument, stance, rating_scales, notice, verify)
        if premises:
            factuality_score, unfactual_premises = self.verify_premises(premises)
        else:
            factuality_score = 1
            unfactual_premises = "No unfactual premise."
        if ans_consistency != 1:
            adjusted_consistency_level = np.max(
                [0, rating_scales.index(ratings["Consistency"].lower()) + ans_consistency - 1]
            )
            extra = (
                "(The answer cannot be deduced from the arguments)" if ans_consistency == -1 else ""
            )
            ratings["Consistency"] = f"{rating_scales[adjusted_consistency_level]}{extra}"
            
        logging.debug(f"ratings values {list(ratings.values())}")
        ratings_numerical = [
            rating_scales.index(g_norm) if g_norm in rating_scales else 0
            for g_norm in [
                str(grade).split("(")[0].strip().lower() if grade is not None else ""
                for grade in ratings.values()
            ]
        ]
        soundness_score = float(
            theta * factuality_score
            + (1 - theta) * np.mean(ratings_numerical) / (len(rating_scales) - 1)
        )

        feedback = f"Logic consistency: {ratings['Consistency']}\nClarity: {ratings['Clarity']}\nConciseness: {ratings['Conciseness']}\n{unfactual_premises}\n"
        return soundness_score, feedback

    def summarize_feedback(self, feedback: List[str], rating_scales: Optional[List[str]] = ["bad", "modest", "good", "excellent"]) -> str:
        summary_sys_msg = "Be factual and concise in your summarization."
        prompt = (
            f"Summarize by combining the feedback from several individuals: {feedback}"
            f" Note the rating scales are {rating_scales}. You should aggregate the ratings from both sides."
            " Also, highlight any unfactual premise mentioned.\n"
            "Summary:"
        )
        summary_all = self.call_api(summary_sys_msg, prompt)
        logging.debug(f"summary_all: {summary_all}")
        return summary_all

    def get_collective_feedback(self, record, arguments: Dict[str, str],
        n_raters: Optional[int] = 1, verify: Optional[bool] = False):
        ranking = []
        for answer, argument in arguments.items():
            soundness_scores, feedback_all = [], []
            for _ in range(n_raters):
                score_supporting, feedback_supporting = self.generate_feedback(record, answer, argument, rater_choice="supporting")
                score_opposing, feedback_opposing = self.generate_feedback(record, answer, argument, rater_choice="opposing", verify=verify)
                soundness_scores.extend([score_supporting, score_opposing])
                feedback_all.extend([feedback_supporting, feedback_opposing])

            ranking.append(tuple(
                [
                    answer,
                    argument,
                    np.mean(soundness_scores),
                    self.summarize_feedback(feedback_all),
                ]
            ))      

        ranking = sorted(ranking, key=lambda t: t[2], reverse=True)
        return ranking

    def revote(self, record, mappings: List[Tuple[BaseAgent, float, str, str]]):
        question = record["question"]
        votings_all = []
        for mapping in mappings:
            agent, initial_conf, original_observations, new_observations = mapping
            prompt = f"Given the question: '{question}', \n{original_observations}\nHere are some new observations:\n{new_observations}"
            prompt += "Give your final answer (as short as possible). "
            prompt += "Considering your original belief, group consensus and new observations, and weighing arguments from multiple sides (including your own), "
            prompt += "give rationales for whether you would adjust your original confidence score.\nFollow this format:\n"
            prompt += "Answer:\nRationales:"
        
            if self.dataset_name == "coqa":
                prompt = f"Context: {record['story']}\n" + prompt
            revote_intermediate = agent.chat(prompt)
            logging.debug(f"revote_intermediate: {revote_intermediate}")
            rationale = revote_intermediate.split("Rationales:")[-1].strip()
            prompt_trigger = f"Recall your orignal confidence for your answer was {initial_conf}. "
            prompt_trigger += f"Given the rationale:\n'''{rationale}'''\nprovide your final confidence score (a float from 0 to 1). Follow this format:\nConfidence:"
            revote_conf = agent.chat(prompt_trigger)

            try:
                conf_parsed = float(
                    revote_conf.split("Rationales:")[0].split("Confidence:")[-1].split("\n")[0].strip()
                )
            except ValueError:
                logging.debug(f"conf_parsed: {revote_conf}")
                conf_parsed = 1.0
            
            votings_all.append(tuple((mapping[0].id, extract_ans(revote_intermediate), conf_parsed, rationale)))  # Drop mapping[0].model_type.
        logging.debug(f"votings_all {votings_all}")
        return votings_all

    def construct_stances(self, votes,
        record,
        filter_abstain: Optional[bool] = True,
        conf_rationales: Optional[List[str]] = None):

        query = record["question"]
        classes = {}
        if filter_abstain:
            votes = [vote for vote in votes if "abstain" not in vote[1].lower()]  # Filter on element 1.
        for i, (id, ans, verb_conf, *_) in enumerate(votes):
            logging.debug(f"construct_stances: {i, id, ans, verb_conf}")
            conf_rationale = conf_rationales[i] if conf_rationales else None  # for now, only include in stage 2
            if i == 0:
                classes.update({ans: [verb_conf, 1, conf_rationale]})
            else:
                merged_answer_class = None
                for unique_class in classes.keys():
                    if eval_ans(self.dataset_name, query, ans, unique_class, method="nli_cls"):
                        # not a new class, merge with the equivalent answer class
                        merged_answer_class = unique_class
                        prev_verb_conf, prev_count, prev_rationale = classes[merged_answer_class]
                        classes[merged_answer_class] = [
                            (prev_verb_conf * prev_count + verb_conf) / (prev_count + 1),
                            prev_count + 1,
                            prev_rationale,
                        ]
                        break
                if not merged_answer_class:
                    # a new class, add the answer to the answer set
                    classes.update({ans: [verb_conf, 1, conf_rationale]})

        stances = []
        for ans_class, (verb_conf, count, rationale) in classes.items():
            stances.append([ans_class, float(verb_conf), int(count), rationale])
        logging.debug(f"final ans_set: {stances}")
        return stances

    def deliberate_with_feedback(self, record, agents: List[BaseAgent],
        stance_list, # stances：[ans_class, float(verb_conf), int(count), rationale]
        self_popularity: bool = False, verify: bool = False):
        """Stage 2: m general agents, each with an assigned stance and corresponding confidence (verb/logit-based). 
        Group deliberation process:

        agents: the list of general agents
        stance_list: [<unique_answer, verb_confidence, count, conf_rationale>], sorted by count ascendingly
        """
        agents_observations_mapping = []
        class_count = [stance_stats[2] for stance_stats in stance_list]
        arguments = {t[0]: None for t in stance_list}  # stance: argument
        if len(stance_list) == 1:
            # if reaching consensus in stage 1, assign only 3 general agents, and no feedback needed
            unique_answer, initial_verb_conf, *_ = stance_list[0]
            # initial_conf = max(initial_verb_conf, initial_seq_prob)
            initial_conf = initial_verb_conf
            original_observations = (
                f"Your original answer is '{unique_answer}', with a confidence of {initial_conf:.2f}"
            )
            new_observations = "Through deliberation, all other people have agreed with your answer, reaching a consensus."
            agents_observations_mapping = [
                tuple((agents[i], initial_conf, original_observations, new_observations)) for i in range(3)
            ]
        else:
            assignment_quantities = np.cumsum(class_count)  # Cumulative sum.
            logging.debug(f"assignment_quantities {assignment_quantities}")
            m = assignment_quantities[-1]
            logging.debug(f"agent count: {agents}, stance_list {stance_list} m {m}")

            curr_stance_index = 0
            for index, agent in enumerate(agents):  # If long_form is false, this only iterates len(assignment_quantities) times.
                # stance_list already sorted
                if index == assignment_quantities[curr_stance_index]:
                    curr_stance_index += 1
                if curr_stance_index == len(assignment_quantities):
                    break
                assigned_ans, initial_verb_conf, count, _ = stance_list[curr_stance_index]  
                initial_conf =initial_verb_conf    
                agents_observations_mapping.append([agent, assigned_ans, initial_conf, count])

                if not arguments.get(assigned_ans):
                    arguments[assigned_ans] = agent.generate_argument(self.dataset_name, record, assigned_ans)  # Generate the argument for why this answer is correct; temperature should be 0.1.
            logging.info(f"deliberator arguments: {arguments}")
        
            ranking = self.get_collective_feedback(record, arguments, verify = verify)
            logging.debug(f"ranking {record['question']}; {ranking}")

            # construct new observations and update agents_observations_mapping
            for i, mapping in enumerate(agents_observations_mapping):
                agent, assigned_ans, initial_conf, count = mapping
                original_observations = (
                    f"Your original answer is {assigned_ans}, with a confidence of {initial_conf:.2f}"
                )
                
                for rank, (ans, argument, soundness, feedback) in enumerate(ranking):
                    general_feedback = (
                        f"'''{argument}'''\n, which received the following rating and feedback from other deliberators:"
                        f"Soundness score: {soundness:.2f} (ranked {rank+1} out of {len(ranking)})\n"
                        f"Feedback: {feedback}"
                    )
                    if ans == assigned_ans:
                        feedback_supporting = f"An argument supporting your original answer is\n{general_feedback}"
                        if not self_popularity:
                            feedback_supporting += f"\nNote that {count-1} other {'person' if count == 2 else 'people'} (out of {m}) also agreed with you."
                    else:
                        feedback_opposing = f"An argument from the opposing side is\n{general_feedback}"
                        if not self_popularity:
                            feedback_opposing += f"\nNote {m - count} {'person' if m - count == 1 else 'people'} disagreed with you."                   
                        
                self_estimate_popularity = f"Based on the evidence presented, estimate how many deliberators (including yourself, out of {m}) are on your side." if self_popularity else ""
                new_observations = f"Recall that your original confidence was {initial_conf:.2f}\n{feedback_opposing}\n{feedback_supporting}\n{self_estimate_popularity}"
                agents_observations_mapping[i] = tuple((agent, initial_conf, original_observations, new_observations))

        # re-voting with new observations (and the corresponding ranking/feedback, if no early consensus)
        final_votes_raw = self.revote(record, agents_observations_mapping) # (mapping[0].id, extract_ans(revote_intermediate), conf_parsed, rationale)
        rationales = [vote[-1] for vote in final_votes_raw]
        final_set = self.construct_stances( # [ans_class, float(verb_conf), int(count), rationale]
            final_votes_raw, record, conf_rationales=rationales
        )
        return arguments, final_votes_raw, final_set

    def save_vote_history(self, 
        record,
        original_votes,
        original_stance_list,
        final_votes,
        final_stance_list,
        final_majority,
        output_filepath,
        dataset,
    ):
        vote_keys = ["agent_id", "answer", "verbal_confidence",  "confidence_metrics"]
        stance_keys = ["answer_class", "avg_verbal_confidence",  "count", "rationale"]

        original_votes_with_keys = [dict(zip(vote_keys, original_vote)) for original_vote in original_votes]    
        original_stance_list_with_keys = [dict(zip(stance_keys, stance)) for stance in original_stance_list]
        
        vote_keys[-1] = "rationale"
        final_votes_with_keys = [dict(zip(vote_keys, final_vote)) for final_vote in final_votes]
        final_stance_list_with_keys = [dict(zip(stance_keys, stance)) for stance in final_stance_list]

        record.update({
            "original_votes": original_votes_with_keys,
            "original_stances": original_stance_list_with_keys,
            "final_votes": final_votes_with_keys,
            "final_stances": final_stance_list_with_keys,
            "final_majority_ans": final_majority[0],
            "final_verbal_confidence": final_majority[1],
        })
        with open(f"{output_filepath}l", mode="a") as fp:
            fp.write(json.dumps(record, indent=4) + "\n")

    def agents_deliberation_single_thread(
        self, record,
        expert_agent_pool: Dict[str, Any],
        general_agent_pool: List[Any],
        output_filepath,
        dataset: str
    ):
        # Stage 1
        original_votes, stances = self.stance_generation(record, expert_agent_pool)
        if not len(stances):
            logging.info(f"Skip {record['id']}")
            return
        original_stance_list = sorted(stances, key=lambda t: t[2])
            
        # Stage 2
        arguments, final_votes, final_ans_set = self.deliberate_with_feedback(
            record, general_agent_pool, original_stance_list, self_popularity=False, verify=False, 
        )
        for i, ans_cls in enumerate(original_stance_list):
            if not ans_cls[-1]:
                original_stance_list[i][-1] = arguments[ans_cls[0]]

        if not len(final_ans_set):
            logging.info(f"All agents abstained after deliberation for {record['id']}")
            return
        final_majority = sorted(list(final_ans_set), key=lambda t: t[2])[-1]
        logging.debug(f"final majority vote: {final_majority}")
        self.save_vote_history(
                record,
                original_votes,
                original_stance_list,
                final_votes,
                final_ans_set,
                final_majority,
                output_filepath,
                dataset,
            )

    def generate(self):
            
        if self.agent_ensemble:
            slots_allocation = self.allocate_agent_slots()
            expert_agent_pool = self.populate_expert_agents(slots_allocation)  
        else:
            expert_agent_pool = self.populate_expert_agents(
                dict({"knowledge": self.num_agent // 2, "cot": self.num_agent // 2})
            )

        general_agent_pool = self.populate_general_agents(self.num_agent)

        logging.debug(f"expert agents: {expert_agent_pool}\ngeneral agents: {general_agent_pool}")

        if "qwen3-4b" in self.model_name.lower():
            self.max_workers = 1
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(
                    self.agents_deliberation_single_thread,
                    item,
                    expert_agent_pool,
                    general_agent_pool,
                    self.save_path,
                    self.dataset_name
                )
                for item in self.test_dataset
            ]

            for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating", unit="item"):
                try:
                    future.result()
                except Exception as e:
                    logging.exception(f"One evaluation item failed: {e}")

    def extract(self):
        data = []
        buf_lines = []
        depth = 0
        in_str = False
        escape = False

        def feed_line(line: str):
            nonlocal depth, in_str, escape
            for c in line:
                if escape:
                    escape = False
                    continue
                if c == "\\":
                    if in_str:
                        escape = True
                    continue
                if c == '"':
                    in_str = not in_str
                    continue
                if not in_str:
                    if c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1

        with open(f"{self.save_path}l", "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                buf_lines.append(line)
                feed_line(line)

                if depth == 0 and buf_lines:
                    obj_text = "".join(buf_lines).strip()
                    buf_lines = []
                    if not obj_text:
                        continue
                    try:
                        obj = json.loads(obj_text)
                    except json.JSONDecodeError as e:
                        raise ValueError(
                            f"JSON object decode error ending near line {line_no}: {e}\n"
                            f"Object head: {obj_text[:200]}"
                        ) from e
                    data.append(obj)
                    
        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    
    def calculate(self):
        with open(self.save_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        for item in dataset:
            final_stances = item["final_stances"]
            sum_count = 0
            for stance in final_stances:
                sum_count += stance["count"]
            
            for stance in final_stances:
                stance["avg_verbal_confidence"] = (stance["avg_verbal_confidence"]*stance["count"])/(sum_count)

            best = max(final_stances, key=lambda x: (x["count"], x["avg_verbal_confidence"]))

            answer_class = best["answer_class"]
            avg_verbal_confidence = best["avg_verbal_confidence"]

            item["pred_answer"] = answer_class
            item["pred_score"] = avg_verbal_confidence

        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)
