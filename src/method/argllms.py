import json
import os
from typing import Any, Dict, List, Optional
from copy import deepcopy
import re
from tqdm import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.utils import build_sample_path, sample_generate, extract_final_answer
from src import config

def baseline_formatter(response):
    if "false" in response.lower() and "true" in response.lower():
        return response.lower().rindex("true") > response.lower().rindex("false")
    elif "true" in response.lower():
        return True
    elif "false" in response.lower():
        return False
    else:
        return False

class ParseSkipSampleError(Exception):
    pass


class BaselinePrompts:
    @staticmethod
    def chatgpt(statement, direct=False, **_):
        if not direct:
            constraints = {
                "constraint_prefix": "Let's think step by step.",
            }

            prompt = f"""
            Claim: {statement}

            Instructions:
                Consider the claim and determine whether it is true or false.
                Think step by step before providing the final answer.
                Utilize critical thinking and logical reasoning in your evaluation.
                Ensure clarity in your reasoning."""

            return prompt, constraints, baseline_formatter

        constraints = {
            "constraint_prefix": "Answer:",
            "constraint_options": [" True", " False"],
            "constraint_end_after_options": True,
        }

        prompt = f"""
            Claim: {statement}

            Instructions:
                Consider the claim and determine whether it is true or false.
                Utilize critical thinking and logical reasoning in your evaluation.
                Ensure clarity in your reasoning."""

        return prompt, constraints, baseline_formatter

    @staticmethod
    def analyst(statement, direct=False, **_):
        if not direct:
            constraints = {
                "constraint_prefix": "Let's think step by step.",
            }

            instructions = (
                f"You are an analyst evaluating whether statements are true or false. "
                f'For the statement:\n\nStatement: "{statement}"\n\nplease determine '
                f"whether the statement is correct, accurate and truthful. "
                "Please think step by step before providing the final answer."
            )
            return instructions, constraints, baseline_formatter

        constraints = {
            "constraint_prefix": "Result:",
            "constraint_options": [" True", " False"],
            "constraint_end_after_options": True,
        }

        instructions = (
            f"You are an analyst evaluating whether statements are true or false. "
            f'For the statement:\n\nStatement: "{statement}"\n\nplease determine '
            f"whether the statement is correct, accurate and truthful. "
            "Please give the result in the form 'Result: True' if the statement "
            "is true and 'Result: False' if the statement is false."
        )

        return instructions, constraints, baseline_formatter

    @staticmethod
    def opro(statement, direct=False, **_):
        if not direct:
            constraints = {
                "constraint_prefix": "Let's think step by step.",
            }

            prompt = f"""Please provide an assessment based on the factuality and truthfulness of the following statement. If the statement is partially false or has to be interpreted in a very specific way to be considered true, you should consider it false.
    Statement: {statement}
    Now take a deep breath, and determine whether the statement is true or false."""


            return prompt, constraints, baseline_formatter

        constraints = {
            "constraint_prefix": "Assessment:",
            "constraint_options": [" True", " False"],
            "constraint_end_after_options": True,
        }

        prompt = f"""Please provide an assessment based on the factuality and truthfulness of the following statement. If the statement is partially false or has to be interpreted in a very specific way to be considered true, you should consider it false. Please think step by step before providing the final answer.
Statement: {statement}
Now take a deep breath, think step by step and determine whether the statement is true or false."""

        return prompt, constraints, baseline_formatter


class ArgumentMiningPrompts:
    @staticmethod
    def chatgpt(statement, support=False, **_):
        def formatter(argument, prompt):
            if "N/A" in argument or "n/a" in argument:
                return "N/A"
            return re.sub(
                " +",
                " ",
                "".join(re.findall(r"(.*[.?!])", argument.split("\n\n")[0])).strip(
                    '"\n'
                ),
            )

        return (
            f"""
        Claim: {statement}

        Instructions:
            Provide a concise argument {"supporting" if support else "opposing"} the claim in less than 2 sentences.
            Utilize critical thinking and logical reasoning in your argument.
            Ensure clarity in your reasoning.
            Avoid circular reasoning or fallacious arguments.
            If you cannot return a valid and convincing argument for this claim, reply N/A.

        {"Supporting" if support else "Opposing"} Argument for '{statement}':""",
            {},
            formatter,
        )

    @staticmethod
    def debater(statement, support=False, **_):
        def formatter(argument, prompt):
            if "N/A" in argument or "n/a" in argument:
                return "N/A"
            return argument

        return (
            (
                "You are a professional debater who will try to provide arguments on a topic even if "
                "they go against your personal believes. Please give a brief, one-sentence argument "
                f"{'in favour of' if support else 'against'} the statement:\n\nStatement: \"{statement}\"\n\n"
                "Note that the provided argument should provide a clear justification why the considered "
                f"statement is {'true and accurate' if support else 'untrue or inaccurate'}. "
                "The argument should also be as self-contained as possible. "
                "Please reply only with the argument sentence without any further commentary. "
                "If you are truly unable to provide such an argument, reply N/A."
            ),
            {},
            formatter,
        )

    @staticmethod
    def opro(statement, support=False, **_):
        def formatter(argument, prompt):
            if "N/A" in argument or "n/a" in argument:
                return "N/A"
            return argument

        return (
            f"""Please provide a single short argument {"supporting" if support else "attacking"} the following candidate answer. Construct the argument so it refers to the truthfulness of the candidate answer. Only provide an argument if you think there is a valid and convincing {"support" if support else "attack"} for this candidate answer (there is a non-zero probability that the claim is true), otherwise return: N/A.
        {statement}
        Now take a deep breath and come up with an argument.
        Argument:""",
            {},
            formatter,
        )


class UncertaintyEvaluatorPrompts:
    @staticmethod
    def chatgpt(statement, verbal=False, **_):
        if verbal:
            certainty_dict = {
                "certain": 0.95,
                "almost certain": 0.8,
                "quite certain": 0.65,
                "moderately certain": 0.5,
                "slightly certain": 0.35,
                "almost uncertain": 0.2,
                "uncertain": 0.05,
            }

            def formatter(certainty):
                return certainty_dict[certainty.replace("Certainty:", "").strip()]

            constraints = {
                "constraint_prefix": "Certainty:",
                "constraint_options": certainty_dict.keys(),
                "constraint_end_after_options": True,
            }

            prompt = f"""Task Description:
        Provide your certainty for a given argument. The certainty should reflect the level of confidence in the validity or strength of the argument.

        Instructions:
        1. Consider the argument provided and evaluate its clarity, logical coherence, and the amount of evidence presented.
        2. Provide your certainty using the following numerical mapping: {" ".join([k + ": " + str(certainty_dict[k]) for k in certainty_dict.keys()])} 
        3. Explain the reasoning behind your certainty, taking into account factors such as ambiguity in language, lack of evidence, and logical fallacies.

        Argument: {statement}

        Additional Guidance:
        Consider the context in which the argument is presented and any potential biases or assumptions that may affect its credibility.
        Evaluate the strength of the evidence provided and the coherence of the argument's structure.
        Take into account any potential counterarguments or alternative interpretations that could impact the certainty of the argument.

        Please respond in the following form:

        Certainty: Your certainty on the argument"""

        else:

            def formatter(certainty):
                return float(certainty.replace("Certainty Score:", "").strip().split()[0]) # 修改

            constraints = {
                "constraint_prefix": "Certainty Score:",
                "constraint_options": [f" {x / 100}" for x in range(0, 101)],
                "constraint_end_after_options": True,
            }

            prompt = f"""Task Description:
        Generate a certainty score for a given argument. The certainty score should reflect the level of confidence in the validity or strength of the argument.
        
        Instructions:
        1. Consider the argument provided and evaluate its clarity, logical coherence, and the amount of evidence presented.
        2. Assign a numerical certainty score on a scale from 0 to 1, where 0 indicates maximum uncertainty, 1 indicates maximum certainty, and values in between represent varying degrees of certainty.
        3. Explain the reasoning behind the certainty score, taking into account factors such as ambiguity in language, lack of evidence, and logical fallacies.
        
        Argument: {statement}
        
        Additional Guidance:
        Consider the context in which the argument is presented and any potential biases or assumptions that may affect its credibility.
        Evaluate the strength of the evidence provided and the coherence of the argument's structure.
        Take into account any potential counterarguments or alternative interpretations that could impact the certainty of the argument.
        
        Please respond in the following form:
        
        Certainty Score: Your certainty score on the argument"""

        return prompt, constraints, formatter

    @staticmethod
    def analyst(statement, claim=None, support=False, verbal=False, topic=False, **_):
        if not topic and claim is None:
            raise ValueError(
                "Claim is required for the analyst prompt without topic flag, but was None"
            )

        if verbal:

            def formatter(output):
                likelihood = output.replace("Confidence in argument:", "").strip()
                likelihood_dict = {
                    "fully confident": 0.95,
                    "highly confident": 0.8,
                    "quite confident": 0.65,
                    "moderately confident": 0.5,
                    "slightly confident": 0.35,
                    "not very confident": 0.2,
                    "not confident at all": 0.05,
                }
                return likelihood_dict[likelihood]

            relation = "supports" if support else "refutes"
            options = [
                "fully confident",
                "highly confident",
                "quite confident",
                "moderately confident",
                "slightly confident",
                "not very confident",
                "not confident at all",
            ]
            q = '"'
            constraints = {
                "constraint_prefix": "Confidence in argument:",
                "constraint_options": options,
                "constraint_end_after_options": True,
            }

            if topic:
                instructions = (
                    f"You are an analyst evaluating the validity of statements. "
                    f'For the statement:\n\nStatement: "{statement}"\n\nplease give your confidence '
                    f"that the statement is correct, accurate and truthful. "
                )
            else:
                instructions = (
                    f"You are an analyst evaluating the validity and relevance of arguments. "
                    f'For the argument:\n\nArgument: "{statement}"\n\nplease give your confidence '
                    f"that the argument presents a compelling case {'in favour of' if support else 'against'} "
                    f'the statement:\n\nStatement: "{claim}"\n\nYour assessment should be based '
                    f"on how well the argument {'supports' if support else 'refutes'} the considered "
                    "statement as well as the correctness, accuracy and truthfulness of the given argument. "
                )

            return (
                instructions
                + (
                    f"Your response should be chosen out of the options: "
                    f'{", ".join([q + o + q for o in options])}. '
                    "Please respond in the following form:"
                    f"\n\nConfidence in {'argument' if not topic else 'statement'}: "
                    f"Your confidence in the {'argument' if not topic else 'statement'} validity"
                ),
                constraints,
                formatter,
            )

        def formatter(output):
            likelihood_re = re.compile(
                r"""
                (?ix)                       
                \*{0,2}\s*likelihood\s*\*{0,2}   
                \s*[:：]\s*                 
                \*{0,2}\s*                  
                (?P<num>[0-9]+(?:\.[0-9]+)?)
                \s*%                        
                \s*\*{0,2}                  
                """,
                flags=re.IGNORECASE | re.VERBOSE,
            )
            m = likelihood_re.search(output)
            likelihood = float(m.group("num")) if m else None
            try:
                return int(likelihood) / 100
            except Exception:
                raise ParseSkipSampleError(f"Cannot parse likelihood from output: {output}")

        constraints = {
            "constraint_prefix": "Likelihood:",
            "constraint_options": [f" {l}%" for l in range(0, 101)],
            "constraint_end_after_options": True,
        }

        if topic:
            instructions = (
                f"You are an analyst evaluating the validity of the candidate answer. "
                f'\n\n{statement}\n\nPlease give your confidence '
                f"that the candidate answer is correct, accurate and truthful. "
                f"Your response should be between 0% and 100% with 0% indicating that the "
                f"considered answer is definitely invalid, 100% indicating that the considered answer is "
            )
        else:
            instructions = (
                "You are an analyst evaluating the validity and relevance of arguments. "
                f'For the argument:\n\nArgument: "{statement}"\n\nplease give your confidence '
                f"that the argument presents a compelling case {'in favour of' if support else 'against'} "
                f'the candidate answer:\n\n{claim}\n\nYour assessment should be based '
                f"on how well the argument {'supports' if support else 'refutes'} the considered "
                "answer as well as the correctness, accuracy and truthfulness of the given argument. "
                f"Your response should be between 0% and 100% with 0% indicating that the "
                f"considered argument is definitely invalid, 100% indicating that the considered argument is "
            )

        return (
            instructions
            + (
                "definitely valid and values in between indicating various levels of "
                "uncertainty. Your estimates should be well-calibrated, so feel free to "
                "err on the side of caution and output moderate probabilities if you are "
                "not completely sure in your assessment. "
                "Please respond in the following form:"
                "\n\nLikelihood: The predicted likelihood that the considered "
                f"{'argument' if not topic else 'answer'} is valid"
            ),
            constraints,
            formatter,
        )


    @staticmethod
    def opro(statement, verbal=False, support=False, claim=None, topic=False):
        if verbal:

            def formatter(output):
                likelihood = output.replace("Quality score:", "").strip()
                likelihood_dict = {
                    "fully confident": 0.95,
                    "highly confident": 0.8,
                    "quite confident": 0.65,
                    "chances about even": 0.5,
                    "quite unconfident": 0.35,
                    "highly unconfident": 0.2,
                    "fully unconfident": 0.05,
                }
                return likelihood_dict[likelihood]

            options = [
                "fully confident",
                "highly confident",
                "quite confident",
                "chances about even",
                "quite unconfident",
                "highly unconfident",
                "fully unconfident",
            ]
            q = '"'
            constraints = {
                "constraint_prefix": "Quality score:",
                "constraint_options": options,
                "constraint_end_after_options": True,
            }

            prompt = f"""Please provide a quality score (as a verbal confidence taken from the list that follows) based on the factuality and truthfulness of the following statement. If the statement is partially false or has to be interpreted in a very specific way to be considered true, it should receive a low score. Statement: {statement} Now take a deep breath and give a quality score, out of the options {", ".join([q + o + q for o in options])}."""
        else:

            def formatter(certainty):
                return int(certainty.replace("Quality score:", "").strip().split()[0]) / 100 # 修改

            constraints = {
                "constraint_prefix": "Quality score:",
                "constraint_options": [f" {x}" for x in range(0, 101)],
                "constraint_end_after_options": True,
            }

            if topic:
                prompt = f"""Please provide a quality score (as a single numerical value between 0 and 100) based the factuality and truthfulness of the following statement. If the statement is partially false or has to be interpreted in a very specific way to be considered true, it should receive a low score.
Statement: {statement}
Now take a deep breath and give a quality score."""
            else:
                prompt = f"""Please provide a quality score (as a single numerical value between 0 and 100) based on factuality, relevance and effectiveness, for how well the following argument {"supports" if support else "attacks"} the claim. If the argument suggests that the claim is partially false or must be interpreted in a specific way to be considered true, it should receive a low score.
Claim: {claim}
{"Supporting" if support else "Attacking"} argument: {statement}
Now take a deep breath and give a quality score."""

        return prompt, constraints, formatter


class SumAggregation:
    def __init__(self) -> None:
        pass

    def aggregate_strength(self, attackers, supporters, state):
        aggregate = 0
        for a in attackers:
            aggregate -= state[a]

        for s in supporters:
            aggregate += state[s]

        return aggregate
    
    def __str__(self) -> str:
        return __class__.__name__

class QuadraticMaximumInfluence:
    def __init__(self, conservativeness) -> None:
        self.conservativeness = conservativeness

    def compute_strength(self, weight, aggregate):
        strength = weight

        scaled_aggregate = aggregate / self.conservativeness
        h = scaled_aggregate**2 / (1 + scaled_aggregate**2)

        if (aggregate > 0):
            strength += h * (1 - weight)
        else:
            strength -= h * weight

        return strength

    def __str__(self) -> str:
        return __class__.__name__ + f"({self.conservativeness})"

class ProductAggregation:
    def __init__(self) -> None:
        pass

    def aggregate_strength(self, attackers, supporters, state):
        support_value = 1
        for a in attackers:
            support_value *= 1-state[a]

        attack_value = 1
        for s in supporters:
            attack_value *= 1-state[s]

        return support_value - attack_value

    def __str__(self) -> str:
        return __class__.__name__

class LinearInfluence:
    def __init__(self, conservativeness) -> None:
        self.conservativeness = conservativeness

    def compute_strength(self, weight, aggregate):
        strength = weight
        if (aggregate > 0):
            strength += aggregate * (1-weight)/self.conservativeness
        else:
            strength += aggregate*weight/self.conservativeness

        return strength

    def __str__(self) -> str:
        return __class__.__name__ + f"({self.conservativeness})"

import math
class EulerBasedInfluence:
    def __init__(self) -> None:
        pass

    def compute_strength(self, weight, aggregate):
        return 1 - (1-weight**2) / (1 + weight * math.exp(aggregate))

    def __str__(self) -> str:
        return __class__.__name__

class Argument:
    def __init__(self, name, arg, initial_weight, strength=None, attackers=None, supporters=None):
        self.name = name
        self.arg = arg
        self.initial_weight = initial_weight
        self.strength = strength
        self.attackers = attackers
        self.supporters = supporters
        self.parent = None

        if type(initial_weight) != int and type(initial_weight) != float:
            raise TypeError("initial_weight must be of type integer or float")

        if strength is None:
            self.strength = initial_weight

        if attackers is None:
            self.attackers = []

        if supporters is None:
            self.supporters = []

    def get_name(self):
        return self.name

    def get_arg(self):
        return self.arg

    def add_attacker(self, attacker):
        self.attackers.append(attacker)

    def add_supporter(self, supporter):
        self.supporters.append(supporter)

    def add_parent(self, parent):
        self.parent = parent

    def get_initial_weight(self):
        return self.initial_weight

    def reset_initial_weight(self, weight):
        self.initial_weight = weight

    def __repr__(self) -> str:
        return (f"Argument: {self.arg}, initial weight: {self.initial_weight}, strength: {self.strength}, attackers:"
                f"{self.attackers}, supporters: {self.supporters}")

    def __str__(self) -> str:
        return (f"Argument: {self.arg}, initial weight: {self.initial_weight}, strength: {self.strength}, attackers:"
                f"{self.attackers}, supporters: {self.supporters}")
        
    def _to_shallow_dict(self):
        return {
            'name': self.name,
            'argument': self.arg,
            'initial_weight': self.initial_weight,
            'strength': self.strength,
        }

    @classmethod
    def _from_shallow_dict(cls, d):
        return cls(
            d['name'],
            d['argument'],
            d['initial_weight'],
            strength=d['strength'],
        )

class Attack:
    def __init__(self, attacker, attacked) -> None:
        self.attacker = attacker
        self.attacked = attacked

    def get_attacker(self):
        return self.attacker

    def get_attacked(self):
        return self.attacked

    def __repr__(self) -> str:
        return f"Attack({self.attacker}, {self.attacked})"

    def __str__(self) -> str:
        return f"Attack by {self.attacker} to {self.attacked}"

class Support:
    def __init__(self, supporter, supported) -> None:
        self.supporter = supporter
        self.supported = supported

    def get_supporter(self):
        return self.supporter

    def get_supported(self):
        return self.supported


class BAG:

    def __init__(self, path=None):
    
        self.arguments = {}
        self.attacks = []
        self.supports = []
        
        self.path = path

        if (path is None):
            pass
        else:
            with open(os.path.abspath(path), "r") as f:
                for line in f.readlines():
                    k_name = line.split("(")[0]
                    if k_name in string.whitespace:
                        pass
                    else:
                        k_args = re.findall(rf"{k_name}\((.*?)\)", line)[0].replace(" ", "").split(",")
                        if k_name == "arg":
                            argument = Argument(k_args[0], float(k_args[1]), None, [], [])
                            self.arguments[argument.name] = argument

                        elif k_name == "att":
                            attacker = self.arguments[k_args[0]]
                            attacked = self.arguments[k_args[1]]
                            self.add_attack(attacker, attacked)

                        elif k_name == "sup":
                            supporter = self.arguments[k_args[0]]
                            supported = self.arguments[k_args[1]]
                            self.add_support(supporter, supported)

    def add_attack(self, attacker, attacked):
        if type(attacker) != Argument:
            raise TypeError("attacker must be of type Argument")

        if type(attacked) != Argument:
            raise TypeError("attacked must be of type Argument")

        if attacker.name in self.arguments:
            attacker = self.arguments[attacker.name]
        else:
            self.arguments[attacker.name] = attacker

        if attacked.name in self.arguments:
            attacked = self.arguments[attacked.name]
        else:
            self.arguments[attacked.name] = attacked
            
        attacked.add_attacker(attacker)

        self.attacks.append(Attack(attacker, attacked))

    def add_support(self, supporter, supported):
        if type(supporter) != Argument:
            raise TypeError("supporter must be of type Argument")

        if type(supported) != Argument:
            raise TypeError("supported must be of type Argument")

        if supporter.name in self.arguments:
            supporter = self.arguments[supporter.name]
        else:
            self.arguments[supporter.name] = supporter

        if supported.name in self.arguments:
            supported = self.arguments[supported.name]
        else:
            self.arguments[supported.name] = supported

        supported.add_supporter(supporter)

        self.supports.append(Support(supporter, supported))

    def reset_strength_values(self):
        for a in list(self.arguments.values()):
            a.strength = a.initial_weight

    def get_arguments(self):
        return list(self.arguments.values())

    def __str__(self) -> str:
        return f"BAG set to read from {self.path} with arguments: {self.arguments}, attacks: {self.attacks} and supports: {self.supports}"

    def __repr__(self) -> str:
        return f"BAG({self.path}) Arguments: {self.arguments} Attacks: {self.attacks} Supports: {self.supports}"

    def to_dict(self):
        return {
            'arguments': {
                n: a._to_shallow_dict()
                for n, a in self.arguments.items()
            },
            'attacks': [
                [a.attacker.name, a.attacked.name]
                for a in self.attacks
            ],
            'supports': [
                [s.supporter.name, s.supported.name]
                for s in self.supports
            ],
        }

    @classmethod
    def from_dict(cls, d):
        bag = cls()
        arguments = {
            n: Argument._from_shallow_dict(a)
            for n, a in d['arguments'].items()
        }
        for attacker_name, attacked_name in d['attacks']:
            bag.add_attack(arguments[attacker_name], arguments[attacked_name])
        for supporter_name, supported_name in d['supports']:
            bag.add_support(arguments[supporter_name], arguments[supported_name])
        return bag


class ArgumentMiner:
    def __init__(self, llm_manager, generate_prompt, depth=1, breadth=1, temperature=1.0):
        self.depth = depth
        self.breadth = breadth 
        self.llm_manager = llm_manager
        self.generate_prompt = generate_prompt
        self.temperature = temperature

    def generate_args_for_parent(self, parent, name, base_score_generator): 
        s_prompt, s_constraints, s_format_args = self.generate_prompt( 
            parent.get_arg(), support=True
        )
        messages=[{"role": "user", "content": s_prompt}]
        sup = s_format_args( 
            self.llm_manager.generate_one(message=messages, temperature=self.temperature),
            s_prompt,
        )
        a_prompt, a_constraints, a_format_args = self.generate_prompt(parent.get_arg())
        messages=[{"role": "user", "content": a_prompt}]
        att = a_format_args( 
            self.llm_manager.generate_one(message=messages, temperature=self.temperature),
            a_prompt,
        )
        sup_base_score = base_score_generator(sup, claim=parent.get_arg(), support=True)
        att_base_score = base_score_generator(att, claim=parent.get_arg(), support=False)
        s = Argument(f"S{name}", sup, float(sup_base_score))
        a = Argument(f"A{name}", att, float(att_base_score))
        self.argument_tree.add_support(s, parent)
        self.argument_tree.add_attack(a, parent)
        return s, a

    def generate_arguments(self, statement, base_score_generator):
        """Generates arguments for and against a statement, up to the given breadth and depth."""
        self.argument_tree = BAG()  
        topic = Argument(f"db0", statement, 0.5)
        topic_base_score = base_score_generator(statement, topic=True)
        
        previous_layer = []
        
        for d in range(1, self.depth + 1):
            new_layer = []
            
            if d == 1:
                for b in range(1, self.breadth + 1):
                    s, a = self.generate_args_for_parent(
                        parent=topic,
                        name=f"db0←d{d}b{b}",
                        base_score_generator=base_score_generator
                    )
                    
                    if s.arg != "N/A":
                        new_layer.append(s)
                    if a.arg != "N/A":
                        new_layer.append(a)
            else:
                for p in previous_layer:
                    for b in range(1, self.breadth + 1):
                        s, a = self.generate_args_for_parent(
                            parent=p,
                            name=f"{p.name}←d{d}b{b}",
                            base_score_generator=base_score_generator
                        )
                        
                        if s.arg != "N/A":
                            new_layer.append(s)
                        if a.arg != "N/A":
                            new_layer.append(a)

            previous_layer = new_layer

        topic_base_score_bag = deepcopy(self.argument_tree)
        topic_base_score_bag.arguments[topic.name].reset_initial_weight(
            topic_base_score
        )

        return self.argument_tree, topic_base_score_bag

    """If argument is similar to other arguments in same branch then we cut of that argument."""

    def cut_arguments(self, arguments):
        pass

import random

class UncertaintyEstimator:
    def __init__(self, llm_manager, generate_prompt, verbal=False, temperature=1.0):
        self.llm_manager = llm_manager
        self.generate_prompt = generate_prompt
        self.verbal = verbal 
        self.temperature = temperature

    def generate_base_score(self, statement, claim=None, support=False, topic=False):
        if statement == "N/A":
            return 0.0
        prompt, constraints, formatter = self.generate_prompt(
            statement, claim=claim, verbal=self.verbal, support=support, topic=topic
        )
        messages=[{"role": "user", "content": prompt}]
        raw_output = self.llm_manager.generate_one(message=messages, temperature=self.temperature)
        return formatter(raw_output)

    def __call__(self, statement, claim=None, support=False, topic=False):
        return self.generate_base_score(statement, claim=claim, support=support, topic=topic)

def computeTopOrder(bag):
    args = bag.arguments.values()

    #compute topological order
    indeg = {arg:0 for arg in args}

    #store indegree and parents
    attacks = {arg:[] for arg in args}
    supports = {arg:[] for arg in args}

    for att in bag.attacks:
        indeg[att.get_attacked()] += 1
        attacks[att.get_attacker()].append(att.get_attacked())
    for sup in bag.supports:
        indeg[sup.get_supported()] += 1
        supports[sup.get_supporter()].append(sup.get_supported())

    #determine source arguments
    source_args = []
    for arg in args:
        if indeg[arg] == 0:
            source_args.append(arg)

    #build up order
    order = []

    while(len(source_args) > 0):

        arg = source_args.pop(0)
        order.append(arg)

        #update children
        for c in attacks[arg]:
            indeg[c] -= 1
            if indeg[c]==0:
                source_args.append(c)

        for c in supports[arg]:
            indeg[c] -= 1
            if indeg[c]==0:
                source_args.append(c)

    #if node is missing in order, the bag must be cyclic
    if len(order) != len(args):   
        print(f"Graph contains cycles. Found partial topological order {[arg.name for arg in order]}.")
        return None
          
    return order

def computeStrengthValues(bag, agg_f, inf_f):
    
    order = computeTopOrder(bag)
    if order == None:
        return None
    
    strength = {arg:arg.initial_weight for arg in order}
    
    for arg in order:
        agg = agg_f.aggregate_strength(arg.attackers, arg.supporters, strength)
        s = inf_f.compute_strength(arg.initial_weight, agg)
        
        arg.strength = s
        strength[arg] = s
        
    return strength
    

class ArgLLMs:

    def __init__(
        self,
        model_name,
        model,
        dataset_name,
        dev_dataset,
        test_dataset,
        temperature,
        max_workers, 
        depth = 2, 
        breadth = 1,
        semantics = "dfquad",
        am_prompt = "opro",
        ue_prompt = "analyst",
        verbal = False
    ):
        self.depth = depth
        self.breadth = breadth
        self.semantics = semantics
        self.verbal = verbal
        self.am_prompt = am_prompt
        self.ue_prompt = ue_prompt
        self.model_name = model_name
        self.model = model
        self.dataset_name = dataset_name
        self.dataset = test_dataset
        self.temperature = temperature
        self.max_workers = max_workers
        self.save_path = f"{config.OUTPUT_DIR}/argllms/{dataset_name}/{model_name}.json"
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        self.show = True


    def build_system_prompt(self, item):
        return self.system_prompt.format(description=item["description"]) 

    def build_user_prompt(self, item: Dict[str, Any]):
        return f"Question: {item['question']}"

    def build_coqa_user_prompt(self, item):
        return f"Context: {item['story']}\nQuestion: {item['question']}"

    def _process_one_item(
        self,
        data: Dict[str, Any],
        model: Any,
        am_prompt_class: Any,
        ue_prompt_class: Any,
        am_prompts: List[str],
        ue_prompts: List[str],
        temperature: float,
    ) -> Dict[str, Any]:
        try:
            greedy_answer = data["greedy_results"]["answer"]
            question = data["question"]

            result = {
                "id": data["id"],
                "question": data["question"],
                "description": data["description"],
                "ground_truth": data["ground_truth"],
                "pred_answer": greedy_answer,
            }

            if self.dataset_name == "coqa":
                result["story"] = data["story"]

            if greedy_answer is None:
                return result

            claim = f"Question: {question}\nCandidate Answer: {greedy_answer}"
            if self.dataset_name == "coqa":
                claim = f"Context: {data['story']}\n" + claim

            for am_prompt in am_prompts:
                for ue_prompt in ue_prompts:
                    generate_prompt_am = getattr(am_prompt_class, am_prompt)
                    generate_prompt_ue = getattr(ue_prompt_class, ue_prompt)

                    ue = UncertaintyEstimator(
                        llm_manager=model,
                        generate_prompt=generate_prompt_ue,
                        verbal=self.verbal,
                        temperature=temperature,
                    )
                    am = ArgumentMiner(
                        llm_manager=model,
                        generate_prompt=generate_prompt_am,
                        depth=self.depth,
                        breadth=self.breadth,
                        temperature=temperature,
                    )

                    t_base, t_estimated = am.generate_arguments(claim, ue)
                    computeStrengthValues(t_base, agg_f=self._agg_f, inf_f=self._inf_f)
                    computeStrengthValues(t_estimated, agg_f=self._agg_f, inf_f=self._inf_f)

                    result[f"{am_prompt}-{ue_prompt}"] = {
                        "base": {
                            "bag": t_base.to_dict(),
                            "prediction": t_base.arguments["db0"].strength,
                        },
                        "estimated": {
                            "bag": t_estimated.to_dict(),
                            "prediction": t_estimated.arguments["db0"].strength,
                        },
                    }
            return result

        except ParseSkipSampleError as e:
            print(f"[skip] id={data.get('id')} parse failed: {e}")
            return None

    def generate(self, **kwargs):
        sample_test_path, _ = build_sample_path(self.save_path)

        if self.dataset_name == "coqa":
            build_user = self.build_coqa_user_prompt
        else:
            build_user = self.build_user_prompt

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

        with open(sample_test_path, "r", encoding="utf-8") as f:
            sample_dataset = json.load(f)

        if "answer" not in sample_dataset[0].get("greedy_results", {}).keys():
            for item in tqdm(sample_dataset):
                samples = item.get("sample_results", [])
                for s in samples:
                    if "answer" not in s.keys():
                        text = s.get("text", "")
                        s["answer"] = extract_final_answer(text)

                greedy = item.get("greedy_results", None)
                if "answer" not in greedy.keys():
                    text = greedy.get("text", "")
                    greedy["answer"] = extract_final_answer(text)

        # baseline_prompt_class = BaselinePrompts()
        am_prompt_class = ArgumentMiningPrompts()
        ue_prompt_class = UncertaintyEvaluatorPrompts()

        am_prompts = [func for func in dir(am_prompt_class) if "__" not in func]
        ue_prompts = [func for func in dir(ue_prompt_class) if "__" not in func]

        if self.semantics == "qe":
            self._agg_f = SumAggregation()
            self._inf_f = QuadraticMaximumInfluence(conservativeness=1)
        elif self.semantics == "dfquad":
            self._agg_f = ProductAggregation()
            self._inf_f = LinearInfluence(conservativeness=1)
        elif self.semantics == "eb":
            self._agg_f = SumAggregation()
            self._inf_f = EulerBasedInfluence()
        else:
            raise ValueError(f"Unknown semantics: {self.semantics}")

        if self.am_prompt != "all":
            am_prompts = [self.am_prompt]
        if self.ue_prompt != "all":
            ue_prompts = [self.ue_prompt]

        if "qwen3-4b" in self.model_name.lower():
            self.max_workers = 1

        results = [None] * len(sample_dataset)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self._process_one_item,
                    data,
                    self.model,
                    am_prompt_class,
                    ue_prompt_class,
                    am_prompts,
                    ue_prompts,
                    self.temperature,
                ): idx
                for idx, data in enumerate(sample_dataset)
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating", unit="item"):
                idx = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    print(f"[skip] future failed: {e}")
                    result = None
                results[idx] = result

        final_results = [result for result in results if result is not None]
        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(final_results, f, ensure_ascii=False, indent=4)

    def extract(self):
        with open(self.save_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        for item in tqdm(dataset):
            if f"{self.am_prompt}-{self.ue_prompt}" in item.keys():
                item["pred_score"] = item[f"{self.am_prompt}-{self.ue_prompt}"]["estimated"]["prediction"]
            else:
                item["pred_score"] = None

        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)
        
    def calculate(self):
        return
