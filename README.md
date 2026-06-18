<a name="readme-top"></a>

<!-- <div align="center">
  <img src="https://img.shields.io/badge/Repo-Black--Box--UE--Hub-2f80ed?style=for-the-badge" alt="Black-Box-UE-Hub">
  <img src="https://img.shields.io/badge/Scope-Black--Box%20UE-7b61ff?style=for-the-badge" alt="Scope">
  <img src="https://img.shields.io/badge/Methods-35-brightgreen?style=for-the-badge" alt="Methods">
  <img src="https://img.shields.io/badge/Status-Updating-orange?style=for-the-badge" alt="Status">
</div>

<br /> -->

<h1 align="center"> The Hub for Black-Box Uncertainty Estimation in LLMs</h1>

<p align="center">
  <b>A curated hub and unified evaluation framework for black-box uncertainty estimation in large language models.</b>
</p>

<p align="center">
  <img src="fig/overview.png" alt="Overview of black-box UE for LLMs" width="99%">
</p>

<p align="center">
  <b>Overview of black-box UE for LLMs.</b><br>
</p>

<!-- <p align="center">
  <a href="https://awesome.re"><img src="https://awesome.re/badge.svg" alt="Awesome"></a>
  <img src="https://img.shields.io/badge/Taxonomy-5%20Categories-blue" alt="Taxonomy">
  <img src="https://img.shields.io/badge/Tasks-Open--ended%20%7C%20Closed--ended-orange" alt="Tasks">
  <img src="https://img.shields.io/badge/External%20Code-20%20Links-green" alt="External Code Links">
</p> -->

---

## 🔍 Overview

**Black-Box-UE-Hub** collects representative methods for **black-box uncertainty estimation (UE)** in **large language models (LLMs)**. The repository is intended to serve two complementary purposes:

- 📚 **Paper index**: a taxonomy-oriented collection of black-box UE methods, including venue, relative computation cost, calibration property, task applicability, and available code resources.
- 🧩 **Evaluation framework**: a unified pipeline for running selected black-box UE methods and evaluating uncertainty scores with common metrics such as AUROC, ECE, and Brier score.

Black-box UE is especially useful when model internals such as logits, hidden states, or gradients are unavailable. Instead, it estimates reliability from externally observable behavior, including verbalized confidence, sampled responses, reasoning traces, multi-agent deliberation, or combinations of these signals.




<!-- <details>
  <summary>📑 <b>Table of Contents</b></summary>
  <ol>
    <li><a href="#-overview">Overview</a></li>
    <li><a href="#-taxonomy">Taxonomy</a></li>
    <li><a href="#-how-to-read-the-tables">How to Read the Tables</a></li>
    <li><a href="#-method-catalog">Method Catalog</a>
      <ul>
        <li><a href="#%EF%B8%8F-verbalization-based">Verbalization-based</a></li>
        <li><a href="#-sampling-based">Sampling-based</a></li>
        <li><a href="#-explanation-based">Explanation-based</a></li>
        <li><a href="#-multi-agent">Multi-agent</a></li>
        <li><a href="#-hybrid">Hybrid</a></li>
      </ul>
    </li>
    <li><a href="#-evaluation code and framework">Evaluation Code and Framework</a></li>
    <li><a href="#-quick-start">Quick Start</a></li>
    <li><a href="#-evaluation-results">Evaluation Results</a></li>
    <li><a href="#-adding-a-new-method">Adding a New Method</a></li>
    <li><a href="#-contributing">Contributing</a></li>
  </ol>
</details> -->

📑 **Table of Contents**

* [Overview](#-overview)
* [Taxonomy](#-taxonomy)
* [How to Read the Tables](#-how-to-read-the-tables)
* [Method Catalog](#-method-catalog)

  * [Verbalization-based](#%EF%B8%8F-verbalization-based)
  * [Sampling-based](#-sampling-based)
  * [Explanation-based](#-explanation-based)
  * [Multi-agent](#-multi-agent)
  * [Hybrid](#-hybrid)
* [Framework](#-framework)
* [Quick Start](#-quick-start)
* [Evaluation Results](#-evaluation-results)
* [Adding a New Method](#-adding-a-new-method)
* [Contributing](#-contributing)


---

## 🧭 Taxonomy

We organize black-box UE methods into five categories:

| Category | Main signal | Typical use case |
|---|---|---|
| 🗣️ **Verbalization-based** | The model is prompted to express its own confidence or probability distribution. | Low-cost confidence elicitation for both open-ended and closed-ended QA. |
| 🎲 **Sampling-based** | Multiple sampled outputs are compared through lexical, semantic, embedding, graph, or perturbation-based statistics. | Detecting instability, inconsistency, or semantic dispersion in generated answers. |
| 🧠 **Explanation-based** | Explanations, reasoning paths, or reasoning structures are used as uncertainty evidence. | Reasoning-heavy tasks where answer reliability depends on intermediate rationales. |
| 🤝 **Multi-agent** | Multiple agents, personas, or deliberation rounds are used to expose disagreement. | Higher-cost reliability estimation through interaction and debate. |
| 🧩 **Hybrid** | Multiple uncertainty sources or calibration mechanisms are combined. | Stronger calibration when a single signal is insufficient. |

---

## 📌 How to Read the Tables

- **Computation** indicates relative inference cost.
- **Calibration** indicates whether the method directly produces a confidence score in `[0, 1]` without requiring additional calibration or normalization.
- **Open-ended** and **Closed-ended** indicate whether the method is applicable to open-ended QA and/or closed-ended QA.
- **Code** links to publicly available implementations for each specific method.

---

## 📚 Method Catalog

### 🗣️ Verbalization-based
Verbalization-based methods estimate uncertainty by explicitly eliciting the model's own assessment of how likely its answer is to be correct. Their central idea is that uncertainty can be expressed directly in the model's output, either in numeric form or through natural-language cues. Accordingly, we divide this family into two subtypes: \emph{numeric verbalization}, which asks the model to report an explicit probability or score, and \emph{linguistic verbalization}, which infers confidence from hedging expressions or epistemic language.

| Method | Venue | Computation | Calibration | Open-ended | Closed-ended | Code |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| [TopK](https://arxiv.org/abs/2305.14975) | EMNLP 2023 | Low / Medium | ✅ | ✅ | ✅ | — |
| [CoT](https://arxiv.org/abs/2306.13063) | ICLR 2024 | Low / Medium | ✅ | ✅ | ✅ | [Code](https://github.com/MiaoXiong2320/llm-uncertainty) |
| [Verbalized Probability Distribution (VPD)](https://arxiv.org/abs/2402.00367) | arXiv 2025 | Low / Medium | ✅ | ✅ | ✅ | — |
| [Ling](https://arxiv.org/abs/2305.14975) | EMNLP 2023 | Low / Medium | ✅ | ✅ | ✅ | — |
| [Marker Confidence (MarConf)](https://arxiv.org/abs/2505.24778) | ACL 2025 | Low / Medium | ✅ | — | ✅ | [Code](https://github.com/HKUST-KnowComp/MarConf) |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

### 🎲 Sampling-based
Sampling-based methods estimate uncertainty by generating multiple responses for the same input, either directly or under controlled perturbations, and then analyzing their variation, similarity, or stability. The intuition is that if the model produces similar answers across repeated generations, its prediction is more likely to be reliable; in contrast, if the responses vary substantially, the model is likely to be more uncertain. According to whether the responses are generated from the original input or from perturbed variants of the input, these methods can be divided into two categories: \emph{direct sampling} and \emph{perturbation-based sampling}.

| Method | Venue | Computation | Calibration | Open-ended | Closed-ended | Code |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| [Semantic Entropy (SE)](https://arxiv.org/abs/2302.09664) | Nature 2024 | Medium | — | ✅ | ✅ | [Code](https://github.com/lorenzkuhn/semantic_uncertainty) |
| [SelfCheckGPT (SelfCheck)](https://arxiv.org/abs/2303.08896) | EMNLP 2023 | Medium | — | ✅ | — | [Code](https://github.com/potsawee/selfcheckgpt) |
| [Sum of Eigenvalues of the Graph Laplacian (EigV)](https://arxiv.org/abs/2305.19187) | TMLR 2024 | Medium | — | ✅ | — | [Code](https://github.com/zlin7/UQ-NLG) |
| [The Degree Matrix (Deg)](https://arxiv.org/abs/2305.19187) | TMLR 2024 | Medium | — | ✅ | — | [Code](https://github.com/zlin7/UQ-NLG) |
| [Eccentricity (Ecc)](https://arxiv.org/abs/2305.19187) | TMLR 2024 | Medium | — | ✅ | — | [Code](https://github.com/zlin7/UQ-NLG) |
| [Kernel Language Entropy (KLE)](https://arxiv.org/abs/2405.20003) | NeurIPS 2024 | Medium | — | ✅ | — | [Code](https://github.com/AlexanderVNikitin/kernel-language-entropy/tree/main/kle) |
| [Long-text Uncertainty Quantification (LUQ)](https://arxiv.org/abs/2403.20279) | EMNLP 2024 | Medium | ✅ | ✅ | — | [Code](https://github.com/caiqizh/LUQ) |
| [Jiang et al. (GU)](https://arxiv.org/abs/2410.20783) | NeurIPS 2024 | Medium | ✅ | ✅ | — | [Code](https://github.com/Mingjianjiang-1/Graph-based-Uncertainty) |
| [Semantic Embedding Uncertainty (SEU)](https://arxiv.org/abs/2410.22685) | arXiv 2024 | Medium | — | ✅ | — | — |
| [Multi-Dimensional Uncertainty Quantification (MDUQ)](https://arxiv.org/abs/2502.16820) | arXiv 2025 | Medium | — | ✅ | — | — |
| [Convex Hull](https://arxiv.org/abs/2406.19712) | DAI 2024 | High | — | ✅ | — | — |
| [Semantic INconsistency Index (SINdex)](https://arxiv.org/abs/2503.05980) | arXiv 2025 | Medium | — | ✅ | — | — |
| [Semantic Nearest Neighbor Entropy (SNNE)](https://arxiv.org/abs/2506.00245) | ACL Findings 2025 | Medium | — | ✅ | — | [Code](https://github.com/BigML-CS-UCLA/SNNE) |
| [Semantic Structural Entropy (SeSE)](https://arxiv.org/pdf/2511.16275) | arXiv 2025 | Medium | — | ✅ | — | [Code](https://github.com/SELGroup/SeSE) |
| [Sampling with Perturbation for Uncertainty Quantification (SPUQ)](https://arxiv.org/abs/2402.06164) | EACL 2024 | Medium | ✅ | ✅ | — | [Code](https://github.com/intuit-ai-research/SPUQ/tree/main) |
| [Inverse-Entropy (InvE)](https://arxiv.org/abs/2502.00628) | NeurIPS 2025 | Medium | — | ✅ | — | [Code](https://github.com/UMDataScienceLab/Uncertainty-Quantification-for-LLMs) |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

### 🧠 Explanation-based
Explanation-based methods estimate uncertainty from the model's reasoning process rather than from the final answer alone. Their core assumption is that uncertainty may be reflected in the completeness or internal consistency of the generated reasoning chain. Accordingly, these methods either analyze weaknesses within a single reasoning process or compare multiple sampled reasoning processes to quantify confidence.

| Method | Venue | Computation | Calibration | Open-ended | Closed-ended | Code |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| [CoT Explanations (COTA)](https://arxiv.org/abs/2308.16175) | AISTATS 2024 | Medium | ✅ | ✅ | ✅ | — |
| [Think twice before trusting (T3)](https://arxiv.org/abs/2403.09972) | EMNLP Findings 2024 | Medium | ✅ | — | ✅ | — |
| [Topo-UQ](https://arxiv.org/abs/2502.17026) | COLM 2025 | Medium | — | ✅ | ✅ | [Code](https://github.com/LongchaoDa/LLM-Topology) |
| [Introspective UQ (IUQ)](https://arxiv.org/abs/2506.18183) | arXiv 2025 | Low / Medium | ✅ | ✅ | ✅ | — |
| [CenConf](https://arxiv.org/abs/2509.12908) | EMNLP 2025 | Medium | ✅ | ✅ | ✅ | — |
| [PathConv](https://arxiv.org/abs/2509.12908) | EMNLP 2025 | Medium | ✅ | ✅ | ✅ | — |
| [PathWeight](https://arxiv.org/abs/2509.12908) | EMNLP 2025 | Medium | ✅ | ✅ | ✅ | — |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

### 🤝 Multi-agent
Multi-agent methods estimate uncertainty by introducing multiple agents with different roles, perspectives, or capabilities and then using their interactions as an additional source of evidence. Unlike single-model methods, which rely on one model's own generations, these approaches exploit agreement, disagreement, revision, and debate across agents to assess answer reliability. The general intuition is that an answer that remains stable under multi-agent interaction is more likely to be trustworthy.

| Method | Venue | Computation | Calibration | Open-ended | Closed-ended | Code |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| [CollabCalibration (Collab)](https://arxiv.org/abs/2404.09127) | ICLR Workshop 2024 | High | ✅ | ✅ | ✅ | [Code](https://github.com/minnesotanlp/collaborative-calibration) |
| [ArgLLMs](https://arxiv.org/abs/2405.02079) | AAAI 2025 | Medium | ✅ | ✅ | ✅ | [Code](https://github.com/CLArg-group/argumentative-llms) |
| [DiverseAgentEntropy (DAE)](https://arxiv.org/abs/2412.09572) | EMNLP Findings 2025 | High | — | ✅ | ✅ | [Code](https://github.com/amazon-science/DiverseAgentEntropy) |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

### 🧩 Hybrid
Hybrid methods are motivated by the observation that no single uncertainty signal is sufficient to capture model reliability robustly across settings. Different signals encode different aspects of uncertainty: consistency reflects output stability, verbalization reflects self-assessment, and candidate comparison or explanation reflects relative preference or reasoning support. Hybrid methods therefore combine multiple complementary signals in order to produce uncertainty estimates that are more robust than those obtained from any single source alone.

| Method | Venue | Computation | Calibration | Open-ended | Closed-ended | Code |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| [BSDetector](https://arxiv.org/abs/2308.16175) | ACL 2024 | Medium | ✅ | ✅ | — | — |
| [UF Calibration (UF)](https://arxiv.org/abs/2404.02655) | EMNLP 2024 | Medium | ✅ | — | ✅ | — |
| [SteerConf](https://arxiv.org/abs/2503.02863) | NeurIPS 2025 | Medium | ✅ | ✅ | ✅ | [Code](https://github.com/scottjiao/SteerConf) |
| [Distractor-Normalized Coherence (DiNCo)](https://arxiv.org/abs/2509.25532) | ICLR 2026 | Medium | ✅ | ✅ | — | [Code](https://github.com/victorwang37/dinco) |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## 🧱 Evaluation Code and Framework

The code framework is organized around a batch pipeline and a shared method interface. It includes implementations of the major black-box UE methods currently supported in this repository, and new methods can be added progressively under the same interface.

### Core Directory Layout

```text
Black-Box-UE-Hub/
├── run_pipeline.py                  # batch runner for generation, extraction, and UE scoring
├── judge.py                         # answer correctness judging
├── eval.py                          # AUROC, ECE, Brier, and other evaluation metrics
├── run_pipeline.methods.example.yaml # example method configuration
├── dataset/                         # local datasets or processed data files
├── output/                          # method outputs and scored JSON files
├── sample/                          # sampled responses and intermediate caches
└── src/
    ├── config.py                    # project paths and model settings
    ├── dataset/                     # dataset loaders and preprocessing
    ├── method/                      # black-box UE method implementations
    ├── model/                       # local/API model adapters
    └── utils.py                     # shared parsing, sampling, similarity, and evaluation helpers
```

### Method Interface

Each runnable method is expected to follow the same three-stage interface:

```python
class ExampleMethod:
    def generate(self, ...):
        """Generate model outputs, sampled responses, or intermediate caches."""

    def extract(self, ...):
        """Parse raw outputs into method-specific fields."""

    def calculate(self, ...):
        """Compute uncertainty scores and save them to the output file."""
```

---

## 🚀 Quick Start

### 1. Configure Paths and Models

Update project paths, model names, API settings, and output directories in:

```text
src/config.py
```

Adjust method-specific arguments in:

```text
run_pipeline.methods.example.yaml
```

### 2. Run the Pipeline

```bash
python run_pipeline.py \
  --models qwen3-4b-instruct qwen3-30b-instruct \
  --datasets triviaqa coqa hotpotqa truthfulqa_mc \
  --methods cot topk \
  --steps generate extract calculate
```

### 3. Judge Answer Correctness

```bash
python judge.py \
  --methods cot topk \
  --datasets triviaqa hotpotqa coqa truthfulqa_mc \
  --models qwen3-4b-instruct qwen3-30b-instruct deepseek-v3.2 gpt-5-mini
```

### 4. Evaluate Uncertainty Scores

```bash
python eval.py \
  --methods cot topk \
  --datasets triviaqa hotpotqa coqa truthfulqa_mc \
  --models qwen3-4b-instruct qwen3-30b-instruct deepseek-v3.2 gpt-5-mini
```

---

## 📊 Evaluation Results

We report aggregate evaluation results of representative black-box UE methods on open-ended and closed-ended QA tasks. Bar charts summarize the overall comparison across AUROC, ECE, and Brier score, while the table images provide more detailed numerical results for different method groups.

### Open-ended QA
In open-ended QA, sampling-aggregated verbalization methods and hybrid methods are highly competitive. In particular, SteerConf shows strong overall discriminative performance, while sampling-aggregated VPD provides strong calibration. DiNCo also performs well by combining cross-sample consistency with candidate-level comparison. Among sampling-based methods, NLI-relation-based methods are generally stronger on AUROC than methods relying mainly on continuous embedding similarity, suggesting that fine-grained semantic relations are important for open-ended factual reliability estimation.

<p align="center">
  <img src="fig/open-ended.png" alt="Comparison of UE methods on open-ended QA" width="95%">
</p>

<p align="center">
  <b>Figure: Overall comparison on open-ended QA.</b><br>
</p>

#### Verbalization-based methods

Sampling aggregation generally improves single-pass verbalization by making self-reported confidence more stable across repeated elicitation. VPD is especially competitive because it encourages the model to compare multiple candidate answers and assign confidence over the answer space, while purely linguistic uncertainty expressions are harder to map into reliable numerical scores.

<p align="center">
  <img src="fig/open-ended-verbalization.png" alt="Open-ended QA results for verbalization-based methods" width="70%">
</p>

<p align="center">
  <b>Table: Open-ended QA results for verbalization-based methods.</b><br>
</p>

#### Sampling-based methods
Sampling-based methods isolate the quality of uncertainty signals by comparing multiple generated responses under the same answer-generation setting. NLI-based methods such as SelfCheck, Deg, KLE, and SNNE show strong and stable AUROC, because entailment and contradiction relations are often more aligned with factual correctness than broad embedding similarity.

<p align="center">
  <img src="fig/open-ended-sampling.png" alt="Open-ended QA results for sampling-based methods" width="85%">
</p>

<p align="center">
  <b>Table: Open-ended QA results for sampling-based methods.</b><br>
</p>

#### Other methods
This group includes explanation-based, multi-agent, and hybrid methods. Hybrid methods such as SteerConf and DiNCo perform well because they combine complementary signals, including cross-sample consistency, candidate comparison, and confidence stability under controlled prompting. Explanation-based and multi-agent methods can provide additional evidence from reasoning chains or deliberation processes, but their effectiveness is more sensitive to the quality of generated reasoning and the design of interaction mechanisms. In open-ended QA, different reasoning chains may vary substantially in their local steps even when they support the same answer, and multi-agent deliberation may sometimes reinforce noisy or majority-biased signals rather than improve uncertainty estimation.

<p align="center">
  <img src="fig/open-ended-other.png" alt="Open-ended QA results for other black-box UE methods" width="70%">
</p>
<p align="center">
  <b>Table: Open-ended QA results for explanation-based, multi-agent, and hybrid methods.</b><br>
</p>

### Closed-ended QA
In closed-ended QA, the answer space is predefined, so methods that explicitly exploit candidate options show clear advantages. T3 and UF achieve strong discriminative performance by reasoning over candidate explanations or decomposing confidence into question-level uncertainty and answer fidelity. Among more general methods, sampling-aggregated verbalization remains competitive, and VPD shows strong calibration because it directly elicits a normalized confidence distribution over candidate answers.

<p align="center">
  <img src="fig/closed-ended.png" alt="Comparison of UE methods on closed-ended QA" width="95%">
</p>
<p align="center">
  <b>Figure: Overall comparison on closed-ended QA.</b><br>
</p>

<p align="center">
  <img src="fig/closed-ended-all.png" alt="Closed-ended QA results for all methods" width="95%">
</p>
<p align="center">
  <b>Table: Closed-ended QA results for all applicable methods.</b><br>
</p>

---

## 🧩 Adding a New Method

To add a new black-box UE method:

1. Create a new method file under `src/method/`.
2. Implement the `generate`, `extract`, and `calculate` stages.
3. Add method-specific arguments to `run_pipeline.methods.example.yaml`.
4. Run the method through `run_pipeline.py`.
5. Evaluate the saved uncertainty scores with `eval.py`.

---

## 🤗 Contributing

Contributions are welcome. Please open an Issue or Pull Request if you find:

- missing or newly released black-box UE papers,
- broken paper or code links,
- incorrect venue, taxonomy, or applicability information,
- missing implementation details,
- bugs in the evaluation pipeline.



<!-- ## 📜 Citation

If you find this repository useful, please cite the corresponding survey or this repository. The BibTeX entry can be updated once the final survey metadata is available.

```bibtex
@misc{blackboxuehub2026,
  title        = {Black-Box-UE-Hub: A Hub for Black-Box Uncertainty Estimation in Large Language Models},
  howpublished = {\url{https://github.com/<YOUR_GITHUB_ID>/Black-Box-UE-Hub}},
  year         = {2026}
}
```

<p align="right">(<a href="#readme-top">back to top</a>)</p> -->
