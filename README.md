# Multilingual Topic Classification & Headline Generation for Nigerian Languages
### DSN AI Bootcamp Hackathon 2026 — LLM / Agent Track

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![HuggingFace Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-yellow)](https://huggingface.co/docs/transformers/index)
[![PEFT LoRA](https://img.shields.io/badge/PEFT-LoRA-green.svg)](https://github.com/huggingface/peft)
[![Kaggle Leaderboard](https://img.shields.io/badge/Kaggle%20LB-0.57105%20(Rank%20%2339)-orange.svg)](https://www.kaggle.com/competitions/dsn-bootcamp-hackathon-2026-llm-agent-track)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

---

## 📌 Executive Summary

This repository contains the end-to-end competition pipeline and final submission for the **Data Science Nigeria (DSN) AI Bootcamp 2026 (LLM / Agent Track)**.

The task challenges participants to develop a **strictly sub-1B parameter LLM** capable of dual-task execution across four underserved Nigerian languages—**Hausa (`hau`)**, **Igbo (`ibo`)**, **Nigerian Pidgin (`pcm`)**, and **Yoruba (`yor`)**:
1. **Task A: News Topic Classification** — Categorize articles into 7 discrete topics evaluated via pooled **Macro-F1**.
2. **Task B: News Headline Generation** — Generate coherent, culturally aligned headlines in the source language evaluated via **GenScore** (blended ROUGE-L + BERTScore).

The final competition metric is the arithmetic mean:
$$\text{Merged Score} = 0.5 \times \text{Macro-F1 (Task A)} + 0.5 \times \text{GenScore (Task B)}$$

> 💡 **Detailed Engineering Log**: For an exhaustive, version-by-version post-mortem documenting every architectural pivot, bug fix, and empirical failure from Version 1 to Version 12, see [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md).

---

## 🏆 Results & Benchmark Scoreboard

### Public Leaderboard Standing
* **Public Score**: **`0.57105`**
* **Leaderboard Rank**: **#39** out of 74 competing teams.
* **Competition Platform**: Kaggle ([`jezrealmomoh`](https://www.kaggle.com/code/jezrealmomoh/notebookdc2758c08b))

### Dev Set Performance Breakdown ($N = 869$)

| Language | Task A: Macro-F1 | Task B: GenScore (ROUGE-L) | Language Merged Score |
| :--- | :---: | :---: | :---: |
| **Hausa (`hau`)** | `0.8193` | `0.2813` | **`0.5503`** |
| **Igbo (`ibo`)** | `0.6610` | `0.2019` | **`0.4315`** |
| **Nigerian Pidgin (`pcm`)** | `0.6321` | `0.2741` | **`0.4531`** |
| **Yoruba (`yor`)** | `0.5811` | `0.2814` | **`0.4313`** |
| **Pooled Overall** | **`0.7948`** | **`0.2623`** | **`0.5285`** |

*Note: Zero-shot baseline merged score was `0.3812`; LoRA adaptation and dual-key stratified rebalancing produced a **+14.73 point lift** on the dev benchmark.*

---

## ⚙️ Architecture & Engineering Design

```
Raw Article Text (Hausa / Igbo / Pidgin / Yoruba)
   │
   ├──▶ [Task A: Topic Classifier Prompt] ──▶ Qwen2.5-0.5B + LoRA ──▶ Predicted Topic
   │                                                                         │
   └───────────────┬─────────────────────────────────────────────────────────┘
                   ▼
       [Task B: Conditioned Headline Prompt] ──▶ Qwen2.5-0.5B + LoRA ──▶ Generated Headline
```

### 1. Model Backbone
* **Selected Model**: `Qwen/Qwen2.5-0.5B-Instruct`
* **Total Parameter Count**: **`494,032,768`** parameters (~0.494B).
* Strictly satisfies the hard competition constraint ($< 1.0\text{B}$ parameters).

### 2. Parameter-Efficient Fine-Tuning (PEFT / LoRA)
* **Target Projections**: All 7 attention & MLP linear projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
* **Hyperparameters**: $r = 32$, $\alpha = 64$, $\text{dropout} = 0.05$.
* **Training Budget**: 2.0 epochs, Cosine learning rate decay with warmup (`lr = 1.8e-4`), mixed precision `fp16`, effective batch size 16.
* **Hardware & Runtime**: 1x NVIDIA Tesla T4 GPU on Kaggle, total runtime **~77 minutes**.

### 3. Key Engineering Decisions
* **Sequential Multi-Task Conditioning**: Rather than forcing single-pass joint generation (which causes attention degradation and severe label leakage on sub-1B models), the pipeline predicts the topic first, then sequentially feeds that predicted topic as context into headline generation.
* **Dual-Key Stratified Class Rebalancing**: Solved minority class starvation across `(language, topic)` pairs (e.g. boosting under-represented categories in Igbo and Yoruba to match Hausa distributions).
* **Generation Envelope Calibration**: Implemented a constrained decoding window (`max_new_tokens=48`, `min_new_tokens=8`) with soft repetition penalty ($1.15$), preventing diacritic degradation in Yoruba/Igbo while avoiding rigid n-gram bans that break multi-byte African morphology.

---

## 📂 Repository Structure

```text
├── .gitignore
├── pyproject.toml                              # Dependency definitions and build tooling
├── README.md                                   # Technical overview and benchmarks
├── EXPERIMENT_LOG.md                           # Comprehensive v1-to-v12 iteration notes
├── LICENSE                                     # MIT License
├── build_dev_set.py                            # Canonical validation split generator
├── validate_submission.py                     # Strict assertion validator for submission.csv
├── multilingual_topic_headline_dsn2026.ipynb   # Standalone self-contained Kaggle notebook
├── src/                                        # Production modules
│   ├── __init__.py
│   └── data/
│       ├── __init__.py
│       ├── loader.py                           # Robust streaming data loader
│       └── schema.py                           # Pydantic schema validation
├── tests/                                      # Comprehensive test suite (15/15 passing)
│   ├── __init__.py
│   ├── test_data_loader.py                     # Data invariants & schema tests
│   └── test_phase3_logic.py                    # Prompting, parsing & rebalancing tests
├── docs/                                       # Official competition briefs
│   └── DSN Bootcamp_LLM Track Project Brief  (2).pdf
└── kaggle_kernel/                              # Kaggle deployment configuration
    ├── kernel-metadata.json
    └── notebookdc2758c08b.ipynb
```

---

## 🚀 Quickstart & Setup

### 1. Clone & Install Environment
```bash
git clone https://github.com/jezreal-dev/dsn-2026-multilingual-llm.git
cd dsn-2026-multilingual-llm

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"
```

### 2. Run Test Suite
Verify that all 15 unit and integration tests pass:
```bash
pytest tests/ -v
```

### 3. Validate Submission File
To guarantee zero-null, exactly 3,486-row formatting before Kaggle upload:
```bash
python validate_submission.py path/to/submission.csv
```

### 4. Running on Kaggle
1. Upload `multilingual_topic_headline_dsn2026.ipynb` to Kaggle.
2. Under Notebook Options:
   * **Accelerator**: GPU T4 (or GPU T4 x2)
   * **Internet**: ON
3. Click **Run All**. The notebook will fine-tune the LoRA adapter, score the dev benchmark, and output `submission.csv`.

---

## 📜 Findings & Linguistic Insights

1. **Linguistic Asymmetry**: Hausa achieved the highest Macro-F1 (`0.8193`) due to substantial lexical overlap with international news vocabulary, whereas Yoruba presented the greatest topic ambiguity (`0.5811`) due to extensive domain blending between local cultural events and governance.
2. **Generative Orthography**: Task B was significantly harder than Task A across all languages due to tone-marking orthography in Yoruba and Igbo. Calibrating the repetition penalty to soft penalization ($1.15$) prevented character loop collapses without penalizing legitimate reduplicative vocabulary.
3. **Data Rebalancing Efficacy**: Rebalancing the training distribution provided a +2.7 point boost to Igbo Macro-F1 alone, confirming that sub-1B models are highly vulnerable to language-slice starvation.

---

## 📄 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 🤝 Acknowledgments
* **Data Science Nigeria (DSN)** for organizing the AI Bootcamp 2026 Hackathon.
* **MasakhaNEWS** (Adelani et al., 2023) for benchmark datasets in African languages.
