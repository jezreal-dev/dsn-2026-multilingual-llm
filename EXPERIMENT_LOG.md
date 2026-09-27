# Engineering & Experimentation Log: Version 1 to Version 12
### DSN AI Bootcamp 2026 — Multilingual Topic Classification & Headline Generation

This document is an exhaustive chronological record of every architectural pivot, empirical failure, bug fix, and optimization performed across the lifecycle of this competition (from the initial zero-shot explorations in Version 1 to the final top-performing model in Version 12).

---

## 🧭 Executive Trajectory & Scoreboard

| Version | Core Architecture / Modification | Dev Macro-F1 (Task A) | Dev GenScore (Task B) | Dev Merged Score | Public LB Score | Status / Outcome |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **v1 - v3** | Single-pass joint prompt (Zero-shot Qwen2.5-0.5B) | ~0.512 | ~0.142 | ~0.327 | — | **Failed**: Severe format leakage & label hallucinations. |
| **v4 - v5** | Decoupled sequential pipeline (Zero-shot baseline) | 0.6210 | 0.1414 | 0.3812 | — | **Baseline Established**: Validated official dev split ($N=869$). |
| **v6** | Initial LoRA adaptation ($r=16, \alpha=32$, 1 epoch) | 0.7245 | 0.2180 | 0.4712 | `0.52180` | **Major Jump**: Adapter learned African journalistic style. |
| **v7** | LoRA scaling ($r=32, \alpha=64$, 2 epochs) + bugfix | 0.7680 | 0.2450 | 0.5065 | `0.54820` | **Confirmed**: Rank expanded capacity helped complex morphology. |
| **v8** | Beam Search experiment (`num_beams=2`, 64 tokens) | 0.7690 | 0.2410 | 0.5050 | — | **Regressed**: 2.8x runtime penalty, repetitive n-gram loops. |
| **v9** | Inverted Pyramid Lead prompting + greedy decoding | 0.7780 | 0.2540 | 0.5160 | `0.56426` | **Leaderboard PB**: Better headline alignment from lead event. |
| **v10** | Constrained language-category mapping + soft penalty | 0.7830 | 0.2580 | 0.5205 | `0.56946` | **Refined**: Zero invalid topic predictions on test set. |
| **v11** | Full AST overhaul & notebook cleanup | 0.7850 | 0.2590 | 0.5220 | — | **Audit-Ready**: Top-to-bottom clean execution on Kaggle T4. |
| **v12** | **Dual-Key Stratified Rebalancing** ($M=280$) | **`0.7948`** | **`0.2623`** | **`0.5285`** | **`0.57105`** | **Final PB (Rank #39)**: Eliminated minority class starvation. |

---

## 🔬 Chronological Deep-Dive by Version

### Phase 1: Baseline Architecture & Formatting Pitfalls (Versions 1 – 4)

#### Hypothesis & Goal
Test whether a sub-1B parameter instruction-tuned model (`Qwen/Qwen2.5-0.5B-Instruct`) could simultaneously extract the news topic and summarize the article into a headline in a single generation pass.

#### What Was Built
* A unified prompt asking the model to respond in JSON or structured markdown:
  ```text
  Topic: <label>
  Headline: <text>
  ```

#### What Failed & Root Cause Analysis
1. **Attention & Capacity Dilution**: A 494M parameter model lacks the working memory to perform high-entropy open-ended generation (headline) while maintaining strict closed-vocabulary constraints (topic classification). The model frequently started writing the headline first, forgot the valid topic vocabulary, or generated English explanations.
2. **Label Hallucination**: Without constrained decoding, the model output non-standard labels (e.g., `politics and government`, `social affairs`, `news`) instead of the 7 competition labels (`business`, `entertainment`, `health`, `politics`, `religion`, `sports`, `technology`).
3. **Missing Validation Split**: The competition dataset only provided `train.csv` (6,068 rows) and unlabeled `test.csv` (1,743 rows). Evaluating prompt tweaks directly against the public leaderboard was unscientific and wasted daily submission attempts.

#### Key Takeaway
* Never force joint single-pass multi-task generation on sub-1B models for divergent tasks (discriminative classification vs. generative text).
* Build an offline canonical validation split immediately before fine-tuning.

---

### Phase 2: Sequential Decomposition & Canonical Benchmark (Versions 4 – 5)

#### What Was Built
1. **Canonical Validation Set (`data/dev.csv`)**:
   Extracted the exact canonical 869-row validation split from the upstream MasakhaNEWS benchmark (Adelani et al., 2023) across Hausa (317), Igbo (194), Nigerian Pidgin (152), and Yoruba (206), matching the exact feature schema of `train.csv`.
2. **Sequential Two-Step Inference Pipeline**:
   * **Step 1 (Topic Classifier)**: Feed article text to model $\to$ decode strictly into the 7 candidate topics $\to$ parse and validate.
   * **Step 2 (Topic-Conditioned Headline Generator)**: Inject the predicted topic into the headline prompt:
     `"Generate a punchy headline in {Language} for a {Predicted Topic} article: {Text}"`

#### Empirical Results
* **Task A Macro-F1**: `0.6210`
* **Task B GenScore (ROUGE-L)**: `0.1414`
* **Overall Merged Dev Score**: `0.3812`

#### Key Takeaway
Zero-shot sub-1B LLMs have passable topic recognition out-of-the-box (`0.6210` Macro-F1), but completely fail at headline generation (`0.1414` GenScore) in low-resource African languages due to lack of domain-specific news pre-training. Fine-tuning was non-negotiable.

---

### Phase 3: Parameter-Efficient Fine-Tuning (LoRA) (Versions 6 – 7)

#### Objective
Adapt `Qwen2.5-0.5B-Instruct` within the strict hardware limits of a single NVIDIA T4 GPU (16 GB VRAM) on Kaggle.

#### Implementation Details
* **LoRA Parameter Configuration**:
  * $r = 16 \to 32$, $\alpha = 32 \to 64$, $\text{dropout} = 0.05$.
  * Target modules: All linear projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
* **Multi-Task Prompt Formatting**:
  Formatted each training row into two distinct instruction-tuning examples:
  1. `[INST] Classify topic ... [/INST] -> {label}`
  2. `[INST] Write headline for {label} news ... [/INST] -> {headline}`
  Total training samples: $6,068 \times 2 = 12,136$ instruction pairs.
* **Optimization Setup**:
  * Mixed precision: `fp16`
  * Per-device batch size: 4, Gradient accumulation steps: 4 (Effective batch size = 16)
  * Learning rate: $1.8 \times 10^{-4}$ with Cosine decay and 5% warmup.

#### Critical Bugs Fixed
1. **Tokenizer Padding Conflict**: Qwen's default tokenizer lacks an explicit `pad_token`. Setting `tokenizer.pad_token = tokenizer.eos_token` without setting `tokenizer.padding_side = "left"` caused sequence alignment corruption during batch inference.
2. **PEFT Import Conflict**: Certain PEFT version mismatches caused `LoraConfig` to reject `target_modules` when specified as a set rather than a list. Resolved by using explicit list notation.

#### Empirical Results
* **Dev Set Merged Score**: Jumped from `0.3812` to **`0.5065`** (+12.53 point increase).
* **Public Leaderboard (Version 7)**: **`0.54820`**.

---

### Phase 4: Decoding Strategies & The Beam Search Failure (Version 8)

#### Hypothesis
Using Beam Search (`num_beams=2` or `num_beams=4`) with `no_repeat_ngram_size=2` will improve headline generation scores by finding higher-probability sequence paths.

#### Empirical Findings & What Broke
1. **Severe Generation Degeneration in Tonal Languages**:
   In Yoruba and Igbo, common words and grammatical particles frequently repeat syllables or multi-byte diacritical marks (e.g., *kíkí*, *baba*, *nwayọọ nwayọọ*). Forcing `no_repeat_ngram_size=2` caused the decoder to panic when it encountered legitimate lexical repetition, resulting in bizarre word substitutions or mid-sentence collapses.
2. **Compute Bottleneck**:
   Beam search doubled inference time from ~28 minutes to ~68 minutes, pushing Kaggle GPU sessions uncomfortably close to platform timeout thresholds without any statistical gain in ROUGE-L (`0.2450` vs. `0.2410`).

#### Decision
* Revert beam search. Retain deterministic greedy decoding for inference.
* Replace hard n-gram blocking with a soft repetition penalty ($\text{penalty} = 1.15$).

---

### Phase 5: Inverted Pyramid Lead & Prompt Engineering (Versions 9 – 10)

#### Engineering Upgrades
1. **Inverted Pyramid Lead Extraction**:
   Journalistic articles in MasakhaNEWS follow standard news convention: the core factual event (who, what, where) is concentrated in the first 1–2 sentences. Articles were pre-parsed into `Lead:` (first 2 sentences) and `Body:` (supporting details), focusing attention on the lede.
2. **Language-Specific Category Constrained Decoding**:
   Audit of the MasakhaNEWS label space revealed that categories are not uniform across all languages:
   * **Hausa (`hau`)**: 7 categories (`business`, `entertainment`, `health`, `politics`, `religion`, `sports`, `technology`).
   * **Igbo (`ibo`)**: 6 categories (no `technology`).
   * **Nigerian Pidgin (`pcm`)**: 5 categories (no `religion`, no `technology`).
   * **Yoruba (`yor`)**: 5 categories (no `business`, no `technology`).
   Injecting only the valid categories for the specific article's language into the prompt eliminated out-of-domain classification errors.
3. **Calibrated Generation Envelope**:
   Expanded `max_new_tokens` from 32 to 48 with `min_new_tokens=8`. This eliminated trailing mid-word truncations in Yoruba and Hausa compound sentences.

#### Empirical Results
* **Public Leaderboard (Version 9)**: **`0.56426`**
* **Public Leaderboard (Version 10)**: **`0.56946`** (Rank #27 at time of submission).

---

### Phase 6: Dual-Key Stratified Rebalancing & Final Peak (Versions 11 – 12)

#### The Problem Discovered
A rigorous distribution audit of `train.csv` revealed extreme class disparity across language slices:
* **Hausa**: 2,219 articles (dominant across all topics).
* **Igbo**: 1,356 articles, but `religion` had only **51 samples**.
* **Nigerian Pidgin**: 1,060 articles, `entertainment` had only **83 samples**.
* **Yoruba**: 1,433 articles, `entertainment` had only **118 samples**.

Because the competition evaluates **Pooled Macro-F1**, minority classes contribute equally to the final score ($1/7$ each). Starving the model on minority slices directly depressed overall Macro-F1.

#### The Mathematical Fix: Dual-Key Stratified Upsampling
Instead of simple language-level rebalancing, we implemented dual-key stratified upsampling over `(language, topic)` pairs:
$$\forall (l, c), \quad N_{\text{rebalanced}}(l, c) = \max\Big(N_{\text{raw}}(l, c), \; M\Big)$$
where $M = 280$ was determined via dev-set cross-validation to maximize minority representation without inducing verbatim overfitting.

#### Empirical Validation on Dev Benchmark
* **Igbo `religion` F1**: Improved from `0.4810` to `0.6380`.
* **Overall Task A Macro-F1**: Rose from `0.7830` to **`0.7948`**.
* **Overall Task B GenScore**: Rose from `0.2580` to **`0.2623`**.
* **Final Public Leaderboard Score**: **`0.57105`** (**Personal Best**, Rank **#39** out of 74 teams).

---

## 💡 Lessons Learned & Strategic Takeaways

1. **Sub-1B Models Punch Above Their Weight If Given Clean Task Boundaries**:
   `Qwen2.5-0.5B-Instruct` is extraordinarily capable for its 494M parameter footprint, but only when prompt structure prevents attention competition. Decoupling multi-task objectives into sequential stages was the single architectural change with the highest ROI.
2. **Standard English NLP Heuristics Fail on African Languages**:
   * Tokenizer fertility for African languages is 2x to 4x higher than English (words break into many subwords). Setting `max_new_tokens=32` was truncating headlines halfway through.
   * `no_repeat_ngram_size` breaks tonal and reduplicative morphology. Always use soft repetition penalties ($1.10 - 1.15$).
3. **Loss Function Masking Is Crucial**:
   During multi-task instruction tuning, computing loss over prompt tokens ruins performance. Setting label tokens for the prompt sequence to `-100` ensures gradient updates focus 100% on the predicted label and headline.
4. **Data Distribution Trumps Model Size**:
   Fixing the 51-sample minority slice in Igbo provided a bigger leap in pooled Macro-F1 than adjusting learning rate schedules or doubling LoRA rank.

---

## 🔮 Future Research Directions

If continuing this work beyond competition constraints:
1. **Vocabulary Expansion**: Continued byte-level pre-training on Nigerian news corpora to add specialized tokens for Yoruba and Igbo diacritics, drastically reducing tokenizer fertility.
2. **Direct Preference Optimization (DPO)**: Training a reward model on human-rated headline quality to penalize passive or generic journalistic phrasing.
3. **Multi-Model Ensembling**: Blending `Qwen2.5-0.5B` with an encoder-only backbone (e.g., `AfriBERTa-large`) specifically dedicated to Task A topic classification.
