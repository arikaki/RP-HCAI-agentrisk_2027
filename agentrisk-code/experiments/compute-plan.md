# What Can Be Computed Now

*M13 project · written 27 Nov 2026 · for running on your own machine*

The training and dev data are public, Codabench is open for all four subtasks, and official dev-set baselines exist to compare against. So almost every experiment in the proposal can start now. Only the **test set** (released 10 Jan) and the **user study** have to wait.

**Useful facts for planning compute** (from the first 100 training items):

- **Trajectory length:** median about 3,300 characters (roughly 800–900 tokens), longest about 12,000 characters (roughly 3,000 tokens). Every item fits easily in any modern LLM's context window, so no truncation is needed.
- **Data size:** 568 training and 143 dev items. Running an LLM over the whole dev set is a small job even for large models. The cost lies in repeated runs (prompt variants, several samples per item), not in data size.

---

## Your hardware: what fits

**Machine:** Intel i9-13900HX · RTX 4080 Laptop GPU, **12 GB VRAM** · 32 GB RAM.

That's a strong laptop, but **12 GB of VRAM is the limit that matters**. It runs small and mid-size models well, not the largest official baselines. Memory figures below are rough estimates: weights only, plus a little for context.

| Task | Fits on your GPU? | How |
|---|---|---|
| B0, metrics, data analysis | ✅ | CPU only |
| **B1 / S1 with a ~7–9B model** (e.g., Qwen3.5-9B, an official baseline) | ✅ | 4-bit quantised (about 5–6 GB) via **vLLM** (AWQ/GPTQ) or **Ollama / llama.cpp** (GGUF). 8-bit (about 9–10 GB) is tight but may fit with short context. |
| **B1 / S1 with the 27B model** (Qwen3.8-27B-FP8, the best open baseline) | ⚠️ Not fully | Even 4-bit needs about 15–17 GB. Possible with llama.cpp **CPU offload** (32 GB RAM helps), but much slower. Acceptable for one overnight run on 143 dev items, not for many prompt variants. |
| **B1 / S1 with the 122B model** (Qwen3.5-122B-A10B) | ❌ | Needs roughly 60+ GB even at 4-bit. Use external compute (below). |
| **S2a: encoder fine-tune** | ✅ | Use **ModernBERT-large** rather than DeBERTa: it reads up to 8,192 tokens, while DeBERTa is usually limited to 512, which would cut off most trajectories (median about 850 tokens). bf16, batch size 4–8, gradient checkpointing. |
| **S2b: QLoRA fine-tune of a ~7–9B LLM** | ✅ (tight) | 4-bit QLoRA, batch size 1 with gradient accumulation, max length about 2,048 tokens, gradient checkpointing. Libraries like Unsloth reduce memory further. |
| S2b on 27B or larger | ❌ | External compute only |
| Sampling 5–10× per item | ✅ with a 7–9B model | Batched in vLLM; mostly a matter of time |

### Recommended setup

- **Main model on your laptop:** a ~7–9B instruction model at 4-bit, ideally **Qwen3.5-9B**. Its official dev score (MAE 0.608) lets you check your pipeline against a published number.
- **Second, stronger model** for the final B1 vs. S1 comparison: run the 27B model once overnight with CPU offload, or use external compute.
- **Inference engine:** vLLM for fast batched runs if the quantised model fits fully on the GPU; llama.cpp or Ollama if you need CPU offload.
- **Laptop care:** long GPU runs get hot. Use the performance power profile, keep the laptop plugged in and well ventilated, and save outputs after every item, so a crash doesn't lose a run.

### External compute for the larger models (free or cheap options)

| Option | What you get | Notes |
|---|---|---|
| **University GPU cluster or HPC** | Larger GPUs, often free for students | **Ask Prof. Kruschwitz or the university computing centre** whether M13 students can get access |
| **Kaggle notebooks** | Free GPU time each week | A free account is enough; good for a few longer runs. Quotas change, so check current limits |
| **Google Colab** | Free tier with modest GPUs; paid tiers with larger ones | Free sessions time out; fine for short runs |
| **Hosted model APIs** (e.g., OpenRouter or similar) | Large open models such as 27B–122B, paid per token | 143 dev items × a few variants is a small job; check prices first. Log the provider and model version for reproducibility |

**Bottom line:** almost everything in this plan runs on your laptop with a 7–9B model. Only reproducing the official 27B and 122B baselines, and fine-tuning anything larger than about 9B, needs outside compute.

---

## Recommended order

Each step builds on the one before. The first two are mainly about checking that the pipeline is correct.

### 1. Data preparation and exploration · CPU only

- Download the train and dev splits from `AgentRisk/agentrisk-data-2027` on Hugging Face.
- **Check the dev split's columns:** the official Subtask 3 baseline is scored on dev, so dev may now include intervention labels. The train items I looked at had none.
  - *Checked 27 Nov:* dev has the same columns as train (`example_id`, `formatted_interaction`, `proposed_action`, `risk_severity`, `risk_domains`, `risk_drivers`), so **no intervention labels** in the public dev split.
- Label distributions for severity, harm domains and risk drivers, on train and dev.
- **Tag DevOps-like items by hand or with an LLM.** A simple keyword search (paths, `ssh`, `kubectl`, "server" and so on) flagged about half of the first 100 items, which is far too broad to be useful.

### 2. B0: majority baseline · CPU only

- Predict the most common label for everything.
- **Check:** your dev scores should match the official majority baseline (Subtask 1 MAE **1.294**, 2a and 2b Jaccard **0.455**). If they match, your data loading and metrics are right.
- **Make a first Codabench submission with it** to test the submission format end to end.

### 3. B1: holistic LLM judge · GPU

- One prompt: trajectory plus the official label definitions, asking for severity, harm domains and risk drivers as JSON.
- **Use the same open-weight models as the official baselines first**, so you can check you reproduce their numbers:

  | Official baseline | Dev MAE (ST 1) |
  |---|---|
  | Qwen3.5-9B | 0.608 |
  | Qwen3.5-122B-A10B | 0.392 |
  | Qwen3.8-27B-FP8 | 0.238 |

  The organisers' exact prompt isn't published on the task page, so expect your numbers to differ somewhat. The aim is to land in the same range.
- **Tooling:** a batched inference server such as vLLM makes 143 items × several variants fast. Use temperature 0 for the main runs.

### 4. S1: factor-based LLM judge (your main method) · GPU

- **Step 1:** the model rates the seven factors from the task guidelines, each with a one-sentence reason, as JSON.
- **Step 2:** severity, domains and drivers are derived from those ratings, either by the same model in a second prompt or by a small rule or classifier trained on the factor ratings (an interesting comparison in itself).
- **Run it on exactly the same models as B1**, so the B1 vs. S1 comparison (RQ2) isolates the method.
- **Bonus output:** the factor ratings are the explanations you'll need for the user study (RQ4), so save them.

### 5. S2: fine-tuned models · GPU

Two variants, because your hardware allows both:

| Variant | What | Why |
|---|---|---|
| **S2a** | Encoder, preferably **ModernBERT-large** (long context; see hardware section): ordinal head for severity, multi-label heads for domains and drivers | Cheap and fast; the classic fine-tuning baseline |
| **S2b** | QLoRA fine-tune of an open ~7–9B LLM on the 568 training items (fits your 12 GB GPU; larger models need external compute) | Tests R-Judge's finding that fine-tuning beats prompting, on a strong model |

- **Use 5-fold cross-validation on train** for tuning, and keep dev for final comparison, so your dev numbers stay honest.
- **Watch for over-fitting:** 568 items is small, so report the spread across folds, not just the mean.

### 6. Sampling and confidence · GPU, cheap once B1/S1 run

- Sample each item 5–10 times at a higher temperature.
- Use the **majority or mean** as a more stable prediction, and the **spread** as a confidence measure.
- **Why it matters:** this gives you calibration numbers (ECE) for severity, and a confidence signal the escalation policy can use: "ask the human when the judge is unsure".

### 7. S4: escalation policy for Subtask 3 · CPU, on top of the above

- Map predicted severity, confidence and the published illustrative policy to proceed / ask / block.
- Compare simple thresholds with an LLM that reads the policy directly.
- **Score with Macro F1** on dev (if dev has intervention labels) and against the official Subtask 3 baselines: majority **0.175**, best open-weight LLM **0.666**.

### 8. Analysis · CPU

- **Results table:** B0, B1, S1, S2a, S2b (and S3 if built) × four subtasks, with bootstrap confidence intervals.
- **Error analysis:** where do B1 and S1 disagree? Do the three slide examples come out as expected? This replaces the hand-worked ratings on the 2 Dec slide with real outputs.
- **DevOps slice vs. the rest** (first look at RQ3).

---

## What can't be done yet

| Item | When |
|---|---|
| Official test-set predictions | From 10 Jan, when the test data is released |
| DevOps transfer set (new trajectories) | Needs writing and annotating first (planned mid-December to early January) |
| User study | Needs materials, possibly ethics approval, and participants (January) |

## Good practice for every run

- **Log everything:** model name and version, prompt version, temperature, seed, date. Save raw outputs, not just scores, so the report and error analysis can be rebuilt.
- **Don't tune on dev.** Choose prompts and hyperparameters with cross-validation on train, then report dev once per final system.
- **Codabench:** check its submission limits per day or in total before submitting many variants.
