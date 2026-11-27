# AgentRisk Code: B0, B1, S1 and S2a

*M13 project · SemEval-2027 Task 10 · written 27 Nov 2026*

Code for the first experiments in `experiments/compute-plan.md`, set up for your laptop (RTX 4080 Laptop, 12 GB VRAM, 32 GB RAM).

## What's here

| File | What it does |
|---|---|
| `scripts/download_data.py` | Downloads train/dev and the scenario manifest into `data/` |
| `scripts/b0_majority.py` | **B0** majority baseline |
| `scripts/run_llm_judge.py` | **B1** (holistic) and **S1** (factor-based) LLM judges, via any OpenAI-compatible server |
| `scripts/finetune_modernbert.py` | **S2a** ModernBERT-large fine-tuning, 5-fold grouped CV, ensemble predictions |
| `scripts/evaluate.py` | Official metrics, 95 % bootstrap confidence intervals, paired significance test |
| `scripts/make_submission.py` | Validates predictions and builds the Codabench ZIP |
| `agentrisk/` | Shared code: labels, data loading, metrics, prompts, LLM client, aggregation |
| `tests/test_core.py` | Unit tests for parsing, label repair, metrics and aggregation |
| `tests/test_scripts.py` | End-to-end check that `evaluate.py` runs on a non-UTF-8 (Windows cp1252) console |

## What has been tested (27 Nov)

- ✅ **B0 reproduces the official majority baseline exactly** on dev: MAE **1.294**, Jaccard **0.455** for both 2a and 2b. So data loading and metrics match the organisers'.
- ✅ The B0 output passes validation and is zipped in the Codabench format (`predictions.jsonl` at the ZIP root).
- ✅ B1 and S1 run end to end, against a mock backend and against a local stand-in OpenAI-compatible server, including resume, sampling and aggregation, and the "disable thinking" option.
- ✅ The fine-tuning script runs end to end on a tiny random model on CPU: grouped folds, threshold tuning, ensemble predictions, valid submission. Left-truncation keeps the final action.
- ✅ Unit tests pass.
- ✅ **Re-checked on the Windows laptop (Python 3.13):** tests pass, B0 again gives MAE 1.294 and Jaccard 0.455 / 0.455, and B1/S1 run with the mock backend. Fixed a crash in `evaluate.py` on Windows consoles (the ↓/↑ in the table header).
- ℹ️ The public dev split has **no intervention labels** (same columns as train), so Subtask 3 can't be scored locally on dev yet.
- ❌ **Not yet run with a real LLM or real ModernBERT on a GPU.** The batch size and maximum length for 12 GB are estimates, so see "If you run out of GPU memory" below.

---

## 1. Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/WSL: source .venv/bin/activate
pip install -r requirements.txt
```

Install **PyTorch with CUDA** following the selector on pytorch.org before the rest, so the GPU build is used. Check it with:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**Windows note:** vLLM runs on Linux, so on Windows use **WSL2** for vLLM, or use **Ollama** / **llama.cpp**, which run natively. Everything else here runs on Windows directly.

## 2. Data and B0 (CPU, a few minutes)

```bash
python scripts/download_data.py --data-dir data
python tests/test_core.py
python tests/test_scripts.py
python scripts/b0_majority.py --data-dir data --split dev --out runs/b0/dev.jsonl
python scripts/evaluate.py --data-dir data runs/b0/dev.jsonl
```

**Expected:** ST1 MAE 1.294 and ST2a/2b Jaccard 0.455. If you get these numbers, everything is wired up correctly.

**First Codabench submission** (tests the upload process):

```bash
python scripts/make_submission.py runs/b0/dev.jsonl --data-dir data --out submissions/b0_dev.zip
```

Then also check it with the official evaluator from the Codabench **Starting Kit** (Get Started → Files): `python3 evaluate.py submissions/b0_dev.zip --task 1`. Upload the same ZIP to the 1, 2a and 2b competitions; each one reads only its own field.

## 3. B1 and S1 with a local model

### Start a model server (one terminal)

Use a **4-bit 7–9B instruction model**. The organisers' Qwen3.5-9B baseline is the best choice, because its official dev score (MAE 0.608) lets you compare. Pick a 4-bit build of that model (AWQ/GPTQ for vLLM, GGUF for Ollama/llama.cpp) if one is available; otherwise a similar 7–9B instruct model.

```bash
# Option A: vLLM (Linux/WSL2)
vllm serve <model-id-or-local-path> --max-model-len 8192 --gpu-memory-utilization 0.90
#   -> base URL http://localhost:8000/v1 (the scripts' default)

# Option B: Ollama
ollama pull <model-tag>
#   -> base URL http://localhost:11434/v1, use --base-url and --model <model-tag>
```

### Run the judges (second terminal)

```bash
# quick test on 5 examples first
python scripts/run_llm_judge.py --method b1 --model <model-id> --tag qwen9b --limit 5

# full dev set
python scripts/run_llm_judge.py --method b1 --model <model-id> --tag qwen9b
python scripts/run_llm_judge.py --method s1 --model <model-id> --tag qwen9b

python scripts/evaluate.py --data-dir data runs/b0/dev.jsonl runs/b1/qwen9b/dev.jsonl runs/s1/qwen9b/dev.jsonl
# is S1 significantly better than B1? (paired bootstrap)
python scripts/evaluate.py --data-dir data runs/b1/qwen9b/dev.jsonl runs/s1/qwen9b/dev.jsonl --compare runs/s1/qwen9b/dev.jsonl runs/b1/qwen9b/dev.jsonl
```

Useful options:

| Option | When |
|---|---|
| `--disable-thinking` | Qwen3-style "thinking" models on vLLM: faster, shorter answers. Leave it off to let the model reason (then raise `--max-tokens` to about 4096). `<think>` blocks are stripped either way. |
| `--n-samples 5` | Several samples per example (temperature 0.7): median severity, majority labels, and `severity_std` as a confidence signal |
| `--workers 4` | Parallel requests. vLLM handles 4–8 well; Ollama may need 1–2 |
| `--json-mode` | If the server supports `response_format=json_object` and answers often fail to parse |
| `--model mock` | Test the pipeline without any model |

**If the run stops** (crash, laptop sleeps), run the same command again. Finished examples are skipped.

**What gets saved** in `runs/<method>/<tag>/`:

- `raw.jsonl`: every call, with the raw answer.
- `dev.jsonl`: the final predictions. **For S1 this includes the factor ratings**, which are the explanations for the user study.
- `run_info.json`: the settings, for the report.

## 4. S2a: ModernBERT fine-tuning (GPU)

```bash
python scripts/finetune_modernbert.py --data-dir data --tag modernbert-large
python scripts/evaluate.py --data-dir data runs/s2a/modernbert-large/dev.jsonl
```

This trains 5 fold models on train, grouped by scenario, and prints validation scores per epoch. It then writes:

- `cv_metrics.json`: the cross-validated score on train.
- `dev.jsonl`: dev predictions from the 5-model ensemble.

Dev is never used for training. Install `flash-attn` for extra speed if your setup supports it (optional).

**If you run out of GPU memory,** try these one at a time:

1. `--max-len 2048` (the median trajectory is about 850 tokens, so few are cut).
2. `--batch-size 1 --grad-accum 16`.
3. `--model answerdotai/ModernBERT-base` (smaller model).

## 5. Good practice

- **Don't tune on dev.** Pick prompts and settings with a few training examples or cross-validation; report dev once per final system.
- **The dev leaderboard on Codabench isn't a fair comparison.** Dev labels are public, so some teams' dev scores may be over-fitted. The ranking that counts uses the hidden test set in January.
- **Codabench limits:** 30 submissions per day, 1,000 in total, per competition.
- **Before the January test phase:** run `--split test` once the test data is out (put it in `data/test.jsonl`).
- **Prompts:** if you change a prompt in `agentrisk/prompts.py`, bump `PROMPT_VERSION` so your runs stay traceable.

## Known gaps

- **Prompt definitions:** the prompts list the harm-domain and risk-driver **names only**, because the dataset card gives no longer definitions. If the task website or the Starting Kit has fuller definitions, add them to `agentrisk/prompts.py`.
- **Subtask 3 (S4):** the escalation policy isn't in this batch. It's next, once the illustrative policy text is copied from the task website.
- **S2b (QLoRA fine-tune of a 7–9B LLM):** also next.
