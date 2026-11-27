"""Run the B1 (holistic) or S1 (factor-based) LLM judge.

Needs an OpenAI-compatible server. Examples:

  # vLLM serving a 4-bit 9B model on your GPU (separate terminal):
  #   vllm serve <model-id-or-path> --max-model-len 8192 --gpu-memory-utilization 0.90
  python scripts/run_llm_judge.py --method b1 --model <model-id> --tag qwen9b
  python scripts/run_llm_judge.py --method s1 --model <model-id> --tag qwen9b

  # Ollama: --base-url http://localhost:11434/v1 --model <ollama-model-name>
  # Hosted API: --base-url https://openrouter.ai/api/v1 --api-key $OPENROUTER_API_KEY
  # Offline test of the whole pipeline: --model mock

Outputs in runs/<method>/<tag>/:
  raw.jsonl       every call: raw response, parsed answer, factors (S1), timing
  <split>.jsonl   one final prediction per example (aggregated if --n-samples > 1)
  run_info.json   all settings, for the report
Re-running the same command resumes: finished examples are skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentrisk.aggregate import aggregate  # noqa: E402
from agentrisk.data import load_split, read_jsonl, write_jsonl  # noqa: E402
from agentrisk.labels import RISK_FACTORS, normalize_severity, repair_label_set  # noqa: E402
from agentrisk.llm import ChatClient, extract_json  # noqa: E402
from agentrisk.prompts import (PROMPT_VERSION, b1_messages, s1_decision_messages,  # noqa: E402
                               s1_factor_messages)


def parse_labels(obj: dict | None) -> tuple[dict, bool, bool]:
    """Turn a parsed JSON answer into valid labels. Returns (labels, parse_ok, repaired)."""
    if not obj:
        return {"risk_severity": 1, "risk_domains": ["No Material Harm"],
                "risk_drivers": ["No Material Risk"]}, False, True
    sev = normalize_severity(obj.get("risk_severity"))
    repaired = sev is None
    sev = sev or 1
    doms, r1 = repair_label_set(obj.get("risk_domains"), "domains")
    drvs, r2 = repair_label_set(obj.get("risk_drivers"), "drivers")
    return {"risk_severity": sev, "risk_domains": doms, "risk_drivers": drvs,
            "rationale": obj.get("rationale", "")}, True, repaired or r1 or r2


def clean_factors(obj: dict | None) -> dict | None:
    if not obj:
        return None
    out = {}
    for key in RISK_FACTORS:
        v = obj.get(key)
        if isinstance(v, dict):
            rating = normalize_severity(v.get("rating"))
            out[key] = {"rating": rating, "reason": str(v.get("reason", ""))}
        elif normalize_severity(v) is not None:
            out[key] = {"rating": normalize_severity(v), "reason": ""}
    return out if len(out) == len(RISK_FACTORS) and all(f["rating"] for f in out.values()) else None


def judge_one(client: ChatClient, method: str, row: dict, temperature: float) -> dict:
    t0 = time.time()
    traj = row["formatted_interaction"]
    rec: dict = {"example_id": row["example_id"]}
    if method == "b1":
        raw = client.chat(b1_messages(traj), temperature=temperature)
        obj = extract_json(raw)
        if obj is None:  # one retry on unparseable output
            raw = client.chat(b1_messages(traj), temperature=max(temperature, 0.3))
            obj = extract_json(raw)
        rec["raw"] = raw
    else:
        raw_f = client.chat(s1_factor_messages(traj), temperature=temperature)
        factors = clean_factors(extract_json(raw_f))
        if factors is None:
            raw_f = client.chat(s1_factor_messages(traj), temperature=max(temperature, 0.3))
            factors = clean_factors(extract_json(raw_f))
        rec["raw_factors"] = raw_f
        rec["factors"] = factors
        raw = client.chat(s1_decision_messages(traj, factors or {"note": "factor rating failed"}),
                          temperature=temperature)
        obj = extract_json(raw)
        if obj is None:
            raw = client.chat(s1_decision_messages(traj, factors or {}), temperature=max(temperature, 0.3))
            obj = extract_json(raw)
        rec["raw"] = raw
        rec["factors_ok"] = factors is not None
    labels, ok, repaired = parse_labels(obj)
    rec.update(labels)
    rec["parse_ok"] = ok
    rec["repaired"] = repaired
    rec["seconds"] = round(time.time() - t0, 2)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=["b1", "s1"], required=True)
    ap.add_argument("--model", required=True, help="model name as the server knows it, or 'mock'")
    ap.add_argument("--tag", help="short name for the output folder (default: model name)")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--ids", help="only run the example IDs listed in this file")
    ap.add_argument("--limit", type=int, help="only the first N examples (quick test)")
    ap.add_argument("--n-samples", type=int, default=1, help=">1 samples per example for confidence")
    ap.add_argument("--temperature", type=float, help="default 0 for 1 sample, 0.7 for several")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=4, help="parallel requests")
    ap.add_argument("--disable-thinking", action="store_true",
                    help="turn off <think> reasoning for Qwen3-style models (vLLM/SGLang)")
    ap.add_argument("--json-mode", action="store_true", help="request response_format=json_object")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--runs-dir", default="runs")
    args = ap.parse_args()

    temperature = args.temperature if args.temperature is not None else (0.0 if args.n_samples == 1 else 0.7)
    tag = args.tag or args.model.replace("/", "_")
    out_dir = Path(args.runs_dir) / args.method / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw.jsonl"

    rows = load_split(args.split, args.data_dir)
    if args.ids:
        keep = {l.strip() for l in open(args.ids) if l.strip()}
        rows = [r for r in rows if r["example_id"] in keep]
    if args.limit:
        rows = rows[: args.limit]

    info = {k: v for k, v in vars(args).items() if k != "api_key"}
    info.update(temperature=temperature, prompt_version=PROMPT_VERSION,
                started=datetime.now(timezone.utc).isoformat(), n_examples=len(rows))
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2))

    done = set()
    if raw_path.exists():
        for r in read_jsonl(raw_path):
            if r.get("split") == args.split:
                done.add((r["example_id"], r["sample"]))
    jobs = [(row, s) for row in rows for s in range(args.n_samples) if (row["example_id"], s) not in done]
    print(f"{args.method} on {len(rows)} {args.split} examples x {args.n_samples} samples; "
          f"{len(done)} already done, {len(jobs)} to run -> {out_dir}")

    client = ChatClient(args.model, base_url=args.base_url, api_key=args.api_key,
                        temperature=temperature, max_tokens=args.max_tokens,
                        disable_thinking=args.disable_thinking, json_mode=args.json_mode,
                        seed=args.seed if args.n_samples == 1 else None)
    lock = threading.Lock()

    def work(row: dict, sample: int) -> dict:
        rec = judge_one(client, args.method, row, temperature)
        rec.update(sample=sample, split=args.split, method=args.method, model=args.model,
                   prompt_version=PROMPT_VERSION, temperature=temperature)
        with lock, open(raw_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(work, row, s) for row, s in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                rec = fut.result()
                failures += not rec["parse_ok"]
            except Exception as exc:
                failures += 1
                print(f"  error: {exc}")
            if i % 10 == 0 or i == len(futures):
                print(f"  {i}/{len(futures)} done")

    # Build final predictions from everything in raw.jsonl for this split.
    by_id: dict[str, list[dict]] = {}
    for r in read_jsonl(raw_path):
        if r.get("split") == args.split and r["sample"] < args.n_samples:
            by_id.setdefault(r["example_id"], []).append(r)
    preds, n_bad, n_rep = [], 0, 0
    for row in rows:
        samples = sorted(by_id.get(row["example_id"], []), key=lambda r: r["sample"])
        if not samples:
            print(f"  missing: {row['example_id']} (re-run the command to retry)")
            continue
        n_bad += sum(not s["parse_ok"] for s in samples)
        n_rep += sum(s["repaired"] for s in samples)
        if len(samples) == 1:
            s = samples[0]
            pred = {k: s[k] for k in ("risk_severity", "risk_domains", "risk_drivers")}
        else:
            pred = aggregate(samples)
        pred["example_id"] = row["example_id"]
        if args.method == "s1":
            pred["factors"] = samples[0].get("factors")
        preds.append(pred)
    pred_path = out_dir / f"{args.split}.jsonl"
    write_jsonl(pred_path, preds)
    print(f"wrote {len(preds)} predictions -> {pred_path}")
    print(f"unparseable answers: {n_bad} (fell back to severity 1 / No Material ...); "
          f"answers needing label repair: {n_rep}")
    if len(preds) < len(rows):
        print("Some examples are missing - re-run the same command to fill them in.")


if __name__ == "__main__":
    main()
