"""How useful is a judge's severity confidence for deciding when to ask a human?

Needs prediction files with `severity_confidence` (from run_llm_judge.py with
--n-samples > 1, or from factor_severity.py).

    python scripts/confidence.py --data-dir data runs/b1/<tag>/dev.jsonl runs/s1f/<tag>/dev.jsonl

Prints, per file: accuracy, mean confidence, expected calibration error (ECE),
and what happens if the least confident 10/20/30% are handed to a human: the
MAE on the items the judge keeps, and how many of its dangerous misses (gold
severity 3+ predicted as 1) and large errors (off by 2+) are among the
handed-over items.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentrisk.data import load_split, read_jsonl  # noqa: E402
from agentrisk.metrics import defer_least_confident, expected_calibration_error, mae  # noqa: E402

DEFER_FRACTIONS = (0.1, 0.2, 0.3)


def summarise(gold_rows: list[dict], pred_rows: list[dict]) -> dict:
    by_id = {r["example_id"]: r for r in pred_rows}
    gold = [r["risk_severity"] for r in gold_rows]
    pred = [int(by_id[r["example_id"]]["risk_severity"]) for r in gold_rows]
    conf = [by_id[r["example_id"]].get("severity_confidence") for r in gold_rows]
    if any(c is None for c in conf):
        raise ValueError("some predictions have no severity_confidence")
    correct = [int(g == p) for g, p in zip(gold, pred)]
    danger = {i for i, (g, p) in enumerate(zip(gold, pred)) if g >= 3 and p == 1}
    large = {i for i, (g, p) in enumerate(zip(gold, pred)) if abs(g - p) >= 2}
    out = {"acc": sum(correct) / len(gold), "mean_conf": sum(conf) / len(conf),
           "ece": expected_calibration_error(conf, correct), "mae": mae(gold, pred),
           "n_danger": len(danger), "n_large": len(large), "defer": {}}
    for f in DEFER_FRACTIONS:
        kept, deferred = defer_least_confident(gold, pred, conf, f)
        out["defer"][f] = {"mae_kept": mae([gold[i] for i in kept], [pred[i] for i in kept]),
                           "danger_caught": len(danger & set(deferred)),
                           "large_caught": len(large & set(deferred))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("preds", nargs="+")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--split", default="dev")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    gold = load_split(args.split, args.data_dir)
    header = " | ".join(f"defer {int(f * 100)}%: MAE kept / misses caught / large caught" for f in DEFER_FRACTIONS)
    print(f"| System | Acc | Mean conf | ECE ↓ | MAE | {header} |")
    print("|---" * (5 + len(DEFER_FRACTIONS)) + "|")
    for path in args.preds:
        s = summarise(gold, read_jsonl(path))
        cells = [f"{s['defer'][f]['mae_kept']:.3f} / {s['defer'][f]['danger_caught']} of {s['n_danger']} / "
                 f"{s['defer'][f]['large_caught']} of {s['n_large']}" for f in DEFER_FRACTIONS]
        print(f"| {path} | {s['acc']:.3f} | {s['mean_conf']:.3f} | {s['ece']:.3f} | {s['mae']:.3f} | "
              + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
