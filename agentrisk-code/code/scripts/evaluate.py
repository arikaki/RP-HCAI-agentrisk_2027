"""Score one or more prediction files against gold labels.

    python scripts/evaluate.py --data-dir data --split dev runs/b0/dev.jsonl runs/b1/qwen9b/dev.jsonl

Prints a Markdown results table (with 95% bootstrap confidence intervals).
Options:
  --ids FILE      only score the example IDs listed in FILE (one per line),
                  e.g. the DevOps slice for RQ3.
  --compare A B   paired bootstrap test: is A better than B on each subtask?
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentrisk.data import load_split, read_jsonl  # noqa: E402
from agentrisk.metrics import mae, mean_jaccard, paired_bootstrap_pvalue, score_all  # noqa: E402


def fmt_ci(value: float, ci) -> str:
    return f"{value:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def main() -> None:
    # The table uses ↓/↑; Windows consoles default to cp1252, which can't encode them.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", nargs="+", help="prediction JSONL files")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--ids", help="file with example IDs to restrict scoring to")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"),
                    help="paired bootstrap test between two prediction files")
    ap.add_argument("--partial", action="store_true",
                    help="score only examples that every file has predictions for (quick tests)")
    args = ap.parse_args()

    gold = load_split(args.split, args.data_dir)
    if args.ids:
        keep = {l.strip() for l in open(args.ids) if l.strip()}
        gold = [r for r in gold if r["example_id"] in keep]
    if args.partial:
        common = set.intersection(*[{r["example_id"] for r in read_jsonl(p)} for p in args.preds])
        gold = [r for r in gold if r["example_id"] in common]
        print("NOTE: partial scoring - not comparable with full dev-set numbers")
    print(f"Scoring on {len(gold)} {args.split} examples\n")

    print("| System | ST1 MAE ↓ | ST1 QWK ↑ | ST1 Acc | ST2a Jaccard ↑ | ST2b Jaccard ↑ |")
    print("|---|---|---|---|---|---|")
    for path in args.preds:
        s = score_all(gold, read_jsonl(path))
        cells = [
            fmt_ci(s["st1_mae"], s["st1_mae_ci"]) if "st1_mae" in s else "–",
            f"{s['st1_qwk']:.3f}" if "st1_qwk" in s else "–",
            f"{s['st1_acc']:.3f}" if "st1_acc" in s else "–",
            fmt_ci(s["st2a_jaccard"], s["st2a_jaccard_ci"]) if "st2a_jaccard" in s else "–",
            fmt_ci(s["st2b_jaccard"], s["st2b_jaccard_ci"]) if "st2b_jaccard" in s else "–",
        ]
        print(f"| {path} | " + " | ".join(cells) + " |")

    if args.compare:
        a_rows = {r["example_id"]: r for r in read_jsonl(args.compare[0])}
        b_rows = {r["example_id"]: r for r in read_jsonl(args.compare[1])}
        ids = [r["example_id"] for r in gold]
        print(f"\nPaired bootstrap: is {args.compare[0]} better than {args.compare[1]}?")
        g = [r["risk_severity"] for r in gold]
        p = paired_bootstrap_pvalue(mae, g, [a_rows[i]["risk_severity"] for i in ids],
                                    [b_rows[i]["risk_severity"] for i in ids], higher_is_better=False)
        print(f"  ST1 MAE: p = {p:.3f}")
        for key, name in (("risk_domains", "ST2a"), ("risk_drivers", "ST2b")):
            g = [r[key] for r in gold]
            p = paired_bootstrap_pvalue(mean_jaccard, g, [a_rows[i][key] for i in ids],
                                        [b_rows[i][key] for i in ids], higher_is_better=True)
            print(f"  {name} Jaccard: p = {p:.3f}")
        print("  (small p = A is better; p < 0.05 is the usual threshold)")


if __name__ == "__main__":
    main()
