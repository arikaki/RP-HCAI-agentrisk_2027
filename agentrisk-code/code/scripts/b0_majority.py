"""B0: majority-class baseline.

Predicts the most frequent training severity and the most frequent training
label set for domains and drivers, for every example. On dev it reproduces the
official "Majority" baseline (ST1 MAE 1.294, ST2a/2b Jaccard 0.455).

    python scripts/b0_majority.py --data-dir data --split dev --out runs/b0/dev.jsonl
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentrisk.data import load_split, write_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--split", default="dev", help="split to predict (dev or test)")
    ap.add_argument("--out", default="runs/b0/dev.jsonl")
    args = ap.parse_args()

    train = load_split("train", args.data_dir)
    target = load_split(args.split, args.data_dir)

    sev = Counter(r["risk_severity"] for r in train).most_common(1)[0][0]
    dom = Counter(tuple(sorted(r["risk_domains"])) for r in train).most_common(1)[0][0]
    drv = Counter(tuple(sorted(r["risk_drivers"])) for r in train).most_common(1)[0][0]
    print(f"majority severity={sev}, domains={list(dom)}, drivers={list(drv)}")

    preds = [{"example_id": r["example_id"], "risk_severity": sev,
              "risk_domains": list(dom), "risk_drivers": list(drv)} for r in target]
    write_jsonl(args.out, preds)
    print(f"wrote {len(preds)} predictions -> {args.out}")


if __name__ == "__main__":
    main()
